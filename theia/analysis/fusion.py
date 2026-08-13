"""How should the arms be combined? The ablation a reviewer will look for.

The strongest published EGFR-from-CT result (Acad Radiol 2025, n=826, external
AUC 0.889, doi 10.1016/j.acra.2025.04.029) does not come from a better backbone.
It comes from fusing radiomics, deep features and clinical variables, and its
central table compares fusion *topologies* — feature-level concatenation against
hard voting, soft voting and stacking. `baselines._stack_with_theia` currently
does exactly one of these, an unweighted 50/50 rank average, so there is nothing
to compare it against.

Everything here is decision-level, and that is a deliberate limit rather than an
oversight. Feature-level (early) fusion needs the arms' feature matrices; THEIA's
are region embeddings that are not archived, so producing an "early fusion" row
would mean retraining, and inventing one from ranks would be mislabelling a
decision-level result. It is reported as not computed.

The leak that has to be avoided. Every arm's predictions here are already
out-of-fold, so a stacker fitted on all of them and then evaluated on the same
patients would be scoring itself on its own training data. The stacker is
therefore **cross-fitted**: its weights are fitted on the OOF predictions of the
other outer folds and applied to the held-out one, reusing the same fold
assignment the base models used. That keeps the comparison honest, and it is why
the fitted stacker does not automatically beat the fixed average.

WHICH RUN. This defaulted to `results/peri-s42.json` -- the *peritumoral variant*,
seed 42 -- and so the fusion table published in the manuscript was computed on a
model the paper is not about. The symptom was an internally inconsistent row: the
table printed the headline AUC 0.618 beside a difference of -0.108, which is
0.656 - 0.764, the peritumoral arm's number. The conclusion survived the
correction and in fact strengthened, but that was luck.

The default is now the canonical run set, `--theia` takes a comma-separated list,
and every arm is reported as an across-seed mean +/- sd. `results/CANONICAL.json`
is the single definition; `tests/test_fusion.py` asserts the output was built from
it. This is the fifth time a stale default has put a superseded or off-target run
into a reported number, which is why the assertion is a test rather than a habit.

Run: python -m theia.analysis.fusion
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _oof(path: str, gene: str = "egfr") -> list[dict]:
    d = json.load(open(path))
    if "folds" in d:
        return [r for f in d["folds"] for r in (f.get("oof") or [])
                if r.get(f"{gene}_true") in (0, 1) and r.get(f"{gene}_prob") is not None]
    return [r for r in d if r.get(f"{gene}_true") in (0, 1)]


def rank_within_fold(recs: list[dict], key: str) -> dict[str, float]:
    by_fold: dict[int, list[dict]] = {}
    for r in recs:
        by_fold.setdefault(r.get("fold", 0), []).append(r)
    out = {}
    for group in by_fold.values():
        p = np.array([r[key] for r in group], dtype=float)
        ranks = p.argsort().argsort() / max(len(p) - 1, 1)
        for r, v in zip(group, ranks):
            out[r["patient_id"]] = float(v)
    return out


def align(arms: dict[str, list[dict]], gene: str = "egfr"):
    """Patients scored by every arm -> (ids, y, fold, R [n, n_arms])."""
    ranked = {k: rank_within_fold(v, f"{gene}_prob") for k, v in arms.items()}
    truth, fold = {}, {}
    for v in arms.values():
        for r in v:
            truth[r["patient_id"]] = r[f"{gene}_true"]
            fold[r["patient_id"]] = r.get("fold", 0)
    ids = sorted(set(truth).intersection(*[set(d) for d in ranked.values()]))
    y = np.array([truth[i] for i in ids])
    f = np.array([fold[i] for i in ids])
    R = np.column_stack([[ranked[k][i] for i in ids] for k in arms])
    return ids, y, f, R


def hard_vote(R: np.ndarray) -> np.ndarray:
    """Each arm votes 1 above its own median; the score is the vote count.

    Coarse by construction -- with 2-3 arms this takes very few distinct values,
    so its AUC is depressed by ties. That is a property of hard voting, not a bug,
    and it is why the published pipelines land on soft voting instead.
    """
    return (R > np.median(R, axis=0)).sum(axis=1).astype(float)


def soft_vote(R: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(R.shape[1]) if w is None else np.asarray(w, dtype=float)
    return R @ (w / w.sum())


def cross_fitted_stack(R: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Logistic stacker whose weights never see the fold they score.

    Falls back to the unweighted average for any fold whose training portion is
    single-class, rather than emitting a constant that would silently read as
    chance.
    """
    out = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        if len(set(y[tr].tolist())) < 2:
            out[te] = soft_vote(R[te])
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(R[tr], y[tr])
        out[te] = clf.predict_proba(R[te])[:, 1]
    return out


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def _boot(y, a, b, n=5000, seed=1337):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) > 1:
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(2 * min((d <= 0).mean(), (d >= 0).mean())))


def _canonical_runs() -> list[str]:
    """The headline run set, read from its single definition."""
    return list(json.load(open("results/CANONICAL.json"))["headline_runs"])


def one_run(theia_path: str, baselines: str, gene: str, quiet: bool = False) -> dict:
    """Every arm and topology for ONE THEIA run, on the patients all arms cover."""
    arms: dict[str, list[dict]] = {"THEIA": _oof(theia_path, gene)}
    if os.path.exists(baselines):
        blob = json.load(open(baselines)).get("oof") or {}
        for name in ("clinical", "radiomics"):
            if blob.get(name):
                arms[name] = [r for r in blob[name]
                              if r.get(f"{gene}_true") in (0, 1)]
    if len(arms) < 2:
        raise SystemExit(f"need >=2 arms, found {list(arms)}; run theia.analysis.baselines")

    ids, y, folds, R = align(arms, gene)
    names = list(arms)
    if not quiet:
        print(f"[fusion] {len(ids)} patients ({int(y.sum())} positive) scored by all of "
              f"{names}\n")

    results = {}
    for j, n in enumerate(names):
        results[n] = _auc(y, R[:, j])

    best_single = max(names, key=lambda n: results[n])
    ref = R[:, names.index(best_single)]

    topologies = {
        "hard voting": hard_vote(R),
        "soft voting (equal weights)": soft_vote(R),
        "stacking (cross-fitted)": cross_fitted_stack(R, y, folds),
    }
    # Pairwise soft votes, so the table shows whether the gain needs all arms.
    for i, j in combinations(range(len(names)), 2):
        topologies[f"soft voting ({names[i]} + {names[j]})"] = soft_vote(R[:, [i, j]])

    theia_idx = names.index("THEIA")
    for label, s in topologies.items():
        auc = _auc(y, s)
        lo, hi, p = _boot(y, s, ref)
        # Also compare against THEIA alone. The manuscript previously said
        # "adding radiomics harmed the combination (-0.130)" while -0.130 was
        # measured against the CLINICAL model, not against THEIA -- so the
        # sentence described a contrast the table did not contain.
        lo_t, hi_t, p_t = _boot(y, s, R[:, theia_idx])
        results[label] = {"auc": auc, "vs_best_single": auc - results[best_single],
                          "ci_lo": lo, "ci_hi": hi, "p": p,
                          "vs_theia": auc - results["THEIA"],
                          "vs_theia_ci_lo": lo_t, "vs_theia_ci_hi": hi_t,
                          "vs_theia_p": p_t}

    return {"theia_run": os.path.basename(theia_path), "gene": gene,
            "n": len(ids), "n_pos": int(y.sum()), "arms": names,
            "best_single": best_single, "results": results,
            "early_fusion": "not computed; requires archived feature matrices"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theia", default=None,
                    help="comma-separated run JSONs; default is CANONICAL.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--gene", default="egfr")
    ap.add_argument("--out", default="results/fusion.json")
    a = ap.parse_args()

    runs = a.theia.split(",") if a.theia else _canonical_runs()
    print(f"[fusion] runs: {[os.path.basename(r) for r in runs]}")
    per_run = [one_run(r, a.baselines, a.gene, quiet=i > 0)
               for i, r in enumerate(runs)]

    labels = list(per_run[0]["results"])
    agg: dict[str, dict] = {}
    print(f"\n{'arm / topology':<38} {'mean':>7} {'sd':>7}   per-run")
    for lab in labels:
        vals = [(r["results"][lab] if not isinstance(r["results"][lab], dict)
                 else r["results"][lab]["auc"]) for r in per_run]
        entry = {"auc_mean": float(np.mean(vals)),
                 "auc_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                 "per_run": {r["theia_run"]: v for r, v in zip(per_run, vals)}}
        if isinstance(per_run[0]["results"][lab], dict):
            for key in ("vs_best_single", "p", "vs_theia", "vs_theia_p"):
                kv = [r["results"][lab][key] for r in per_run]
                entry[key + "_mean"] = float(np.mean(kv))
                entry[key + "_per_run"] = kv
        agg[lab] = entry
        print(f"{lab:<38} {entry['auc_mean']:>7.3f} {entry['auc_sd']:>7.3f}   "
              + " ".join(f"{v:.3f}" for v in vals))

    print(f"\n[fusion] early (feature-level) fusion: NOT COMPUTED. It needs each arm's "
          f"feature\n         matrix, and THEIA's region embeddings are not archived; "
          f"producing a row\n         from ranks would mislabel a decision-level result "
          f"as feature-level.")
    json.dump({"gene": a.gene, "runs": [os.path.basename(r) for r in runs],
               "from_canonical": a.theia is None,
               "n": per_run[0]["n"], "n_pos": per_run[0]["n_pos"],
               "arms": per_run[0]["arms"], "best_single": per_run[0]["best_single"],
               "aggregate": agg, "per_run": per_run,
               "early_fusion": "not computed; requires archived feature matrices"},
              open(a.out, "w"), indent=2)
    print(f"[fusion] wrote {a.out}")


if __name__ == "__main__":
    main()
