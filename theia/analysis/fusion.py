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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theia", default="results/peri-s42.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--gene", default="egfr")
    ap.add_argument("--out", default="results/fusion.json")
    a = ap.parse_args()

    arms: dict[str, list[dict]] = {"THEIA": _oof(a.theia, a.gene)}
    if os.path.exists(a.baselines):
        blob = json.load(open(a.baselines)).get("oof") or {}
        for name in ("clinical", "radiomics"):
            if blob.get(name):
                arms[name] = [r for r in blob[name]
                              if r.get(f"{a.gene}_true") in (0, 1)]
    if len(arms) < 2:
        raise SystemExit(f"need >=2 arms, found {list(arms)}; run theia.analysis.baselines")

    ids, y, folds, R = align(arms, a.gene)
    names = list(arms)
    print(f"[fusion] {len(ids)} patients ({int(y.sum())} positive) scored by all of "
          f"{names}\n")

    results = {}
    print(f"{'arm / topology':<38} {'AUC':>7}")
    for j, n in enumerate(names):
        results[n] = _auc(y, R[:, j])
        print(f"{n:<38} {results[n]:>7.3f}")

    print()
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

    for label, s in topologies.items():
        auc = _auc(y, s)
        lo, hi, p = _boot(y, s, ref)
        results[label] = {"auc": auc, "vs_best_single": auc - results[best_single],
                          "ci_lo": lo, "ci_hi": hi, "p": p}
        print(f"{label:<38} {auc:>7.3f}   vs {best_single} "
              f"{auc - results[best_single]:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")

    print(f"\n[fusion] early (feature-level) fusion: NOT COMPUTED. It needs each arm's "
          f"feature\n         matrix, and THEIA's region embeddings are not archived; "
          f"producing a row\n         from ranks would mislabel a decision-level result "
          f"as feature-level.")
    json.dump({"gene": a.gene, "n": len(ids), "n_pos": int(y.sum()),
               "arms": names, "best_single": best_single, "results": results,
               "early_fusion": "not computed; requires archived feature matrices"},
              open(a.out, "w"), indent=2)
    print(f"[fusion] wrote {a.out}")


if __name__ == "__main__":
    main()
