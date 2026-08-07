"""Baselines THEIA has to beat, scored on exactly the folds THEIA was scored on.

`diagnostics.py` already answers "is there any signal here at all". This answers
a different and harder question: *is the deep model worth it?* Three comparators,
in ascending order of how uncomfortable they are:

  clinical    age, sex, ethnicity, smoking status, pack-years. No image at all.
              EGFR mutation is strongly enriched in never-smokers and in women,
              so this is not a straw man — it is the first thing a reviewer will
              ask for, and if it matches THEIA then the imaging contributes
              nothing and the paper has no claim.
  radiomics   handcrafted first-order + shape + GLCM texture over the segmented
              lesion. What the field did before deep learning.
  combined    clinical + radiomics, the strongest conventional model available.

And the one that actually decides the paper:

  clinical + THEIA   does the network add anything ON TOP OF what a clinician
                     already knows? Reported as a paired delta with a CI.

Two design decisions that make the comparison honest:

1. Identical folds. Every baseline is fit and scored through
   `nested_kfold_indices` with the same seed and fold count as training, so each
   patient is held out by the same fold that held them out for THEIA. Comparing
   against a baseline cross-validated under its own random split would confound
   the model difference with a split difference, and at ~40 positives a split
   difference alone moves AUC by more than the effect being measured.

2. Paired bootstrap. Because the folds are identical, each patient has a
   THEIA score and a baseline score, so the two AUCs can be compared by
   resampling *patients* and recomputing both AUCs on the same resample. That
   cancels the cohort variance which dominates two independent CIs -- two
   intervals can overlap heavily while the paired difference is unambiguous.

THEIA's side of the comparison comes from `results/<run_id>.json`, which is why
that archive stores out-of-fold predictions and not just summary metrics.

Run:
  python -m theia.analysis.baselines --theia_run results/run10-headline.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from theia.analysis.diagnostics import load_rows, radiomic_features
from theia.data.dataset import nested_kfold_indices
from theia.engine.evaluate import pooled_metrics

# Columns read from the TCIA clinical sheet. Deliberately excludes %GG, tumor
# location and histology: those are read off the scan or the resection specimen,
# so folding them into a "clinical" arm would quietly give it imaging and
# pathology information and make the imaging comparison meaningless.
NUMERIC = ["Age at Histological Diagnosis", "Pack Years"]
CATEGORICAL = ["Gender", "Ethnicity", "Smoking status"]


def clinical_features(rows: list[dict], csv_path: str) -> tuple[np.ndarray, list[str]]:
    """Age, sex, ethnicity, smoking, pack-years -> a dense matrix aligned to rows.

    Missing values are median-filled (numeric) or given their own indicator
    level (categorical) rather than dropped: dropping would silently change the
    cohort between arms, and then the baselines would not be scored on the same
    patients as THEIA.
    """
    df = pd.read_csv(csv_path).set_index("Case ID", drop=False)
    ids = [r["patient_id"] for r in rows]
    missing = [p for p in ids if p not in df.index]
    if missing:
        raise KeyError(f"{len(missing)} patients are not in {csv_path} "
                       f"(first few: {missing[:5]})")
    sub = df.loc[ids]

    blocks, names = [], []
    for col in NUMERIC:
        v = pd.to_numeric(sub[col], errors="coerce")
        # Pack Years is genuinely absent for never-smokers, not unmeasured.
        # Treat it as 0 there, and median-fill only the truly unknown.
        if col == "Pack Years":
            never = sub["Smoking status"].astype(str).str.strip().str.lower() == "nonsmoker"
            v = v.where(~(v.isna() & never), 0.0)
        blocks.append(v.fillna(v.median()).to_numpy()[:, None])
        names.append(col)
        if v.isna().any():
            blocks.append(v.isna().to_numpy().astype(float)[:, None])
            names.append(f"{col}__missing")

    for col in CATEGORICAL:
        vals = sub[col].astype(str).str.strip().str.lower().fillna("unknown")
        dummies = pd.get_dummies(vals, prefix=col, dummy_na=False)
        blocks.append(dummies.to_numpy().astype(float))
        names.extend(dummies.columns.tolist())

    return np.nan_to_num(np.hstack(blocks).astype(np.float64)), names


def oof_predictions(X: np.ndarray, rows: list[dict], gene: str, cfg) -> list[dict]:
    """Fit per outer fold on exactly THEIA's training portion; score its test fold.

    The estimator sees `train + inner_val`, which is the same data THEIA had --
    THEIA fit on train and selected its checkpoint on inner_val, so both arms
    consume identical information. C is chosen by an inner cross-validation
    inside that portion, never using the outer test fold.
    """
    rows_path = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    y = np.array([int(r.get("labels", {}).get(gene.upper(), -1)) for r in rows])
    g = gene.lower()
    out: list[dict] = []

    folds = nested_kfold_indices(rows_path, cfg.split.stratify_on, cfg.split.n_folds,
                                 cfg.seed, float(cfg.split.get("inner_val_frac", 0.2)))
    for fold, (tr, va, te) in enumerate(folds):
        fit_idx = np.array([i for i in list(tr) + list(va) if y[i] != -1])
        test_idx = np.array([i for i in te if y[i] != -1])
        if test_idx.size == 0 or len(set(y[fit_idx].tolist())) < 2:
            continue
        clf = _tuned_logreg(X[fit_idx], y[fit_idx], cfg.seed)
        p = clf.predict_proba(X[test_idx])[:, 1]
        for i, prob in zip(test_idx, p):
            out.append({"patient_id": rows[i]["patient_id"], "fold": fold,
                        f"{g}_prob": float(prob), f"{g}_true": int(y[i])})
    return out


def _tuned_logreg(X: np.ndarray, y: np.ndarray, seed: int):
    """L2 logistic regression with C picked by inner CV on the fitting portion.

    A fixed C would hand the baseline a handicap the deep model does not have
    (THEIA's hyperparameters were tuned over many runs on this cohort). Giving
    the baseline its own inner tuning loop is what makes "THEIA wins" mean
    something.
    """
    best, best_auc = 1.0, -1.0
    n_pos = int(y.sum())
    n_splits = max(2, min(5, n_pos, len(y) - n_pos))
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for a, b in skf.split(X, y):
            if len(set(y[a].tolist())) < 2 or len(set(y[b].tolist())) < 2:
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=5000,
                                                 class_weight="balanced"))
            m.fit(X[a], y[a])
            scores.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        if scores and float(np.mean(scores)) > best_auc:
            best_auc, best = float(np.mean(scores)), C
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=best, max_iter=5000,
                                           class_weight="balanced"))
    clf.fit(X, y)
    return clf


def _ranked(rows: list[dict], key: str) -> dict[str, float]:
    """Within-fold rank normalisation, keyed by patient.

    Each fold is a different fitted model, so raw probabilities are not
    comparable across folds; ranks are. This mirrors what
    evaluate._rank_normalize_by_fold does for THEIA, so both sides of the
    comparison are transformed identically.
    """
    by_fold: dict[int, list[dict]] = {}
    for r in rows:
        if key in r:
            by_fold.setdefault(r.get("fold", 0), []).append(r)
    out: dict[str, float] = {}
    for recs in by_fold.values():
        p = np.array([r[key] for r in recs], dtype=float)
        ranks = p.argsort().argsort() / max(len(p) - 1, 1)
        for r, v in zip(recs, ranks):
            out[r["patient_id"]] = float(v)
    return out


def paired_delta(theia: list[dict], base: list[dict], gene: str,
                 n_boot: int = 2000, seed: int = 1337) -> dict:
    """AUC(THEIA) - AUC(baseline) on the patients both scored, with a paired CI.

    Resamples patients, not predictions, and recomputes BOTH AUCs on the same
    resample. Two independent CIs can overlap almost entirely while the paired
    difference is decisive, because most of the width of each interval is shared
    cohort variance that cancels here.
    """
    g = gene.lower()
    ta, ba = _ranked(theia, f"{g}_prob"), _ranked(base, f"{g}_prob")
    truth = {r["patient_id"]: r[f"{g}_true"] for r in theia if f"{g}_true" in r}
    ids = sorted(set(ta) & set(ba) & set(truth))
    y = np.array([truth[i] for i in ids])
    if len(ids) == 0 or len(set(y.tolist())) < 2:
        return {"n": len(ids), "delta": float("nan")}

    a = np.array([ta[i] for i in ids])
    b = np.array([ba[i] for i in ids])
    obs = float(roc_auc_score(y, a) - roc_auc_score(y, b))

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        if len(set(y[idx].tolist())) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(deltas)
    # Two-sided bootstrap p: how often does the resampled difference cross zero?
    p = float(2 * min((d <= 0).mean(), (d >= 0).mean())) if d.size else float("nan")
    return {"n": len(ids), "n_pos": int(y.sum()), "delta": obs,
            "delta_lo": float(np.percentile(d, 2.5)) if d.size else float("nan"),
            "delta_hi": float(np.percentile(d, 97.5)) if d.size else float("nan"),
            "p_two_sided": min(p, 1.0)}


def main() -> None:
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--theia_run", default=None,
                    help="results/<run_id>.json holding THEIA's out-of-fold predictions")
    ap.add_argument("--gene", default=None, help="defaults to the headline gene")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    gene = a.gene or cfg.data.get("headline_genes", ["EGFR"])[0]
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    csv_path = os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv")

    print(f"[base] {len(rows)} patients, gene={gene}", flush=True)
    Xc, names = clinical_features(rows, csv_path)
    print(f"[base] clinical features {Xc.shape}: {', '.join(names)}", flush=True)
    Xr = radiomic_features(rows)
    print(f"[base] radiomic features {Xr.shape}", flush=True)

    arms = {"clinical": Xc, "radiomics": Xr,
            "clinical+radiomics": np.hstack([Xc, Xr])}

    results: dict[str, dict] = {"gene": gene, "arms": {}}
    preds: dict[str, list[dict]] = {}
    for name, X in arms.items():
        oof = oof_predictions(X, rows, gene, cfg)
        preds[name] = oof
        m = pooled_metrics(oof, [gene], cfg.eval.bootstrap_n)
        results["arms"][name] = m
        g = gene.lower()
        print(f"[base] {name:20s} n={int(m.get(f'{g}_n',0)):4d} "
              f"AUC {m.get(f'{g}_auc', float('nan')):.3f}  "
              f"95% CI [{m.get(f'{g}_auc_lo', float('nan')):.3f}, "
              f"{m.get(f'{g}_auc_hi', float('nan')):.3f}]", flush=True)

    if a.theia_run:
        blob = json.load(open(a.theia_run))
        theia_oof = [r for f in blob["folds"] for r in (f.get("oof") or [])]
        results["theia_run_id"] = blob.get("run_id")
        results["theia_pooled"] = blob.get("pooled")
        print(f"\n[base] paired vs THEIA ({blob.get('run_id')}), "
              f"{len(theia_oof)} out-of-fold predictions:", flush=True)
        for name, oof in preds.items():
            d = paired_delta(theia_oof, oof, gene, cfg.eval.bootstrap_n, cfg.seed)
            results["arms"][name]["paired_vs_theia"] = d
            print(f"[base]   THEIA - {name:20s} dAUC {d['delta']:+.3f} "
                  f"95% CI [{d['delta_lo']:+.3f}, {d['delta_hi']:+.3f}]  "
                  f"p={d['p_two_sided']:.3f}", flush=True)

        # The claim the paper actually rests on: imaging on top of clinical.
        stacked = _stack_with_theia(preds["clinical"], theia_oof, gene)
        if stacked:
            m = pooled_metrics(stacked, [gene], cfg.eval.bootstrap_n)
            results["arms"]["clinical+theia"] = m
            g = gene.lower()
            print(f"[base] {'clinical+theia':20s} n={int(m.get(f'{g}_n',0)):4d} "
                  f"AUC {m.get(f'{g}_auc', float('nan')):.3f}  "
                  f"95% CI [{m.get(f'{g}_auc_lo', float('nan')):.3f}, "
                  f"{m.get(f'{g}_auc_hi', float('nan')):.3f}]", flush=True)
            d = paired_delta(stacked, preds["clinical"], gene, cfg.eval.bootstrap_n,
                             cfg.seed)
            results["arms"]["clinical+theia"]["paired_vs_clinical"] = d
            print(f"[base]   (clinical+theia) - clinical      dAUC {d['delta']:+.3f} "
                  f"95% CI [{d['delta_lo']:+.3f}, {d['delta_hi']:+.3f}]  "
                  f"p={d['p_two_sided']:.3f}", flush=True)

    out = a.out or os.path.join(cfg.paths.results_dir, "baselines.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n[base] wrote {out}")


def _stack_with_theia(clinical_oof: list[dict], theia_oof: list[dict],
                      gene: str) -> list[dict]:
    """Average the two within-fold rank vectors, per patient.

    A fixed 50/50 rank average rather than a fitted stacker on purpose: fitting
    a combiner on out-of-fold predictions that were themselves selected on this
    cohort would leak, and with ~40 positives the fitted weight would be noise
    anyway. An unweighted average cannot flatter the combination.
    """
    g = gene.lower()
    tr, cr = _ranked(theia_oof, f"{g}_prob"), _ranked(clinical_oof, f"{g}_prob")
    truth = {r["patient_id"]: r[f"{g}_true"] for r in theia_oof if f"{g}_true" in r}
    fold = {r["patient_id"]: r.get("fold", 0) for r in theia_oof}
    ids = sorted(set(tr) & set(cr) & set(truth))
    return [{"patient_id": i, "fold": fold.get(i, 0),
             f"{g}_prob": 0.5 * (tr[i] + cr[i]), f"{g}_true": truth[i]} for i in ids]


if __name__ == "__main__":
    main()
