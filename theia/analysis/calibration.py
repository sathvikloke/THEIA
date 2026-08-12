"""Is the model's *probability* right, not just its ranking?

Every headline in this project is an AUC, which is invariant to any monotone
transform of the scores. A model can rank patients perfectly and still assert
80% risk for a population where the true rate is 25%. CLAIM asks for calibration
separately for exactly that reason, and the analysis plan commits to reporting it.

Nothing here needs retraining. The raw per-patient softmax probabilities are
archived in `results/*.json` at `folds[i]['oof'][j]['egfr_prob']`; the within-fold
rank normalisation that makes the folds comparable happens later, inside
`pooled_metrics`, and does not touch what is stored.

Four numbers, and the reason each is here:

* **calibration slope** — logistic regression of the outcome on the predicted
  logit. 1.0 is perfect; below 1.0 means the predictions are too extreme, which
  is the classic signature of overfitting and is what a 40-positive cohort should
  be expected to produce.
* **calibration intercept** (calibration-in-the-large) — fitted with the slope
  fixed at 1, so it isolates systematic over- or under-prediction from spread.
* **Brier score**, with its Murphy decomposition into reliability, resolution and
  uncertainty. Reliability is the part calibration can fix; resolution is the part
  only a better model can.
* **O:E ratio** — observed positives over expected. The blunt instrument, and the
  one a clinical reader will look at first.

A caveat that belongs with these numbers rather than in a footnote: each fold is
a separately fitted model, so pooling raw probabilities across folds mixes five
calibration curves. Per-fold calibration is reported alongside the pooled figure
so that mixing is visible instead of hidden.

Run: python -m theia.analysis.calibration
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, stdev

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Cox calibration: slope from a free fit, intercept with the slope pinned at 1."""
    from sklearn.linear_model import LogisticRegression

    z = _logit(p).reshape(-1, 1)
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan")
    slope = float(LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
                  .fit(z, y).coef_[0][0])
    # Calibration-in-the-large: the MLE offset `a` for logit(p) + a is the root of
    # sum(y - sigmoid(logit(p) + a)) = 0. That sum is strictly decreasing in a, so
    # bisection is exact to machine precision here and has none of the
    # convergence failure modes of running the optimiser on a constant design.
    lo, hi = -20.0, 20.0
    zf = z.ravel()
    for _ in range(200):
        mid = (lo + hi) / 2
        s = float(np.sum(y - 1 / (1 + np.exp(-(zf + mid)))))
        if s > 0:
            lo = mid
        else:
            hi = mid
    return slope, float((lo + hi) / 2)


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score split into reliability, resolution and uncertainty (Murphy).

    brier = reliability - resolution + uncertainty, up to binning error.
    """
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    unc = base * (1 - base)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        nk = int(m.sum())
        if nk == 0:
            continue
        pk, ok = float(p[m].mean()), float(y[m].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2
    n = len(y)
    return {"brier": brier, "reliability": rel / n, "resolution": res / n,
            "uncertainty": unc, "base_rate": base}


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    slope, intercept = calibration_slope_intercept(y, p)
    out = {"n": int(len(y)), "n_pos": int(y.sum()),
           "calibration_slope": slope, "calibration_intercept": intercept,
           "observed": float(y.sum()), "expected": float(p.sum()),
           "oe_ratio": float(y.sum() / max(p.sum(), EPS)),
           "mean_predicted": float(p.mean())}
    out.update(brier_decomposition(y, p, n_bins))
    return out


def curve(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Equal-count bins, not equal-width: at n=153 equal-width bins go empty."""
    order = np.argsort(p)
    pts = []
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        pts.append({"n": int(chunk.size), "mean_predicted": float(p[chunk].mean()),
                    "observed_rate": float(y[chunk].mean())})
    return pts


def load(path: str, gene: str = "egfr", keep: set[str] | None = None):
    d = json.load(open(path))
    recs = [r for f in d.get("folds", []) for r in (f.get("oof") or [])
            if r.get(f"{gene}_true") in (0, 1) and r.get(f"{gene}_prob") is not None
            and (keep is None or r["patient_id"] in keep)]
    y = np.array([r[f"{gene}_true"] for r in recs])
    p = np.array([r[f"{gene}_prob"] for r in recs], dtype=float)
    folds = np.array([r.get("fold", 0) for r in recs])
    return y, p, folds


def main() -> None:
    import pandas as pd

    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default="results/base-s1337-rerun.json,"
                                      "results/ms-s7.json,results/ms-s42.json")
    ap.add_argument("--gene", default="egfr")
    ap.add_argument("--out", default="results/calibration.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rows = [json.loads(l) for l in open(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))]
    has_mask = {r["patient_id"] for r in rows if r.get("has_mask")}
    clin = pd.read_csv(os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv"))
    adeno = {p for p, h in zip(clin["Case ID"], clin["Histology "].astype(str).str.strip())
             if h == "Adenocarcinoma"}

    subsets = {"full cohort": None, "segmented adenocarcinoma": adeno & has_mask}
    runs = [p for p in a.runs.split(",") if os.path.exists(p)]
    out: dict = {"runs": runs, "subsets": {}}

    for label, keep in subsets.items():
        print(f"\n=== {label} ===")
        print(f"{'run':<28} {'n':>4} {'slope':>7} {'intcpt':>7} {'Brier':>7} "
              f"{'relia':>7} {'resol':>7} {'O:E':>6}")
        per = []
        for path in runs:
            y, p, _ = load(path, a.gene, keep)
            r = calibration_report(y, p)
            r["run"] = os.path.basename(path)
            per.append(r)
            print(f"{r['run']:<28} {r['n']:>4} {r['calibration_slope']:>7.3f} "
                  f"{r['calibration_intercept']:>7.3f} {r['brier']:>7.3f} "
                  f"{r['reliability']:>7.4f} {r['resolution']:>7.4f} {r['oe_ratio']:>6.2f}")
        sl = [x["calibration_slope"] for x in per if not np.isnan(x["calibration_slope"])]
        if sl:
            print(f"{'mean':<28} {'':>4} {mean(sl):>7.3f} "
                  f"{mean(x['calibration_intercept'] for x in per):>7.3f} "
                  f"{mean(x['brier'] for x in per):>7.3f}")
            if len(sl) > 1:
                print(f"  slope sd {stdev(sl):.3f}")
        # Curve from the first run, which is the sealed-artifact candidate.
        y, p, _ = load(runs[0], a.gene, keep)
        out["subsets"][label] = {"per_run": per, "curve_first_run": curve(y, p)}

    print("\n[calib] reading: slope < 1 means predictions are too extreme, which is "
          "what a 40-positive cohort\n        should produce. Reliability is the part "
          "recalibration can fix; resolution is the part\n        only a better model can.")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[calib] wrote {a.out}")


if __name__ == "__main__":
    main()
