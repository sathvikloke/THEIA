"""How many patients does the next cohort need?

Written to answer a concrete question before data is requested, rather than
after: at what sample size could this study detect the effect it is looking for?

The design is fixed by what has already been measured. THEIA does not beat the
clinical model, so the claim worth powering is INCREMENTAL: does imaging add to
five chart variables? The estimator is the paired difference in AUC between
(clinical + imaging) and clinical alone, which is what theia.analysis.incremental
computes.

Method: resample the observed cohort with replacement to the target size,
impose a specified true increment by shifting the combined arm's scores, and
count how often the paired bootstrap CI excludes zero. That inherits the real
label prevalence (26% EGFR-mutant), the real clinical-model strength (~0.78),
and the real correlation between the two arms, none of which a closed-form
power formula would capture.

Read the output as OPTIMISTIC, for a reason worth stating plainly. The
simulated imaging arm is independent of the clinical arm given the labels. A
real imaging model is not: ground-glass morphology tracks never-smoker status,
which is most of what the clinical model knows. To the extent that holds, a real
imaging arm of the same standalone AUC delivers a SMALLER increment than
simulated here, and the required n is larger. It also assumes the external
cohort resembles this one in prevalence and clinical-model strength, and this
cohort's own AUC moves +/-0.041 between reruns.

Sanity check the table against reality before trusting it: at n=153 with an
imaging arm at THEIA's measured strength it reports power 0.15, which is
consistent with the observed study detecting nothing.

Run: python -m theia.analysis.power
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def _imaging_arm(y: np.ndarray, auc: float, rng: np.random.Generator) -> np.ndarray:
    """Scores for a hypothetical imaging model of a given AUC.

    Binormal model: positives ~ N(mu, 1), negatives ~ N(0, 1), with
    mu = sqrt(2) * Phi^-1(AUC), which is the standard relation between AUC and
    separation. Crucially the arm is generated INDEPENDENTLY of the clinical
    scores, so its correlation with them arises only through the shared labels —
    the way two real models correlate.

    The first version of this function built the combined arm by mixing the
    LABEL into the clinical scores. That produced two arms correlated far beyond
    anything real, the paired bootstrap became almost variance-free, and the
    table reported 1.00 power everywhere including n=150 for a +0.03 effect —
    which the observed study contradicts outright, since at n=153 the measured
    delta CI was [-0.107, +0.021], a width of 0.128.
    """
    from scipy.stats import norm

    mu = np.sqrt(2) * norm.ppf(min(max(auc, 0.5001), 0.9999))
    z = rng.normal(0.0, 1.0, size=len(y))
    return z + mu * y


def _ranks(v: np.ndarray) -> np.ndarray:
    return v.argsort().argsort() / max(len(v) - 1, 1)


def power_at(n: int, img_auc: float, y_pool: np.ndarray, clin_pool: np.ndarray,
             n_sim: int, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    """(power, mean observed delta) for a cohort of size n.

    The combined arm is an unweighted average of within-arm ranks, which is
    exactly how theia.analysis.incremental builds it, so the simulated estimator
    matches the real one rather than an idealised version of it.
    """
    rng = np.random.default_rng(seed)
    wins, deltas_obs = 0, []
    for _ in range(n_sim):
        idx = rng.integers(0, len(y_pool), n)
        y, clin = y_pool[idx], clin_pool[idx]
        if len(set(y.tolist())) < 2:
            continue
        img = _imaging_arm(y, img_auc, rng)
        both = 0.5 * (_ranks(clin) + _ranks(img))
        deltas_obs.append(roc_auc_score(y, both) - roc_auc_score(y, clin))
        diffs = []
        for _ in range(n_boot):
            b = rng.integers(0, n, n)
            if len(set(y[b].tolist())) < 2:
                continue
            diffs.append(roc_auc_score(y[b], both[b]) - roc_auc_score(y[b], clin[b]))
        if diffs and np.percentile(diffs, 100 * alpha / 2) > 0:
            wins += 1
    return wins / max(n_sim, 1), float(np.mean(deltas_obs)) if deltas_obs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--img_aucs", default="0.65,0.75,0.85")
    ap.add_argument("--sizes", default="150,300,500,800,1200")
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results/power.json")
    a = ap.parse_args()

    blob = json.load(open(a.baselines))
    rows = (blob.get("oof") or {}).get("clinical")
    if not rows:
        raise SystemExit("baselines.json has no clinical out-of-fold predictions; "
                         "run theia.analysis.baselines first")
    y = np.array([r["egfr_true"] for r in rows])
    clin = np.array([r["egfr_prob"] for r in rows], dtype=float)
    print(f"[power] observed cohort: n={len(y)}, {y.sum()} positive "
          f"({100*y.mean():.0f}%), clinical AUC {roc_auc_score(y, clin):.3f}\n")

    img_aucs = [float(x) for x in a.img_aucs.split(",")]
    sizes = [int(x) for x in a.sizes.split(",")]
    table = {}
    print("power to show incremental AUC > 0 (paired bootstrap, 95% CI excludes 0)")
    print("parametrised by the IMAGING arm's standalone AUC; the increment it")
    print("actually delivers on top of clinical is shown beneath each power.\n")
    print("      n  " + "".join(f"   img AUC {v:.2f}" for v in img_aucs))
    for n in sizes:
        cells = [power_at(n, v, y, clin, a.n_sim, a.n_boot, a.alpha, 1337 + n)
                 for v in img_aucs]
        table[n] = {str(v): {"power": p, "mean_delta": d}
                    for v, (p, d) in zip(img_aucs, cells)}
        print(f"   {n:4d}  " + "".join(f"        {p:4.2f}" for p, _ in cells))
        print(f"         " + "".join(f"     (d={d:+.3f})" for _, d in cells))

    print("\n[power] read as a floor: assumes the external cohort resembles this one "
          "in prevalence, clinical-model strength and arm correlation.")
    json.dump({"n_observed": int(len(y)), "prevalence": float(y.mean()),
               "clinical_auc": float(roc_auc_score(y, clin)), "table": table},
              open(a.out, "w"), indent=2)
    print(f"[power] wrote {a.out}")


if __name__ == "__main__":
    main()
