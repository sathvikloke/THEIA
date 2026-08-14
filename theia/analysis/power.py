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


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC (Hanley & McNeil 1982).

    Uses the exponential approximation for the two conditional probabilities
    Q1 = P(two positives both rank above a negative) and Q2 = P(a positive ranks
    above two negatives). It is the standard closed form behind every "how many
    patients do I need" answer in this literature, and it is what CLAIM item 21
    is asking for when it wants a sample-size justification.
    """
    a = float(np.clip(auc, 1e-6, 1 - 1e-6))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def n_for_auc_ci(auc: float, halfwidth: float, prevalence: float,
                 max_n: int = 200_000) -> int:
    """Smallest total n whose 95% CI on the AUC is no wider than +/- halfwidth.

    This answers the question that actually binds an external validation set,
    which is PRECISION, not power: with ~17 positives available, the CI on an AUC
    near 0.63 is about +/- 0.14, which spans chance and simultaneously spans the
    0.80s the literature reports. Such a test set cannot confirm or refute
    anything, and knowing that in advance is the difference between a
    pre-specified finding and a disappointment.
    """
    if not 0 < prevalence < 1:
        raise ValueError(f"prevalence must be in (0,1), got {prevalence}")
    lo, hi = 4, max_n
    if 1.96 * hanley_mcneil_se(auc, max(int(max_n * prevalence), 1),
                               max(max_n - int(max_n * prevalence), 1)) > halfwidth:
        return -1                                  # unreachable within max_n
    while lo < hi:
        mid = (lo + hi) // 2
        n_pos = max(int(round(mid * prevalence)), 1)
        n_neg = max(mid - n_pos, 1)
        if 1.96 * hanley_mcneil_se(auc, n_pos, n_neg) <= halfwidth:
            hi = mid
        else:
            lo = mid + 1
    return lo


def riley_min_n(n_predictors: int, prevalence: float, r2_cs: float | None = None,
                shrinkage: float = 0.9) -> int:
    """Riley et al. (BMJ 2020;368:m441) minimum sample size, criterion 1.

        n = P / ( (S - 1) * ln(1 - R2_cs / S) )

    with S the target expected shrinkage (0.9 = at most 10% overfitting), P the
    number of candidate predictors, and R2_cs the anticipated Cox-Snell R^2.

    NOTE: an earlier version of this function divided by (1 - 0.10/1) = 0.9 where
    the criterion needs |S - 1| = 0.1, so every number it produced was 9x too
    small. It reported 46 for the clinical model against a true 414, and 4,707
    for the frozen probe against 42,363 -- and the wrong 46 was quoted in
    ANALYSIS_PLAN next to the BMJ DOI, with a tick mark saying 153 patients
    sufficed. They do not. Correcting it strengthens the argument it was
    supporting, since it means neither arm is identifiable at this cohort size.

    Applied ONLY where its inputs are defined: the 5-variable clinical model and
    the frozen-feature probe. It needs a candidate-predictor count and a target
    Cox-Snell R^2, and neither is meaningful for an 88M-parameter frozen ViT, so
    quoting it for the deep arm would be arithmetic dressed as justification.
    """
    p = float(prevalence)
    if not 0 < p < 1:
        raise ValueError(f"prevalence must be in (0,1), got {prevalence}")
    if r2_cs is None:
        # Maximum attainable Cox-Snell R^2 for a binary outcome at this
        # prevalence, scaled to 15% of it -- Riley's recommendation when no
        # prior model exists to estimate R^2 from.
        max_r2 = 1 - (p ** p * (1 - p) ** (1 - p)) ** 2
        r2_cs = 0.15 * max_r2
    if not 0 < r2_cs < shrinkage:
        raise ValueError(f"need 0 < r2_cs < shrinkage, got {r2_cs} and {shrinkage}")
    return int(np.ceil(n_predictors / ((shrinkage - 1) * np.log(1 - r2_cs / shrinkage))))



def _clinical_n_params(cfg) -> int:
    """Columns in the clinical design matrix, not the count of named variables."""
    import json as _json
    import os as _os

    from theia.analysis.baselines import clinical_features

    rows = [_json.loads(l) for l in
            open(_os.path.join(cfg.paths.processed_dir, "rows.jsonl"))]
    rows = [r for r in rows if int(r.get("labels", {}).get("EGFR", -1)) in (0, 1)]
    _, names = clinical_features(
        rows, _os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv"))
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--img_aucs", default="0.65,0.75,0.85")
    ap.add_argument("--sizes", default="150,300,500,800,1200")
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/power.json")
    a = ap.parse_args()

    from theia.config import load_config
    cfg = load_config(a.config)

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

    # --- precision, which is what actually binds an external test set ----------
    prev = float(y.mean())
    print(f"\n[power] PRECISION (Hanley-McNeil), at prevalence {prev:.3f}")
    print("  n needed for a 95% CI half-width of:")
    print(f"  {'AUC':>6}" + "".join(f"{f'+/-{h}':>12}" for h in (0.15, 0.10, 0.05)))
    precision = {}
    for auc in (0.63, 0.70, 0.80):
        cells = [n_for_auc_ci(auc, h, prev) for h in (0.15, 0.10, 0.05)]
        precision[str(auc)] = {str(h): n for h, n in zip((0.15, 0.10, 0.05), cells)}
        print(f"  {auc:>6.2f}" + "".join(
            f"{('unreachable' if n < 0 else n):>12}" for n in cells))

    n_obs = len(y)
    n_pos = int(y.sum())
    se = hanley_mcneil_se(0.63, n_pos, n_obs - n_pos)
    print(f"\n  observed cohort n={n_obs} ({n_pos} positive): "
          f"95% CI half-width at AUC 0.63 is +/-{1.96*se:.3f}")
    ext_pos, ext_n = 17, 96
    se_ext = hanley_mcneil_se(0.63, ext_pos, ext_n - ext_pos)
    print(f"  attainable external set n~{ext_n} ({ext_pos} positive): "
          f"+/-{1.96*se_ext:.3f}  <- spans chance AND the published 0.80s")

    # Riley counts FITTED PARAMETERS, not named variables. The clinical model is
    # described as "five variables" -- age, sex, ethnicity, smoking status,
    # pack-years -- but one-hot encoding and the pack-years missingness indicator
    # expand it to 14 design-matrix columns. Passing 5 here understated the
    # minimum by a factor of ~2.8 (414 against 1159) and that number reached the
    # manuscript. Derive it from the matrix rather than restating the prose.
    n_clin = _clinical_n_params(cfg)
    riley = {f"clinical ({n_clin} fitted parameters, 5 named variables)":
             riley_min_n(n_clin, prev),
             "frozen probe (512 features)": riley_min_n(512, prev)}
    print("\n[power] Riley minimum n (shrinkage <= 10%), applicable arms only:")
    for k, v in riley.items():
        print(f"  {k:<30} {v}")
    print("  not computed for the deep arm: Riley needs a candidate-predictor count\n"
          "  and a target Cox-Snell R^2, neither defined for a frozen 88M-parameter ViT.")

    json.dump({"n_observed": int(len(y)), "prevalence": prev,
               "clinical_auc": float(roc_auc_score(y, clin)), "table": table,
               "precision_n_for_ci": precision,
               "observed_ci_halfwidth_at_063": float(1.96 * se),
               "attainable_external_ci_halfwidth_at_063": float(1.96 * se_ext),
               "riley_min_n": riley},
              open(a.out, "w"), indent=2)
    print(f"[power] wrote {a.out}")


if __name__ == "__main__":
    main()
