"""The primary endpoint, resampled over patients instead of counted over folds.

THE CRITICISM THIS ANSWERS. The pre-specified H1 counts how many run x fold
checkpoints beat their own spatial shuffle, and reports 11 of 15 against a 71.4%
threshold -- a margin of one checkpoint. Two objections are fair and neither is
answered by the fold count:

  1. The 15 evaluations are not 15 studies. They are three seeds x five folds
     scored on the SAME 420 external patients, with heavily overlapping training
     sets. The count treats dependent replicates as independent, and a threshold
     cleared by one of them is not a stable endpoint.

  2. The shuffle is the wrong opponent. These crops are lesion-centred, so a
     Gaussian blob at the frame centre already scores 0.270 attention mass
     against the trained model's 0.292. Almost all of the margin over a SHUFFLE
     (0.048) is reproduced by looking at the middle. The comparison that carries
     information is against the CENTRE PRIOR, and on the metrics where the centre
     prior is not free to cheat -- pointing accuracy and area-matched IoU, both
     invariant to the prior's width.

WHAT THIS DOES NOT DO. It does not replace H1. H1 was pre-specified before the
external evaluation and is reported as pre-specified, pass or fail; swapping in a
better endpoint after seeing the data is the precise failure the analysis plan
exists to prevent. This is a labelled post-hoc analysis reported ALONGSIDE it.

DESIGN. Each patient contributes one value per estimator: the trained value is
averaged over the 15 checkpoints, so training-run variability enters as a fixed
effect rather than as pseudo-replication. Patients are then resampled with
replacement 10,000 times, and the paired difference trained - centre is
recomputed within each resample. A hierarchical variant additionally resamples
the three training runs, which is the conservative reading.

Run: python -m theia.analysis.grounding_patient_level
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

METRICS = ("grounding_pointing", "grounding_iou", "grounding_mass")


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                     seed: int = 1337) -> dict:
    """Paired bootstrap over patients for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    n = a.size
    obs = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"difference": obs,
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p": min(p, 1.0)}


def hierarchical_bootstrap(per_ckpt: np.ndarray, b: np.ndarray,
                           n_runs: int = 3, n_boot: int = 10000,
                           seed: int = 1337) -> dict:
    """Resample patients AND training runs.

    `per_ckpt` is [n_checkpoints, n_patients]. Checkpoints are grouped into runs
    of equal size and runs are resampled too, so a conclusion that depends on
    which three seeds were trained shows up as a wider interval.
    """
    rng = np.random.default_rng(seed)
    k, n = per_ckpt.shape
    per_run = per_ckpt.reshape(n_runs, k // n_runs, n).mean(1)   # [runs, patients]
    obs = float(per_run.mean(0).mean() - b.mean())
    d = np.empty(n_boot)
    for i in range(n_boot):
        pi = rng.integers(0, n, size=n)
        ri = rng.integers(0, n_runs, size=n_runs)
        d[i] = per_run[np.ix_(ri, pi)].mean(0).mean() - b[pi].mean()
    p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"difference": obs,
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p": min(p, 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",
                    default="results/external_grounding_canonical_per_patient.npz")
    ap.add_argument("--out", default="results/grounding_patient_level.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        raise SystemExit(
            f"[pl] {a.npz} not found. Regenerate it with:\n"
            f"    python -m theia.analysis.external_grounding\n"
            f"which now writes per-patient vectors alongside the fold summary.")

    z = np.load(a.npz)
    n_pat = z["trained_grounding_pointing"].size
    n_ck = z["trained_grounding_pointing_all"].shape[0]
    print(f"[pl] {n_pat} external patients x {n_ck} checkpoints\n")

    out: dict = {"n_patients": int(n_pat), "n_checkpoints": int(n_ck),
                 "n_boot": a.n_boot, "comparisons": {}}

    for m in METRICS:
        tr = z[f"trained_{m}"]
        allc = z[f"trained_{m}_all"]
        for ctrl in ("centre", "random"):
            key = f"{ctrl}_{m}"
            if key not in z:
                continue
            r = paired_bootstrap(tr, z[key], a.n_boot)
            h = hierarchical_bootstrap(allc, z[key], n_boot=a.n_boot)
            out["comparisons"][f"{m} vs {ctrl}"] = {"patient_level": r,
                                                    "hierarchical": h}
            print(f"  {m.replace('grounding_',''):<9} vs {ctrl:<6} "
                  f"{r['difference']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"P={r['p']:.4f}   | hierarchical "
                  f"[{h['ci_lo']:+.3f},{h['ci_hi']:+.3f}] P={h['p']:.4f}")
        # and against the arm's own spatial shuffle, the pre-specified opponent
        sh = f"trained_grounding_mass_shuffled"
        if m == "grounding_mass" and sh in z:
            r = paired_bootstrap(tr, z[sh], a.n_boot)
            out["comparisons"]["grounding_mass vs shuffle"] = {"patient_level": r}
            print(f"  {'mass':<9} vs {'shuffle':<6} "
                  f"{r['difference']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"P={r['p']:.4f}   (the pre-specified contrast)")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n[pl] wrote {a.out}")
    print("[pl] NOTE: this is a labelled post-hoc analysis. The pre-specified H1 "
          "remains\n     the fold-count endpoint and is reported as such.")


if __name__ == "__main__":
    main()
