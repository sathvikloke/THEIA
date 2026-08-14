"""How often does externally-aligned attention emerge from an identical pipeline?

Reads every `results/stability/stab-s*_external.json` produced by
`scripts/stability.sh` and reports the distribution the three canonical runs were
too few to estimate: across independent trainings of the same locked
configuration, what fraction align on the external cohort, and how wide is the
spread?

The alignment criterion is fixed here and not tuned: a fold counts as aligned if
its pointing accuracy exceeds the centre prior's 0.476. That threshold comes from
the already-published control, not from these runs, so it cannot be chosen to
flatter them. The bimodality reported in the manuscript -- successes at 0.76-0.89,
failures at exactly 0.000, nothing between -- means the count is insensitive to
where in that gap the line is drawn, and `--threshold` exists so a reader can
check that rather than take it on trust.

This is a post-hoc characterisation. The external cohort had been inspected long
before these runs existed, so nothing here is confirmatory.

Run: python -m theia.analysis.stability
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

import numpy as np

CENTRE_PRIOR_POINTING = 0.476


def load(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        blob = json.load(open(path))
        run = os.path.basename(path).replace("_external.json", "")
        for row in blob.get("rows", []):
            out.append({"run": run, "fold": row.get("fold"),
                        "pointing": row.get("grounding_pointing"),
                        "iou": row.get("grounding_iou"),
                        "mass": row.get("grounding_mass"),
                        "lift": row.get("grounding_mass_lift")})
    return out


def summarise(rows: list[dict], threshold: float) -> dict:
    runs: dict[str, list[dict]] = {}
    for r in rows:
        runs.setdefault(r["run"], []).append(r)

    per_run = []
    for run, folds in sorted(runs.items()):
        aligned = [f for f in folds if (f["pointing"] or 0) > threshold]
        per_run.append({
            "run": run, "n_folds": len(folds), "n_aligned": len(aligned),
            "frac_aligned": len(aligned) / max(len(folds), 1),
            "mean_pointing": float(np.mean([f["pointing"] for f in folds])),
            "mean_pointing_aligned": (float(np.mean([f["pointing"] for f in aligned]))
                                      if aligned else 0.0),
        })

    fracs = [r["frac_aligned"] for r in per_run]
    # A run "works" if a majority of its folds align -- the manuscript's informal
    # reading of runs A/B/C (2/5, 5/5, 4/5).
    working = [r for r in per_run if r["frac_aligned"] > 0.5]
    return {
        "n_runs": len(per_run), "n_folds": len(rows),
        "threshold_pointing": threshold,
        "fold_level_aligned": sum(r["n_aligned"] for r in per_run),
        "fold_level_frac": sum(r["n_aligned"] for r in per_run) / max(len(rows), 1),
        "run_level_working": len(working),
        "run_level_frac": len(working) / max(len(per_run), 1),
        "frac_aligned_mean": float(np.mean(fracs)) if fracs else 0.0,
        "frac_aligned_sd": float(st.stdev(fracs)) if len(fracs) > 1 else 0.0,
        "per_run": per_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/stability/stab-s*_external.json")
    ap.add_argument("--threshold", type=float, default=CENTRE_PRIOR_POINTING,
                    help="pointing accuracy above which a fold counts as aligned")
    ap.add_argument("--out", default="results/stability_summary.json")
    a = ap.parse_args()

    rows = load(a.pattern)
    if not rows:
        raise SystemExit(f"[stab] nothing matched {a.pattern}; is the run finished?")
    s = summarise(rows, a.threshold)

    print(f"[stab] {s['n_runs']} independent runs, {s['n_folds']} fold-checkpoints")
    print(f"[stab] alignment threshold: pointing > {a.threshold:.3f} "
          f"(the centre prior's own score)\n")
    print(f"{'run':<14}{'aligned':>9}{'mean pointing':>15}")
    for r in s["per_run"]:
        print(f"{r['run']:<14}{r['n_aligned']:>4}/{r['n_folds']:<4}"
              f"{r['mean_pointing']:>15.3f}")
    print(f"\n[stab] fold level: {s['fold_level_aligned']}/{s['n_folds']} "
          f"({100*s['fold_level_frac']:.0f}%) aligned")
    print(f"[stab] run level:  {s['run_level_working']}/{s['n_runs']} "
          f"({100*s['run_level_frac']:.0f}%) align in a majority of folds")
    print(f"[stab] per-run aligned fraction: {s['frac_aligned_mean']:.2f} "
          f"+/- {s['frac_aligned_sd']:.2f}")

    # Sensitivity: the claim should not depend on where in the empty band the
    # threshold sits.
    print("\n[stab] threshold sensitivity (fold-level aligned fraction):")
    for t in (0.10, 0.30, 0.476, 0.60, 0.70):
        alt = summarise(rows, t)
        print(f"       pointing > {t:.3f}: {alt['fold_level_aligned']}/{alt['n_folds']}")

    json.dump(s, open(a.out, "w"), indent=2)
    print(f"\n[stab] wrote {a.out}")
    print("[stab] NOTE: post-hoc. The external cohort was inspected long before "
          "these runs\n       existed; nothing here is confirmatory of H1.")


if __name__ == "__main__":
    main()
