"""Does imaging add anything on top of what a clinician already knows?

This is the question the project turns on, and the single-run answer in
`baselines.py` is not enough for it: pooled EGFR AUC moves +/-0.041 across seeds,
so a delta computed from one run inherits that noise without showing it.

The test is repeated PER SEED and the deltas are then summarised:

  for each archived multi-seed run:
      rebuild the clinical baseline ON THAT RUN'S SPLITS  (cfg.seed drives
          nested_kfold_indices, so the folds must be regenerated per seed or the
          two arms are not paired at all)
      delta = AUC(clinical + THEIA) - AUC(clinical), paired bootstrap
  report mean +/- sd of the deltas, and each seed's own CI

Why not just merge every seed's predictions into one big paired test: a patient
appears once per seed, so the merged set has patients x seeds rows. The paired
bootstrap resamples rows, which would treat three correlated copies of the same
patient as three independent observations and shrink the interval by roughly
sqrt(3) for no added information. Per-seed deltas keep each test honest and the
spread across them is reported explicitly instead.

The combination is an unweighted average of within-fold ranks. Fitting a
combiner on out-of-fold predictions that were themselves selected on this cohort
would leak, and with ~40 positives the fitted weight would be noise. An
unweighted average cannot flatter the combination.

Run: python -m theia.analysis.incremental --pattern 'results/ms-s*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean, stdev

from theia.analysis.baselines import (_stack_with_theia, clinical_features,
                                      oof_predictions, paired_delta)
from theia.analysis.diagnostics import load_rows
from theia.engine.evaluate import pooled_metrics


def main() -> None:
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pattern", default=None,
                    help="glob. Overrides the canonical run set; use with care, "
                         "since a glob is what let a stalled run into the headline.")
    ap.add_argument("--canonical", default="results/CANONICAL.json")
    ap.add_argument("--gene", default=None)
    ap.add_argument("--out", default="results/incremental_value.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    gene = a.gene or cfg.data.get("headline_genes", ["EGFR"])[0]
    g = gene.lower()
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    Xc, _ = clinical_features(rows, os.path.join(cfg.paths.raw_dir, "clinical",
                                                 "clinical.csv"))

    # Default to the named canonical set rather than a glob. `results/ms-s*.json`
    # used to be the definition of the headline, and it silently swept in a run
    # whose fold 1 stalled -- 122 of 153 patients, and the highest AUC of the
    # three seeds. Naming the runs in one file is what stops that recurring.
    if a.pattern:
        paths = sorted(glob.glob(a.pattern))
        if not paths:
            raise SystemExit(f"no results matched {a.pattern!r}")
        print(f"[inc] {len(paths)} run(s) from glob {a.pattern!r}; gene={gene}\n")
    else:
        paths = json.load(open(a.canonical))["headline_runs"]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"{a.canonical} names runs that do not exist: {missing}")
        print(f"[inc] {len(paths)} canonical run(s) from {a.canonical}; gene={gene}\n")

    per_seed = []
    for path in paths:
        blob = json.load(open(path))
        seed = (blob.get("config") or {}).get("seed", cfg.seed)
        stalled = [f["fold"] for f in blob["folds"] if f.get("stalled")]
        theia_oof = [r for f in blob["folds"] if not f.get("stalled")
                     for r in (f.get("oof") or [])]
        if not theia_oof:
            print(f"[inc] {blob.get('run_id')}: no usable predictions, skipping")
            continue

        # Regenerate the baseline on THIS seed's folds. cfg.seed drives the
        # splits, so reusing one baseline across seeds would pair each patient
        # against a model that held out a different set entirely.
        cfg["seed"] = seed
        clin = oof_predictions(Xc, rows, gene, cfg)
        # Restrict the baseline to the patients THEIA actually scored, so a
        # stalled fold does not give the baseline an advantage in coverage.
        scored = {r["patient_id"] for r in theia_oof}
        clin = [r for r in clin if r["patient_id"] in scored]

        both = _stack_with_theia(clin, theia_oof, gene)
        m_clin = pooled_metrics(clin, [gene], cfg.eval.bootstrap_n)
        m_both = pooled_metrics(both, [gene], cfg.eval.bootstrap_n)
        d = paired_delta(both, clin, gene, cfg.eval.bootstrap_n, seed)

        per_seed.append({
            "run_id": blob.get("run_id"), "seed": seed, "n": d["n"],
            "stalled_folds": stalled,
            "clinical_auc": m_clin.get(f"{g}_auc"),
            "clinical_plus_theia_auc": m_both.get(f"{g}_auc"),
            "delta": d["delta"], "delta_lo": d["delta_lo"],
            "delta_hi": d["delta_hi"], "p": d["p_two_sided"],
        })
        print(f"[inc] seed {seed:5d}  clinical {m_clin.get(f'{g}_auc', float('nan')):.3f}"
              f"  +THEIA {m_both.get(f'{g}_auc', float('nan')):.3f}"
              f"  delta {d['delta']:+.3f} [{d['delta_lo']:+.3f}, {d['delta_hi']:+.3f}]"
              f"  p={d['p_two_sided']:.3f}"
              + (f"  ({len(stalled)} stalled fold)" if stalled else ""))

    out = {"gene": gene, "per_seed": per_seed}
    deltas = [p["delta"] for p in per_seed if p["delta"] == p["delta"]]
    if len(deltas) >= 2:
        out["delta_mean"], out["delta_sd"] = mean(deltas), stdev(deltas)
        print(f"\n[inc] ACROSS SEEDS  delta {mean(deltas):+.3f} +/- {stdev(deltas):.3f} "
              f"(sd, n={len(deltas)})  range [{min(deltas):+.3f}, {max(deltas):+.3f}]")
        crosses = any(p["delta_lo"] <= 0 <= p["delta_hi"] for p in per_seed)
        print("[inc] every seed's CI includes zero" if crosses
              else "[inc] at least one seed's CI excludes zero")
    elif deltas:
        out["delta_mean"] = deltas[0]

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[inc] wrote {a.out}")


if __name__ == "__main__":
    main()
