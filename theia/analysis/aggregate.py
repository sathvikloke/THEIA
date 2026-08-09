"""Aggregate archived runs into the number a paper can actually quote.

A single run's pooled AUC carries two sources of variance that its bootstrap CI
does not describe. The CI covers sampling of *patients*; it says nothing about
re-running the same configuration, which on this project moved pooled EGFR
across 0.621 / 0.656 / 0.660 from initialisation and MPS non-determinism alone.
Quoting one run implies a precision the estimator does not have.

What "seed variance" covers here, stated precisely because it is easy to
overclaim: `cfg.seed` drives BOTH the model initialisation and
`nested_kfold_indices`, so changing it re-partitions the cohort as well as
re-initialising the model. The spread reported below is therefore variance over
(split x initialisation) jointly, not initialisation alone. That is the more
useful quantity — it answers "how much would this number move if someone
repeated the study" — but it must not be described as initialisation variance,
and it cannot be decomposed into the two parts without holding one fixed.

Two summaries are produced, and they answer different questions:

  across-seed mean +/- sd     how much the headline moves if you rerun it,
                              re-splitting and re-initialising. The honest
                              headline.
  pooled-over-all-seeds AUC   every seed's out-of-fold predictions merged into
                              one ranking, rank-normalised within (seed, fold).
                              Tighter, and appropriate only as a point estimate
                              of the configuration's central tendency — it does
                              NOT absorb seed variance, so it must be reported
                              beside the sd, never instead of it.

Stalled folds are excluded from the aggregate and counted in the output. A fold
that skipped every optimizer step never trained, and averaging it in would drag
the estimate toward chance for a reason that has nothing to do with the science.

Run: python -m theia.analysis.aggregate --pattern 'results/ms-s*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
from statistics import mean, stdev

from theia.engine.evaluate import pooled_metrics


def load(pattern: str) -> list[dict]:
    blobs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            blob = json.load(fh)
        blob["_path"] = path
        blobs.append(blob)
    if not blobs:
        raise SystemExit(f"no results matched {pattern!r}")
    return blobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/ms-s*.json")
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--out", default="results/multiseed_summary.json")
    a = ap.parse_args()

    g = a.gene.lower()
    blobs = load(a.pattern)
    print(f"[agg] {len(blobs)} run(s) matching {a.pattern}")

    per_run, merged, stalled_total = [], [], 0
    for blob in blobs:
        stalled = [f for f in blob["folds"] if f.get("stalled")]
        stalled_total += len(stalled)
        auc = (blob.get("pooled") or {}).get(f"{g}_auc")
        seed = (blob.get("config") or {}).get("seed")
        if auc is not None and auc == auc:
            per_run.append(auc)
        # Namespace the fold id by run, so rank normalisation never merges two
        # different runs' fold 0 into one ranking.
        for f in blob["folds"]:
            if f.get("stalled"):
                continue
            for r in (f.get("oof") or []):
                merged.append({**r, "fold": f"{blob.get('run_id')}::{r.get('fold')}"})
        print(f"[agg]   {blob.get('run_id'):22s} seed={seed} "
              f"{g.upper()} {auc if auc is None else round(auc, 3)}"
              + (f"  ({len(stalled)} stalled fold(s) excluded)" if stalled else ""))

    out: dict = {"gene": a.gene, "n_runs": len(blobs), "stalled_folds": stalled_total}
    if len(per_run) >= 2:
        m, s = mean(per_run), stdev(per_run)
        out["across_seed_mean"], out["across_seed_sd"] = m, s
        out["across_seed_min"], out["across_seed_max"] = min(per_run), max(per_run)
        print(f"\n[agg] ACROSS-SEED  {a.gene} AUC {m:.3f} +/- {s:.3f} (sd, n={len(per_run)}) "
              f"range [{min(per_run):.3f}, {max(per_run):.3f}]")
        print("[agg]   ^ the headline: includes rerun variance, which a single "
              "run's CI does not. The seed drives both the split and the model "
              "init, so this is (split x init) variance, not init alone.")
    elif per_run:
        out["across_seed_mean"] = per_run[0]
        print(f"\n[agg] only one run; {a.gene} AUC {per_run[0]:.3f} "
              "(no across-seed sd available)")

    if merged:
        pm = pooled_metrics(merged, [a.gene], 2000)
        out["pooled_all_seeds"] = pm
        print(f"[agg] POOLED-ALL   {a.gene} AUC {pm.get(f'{g}_auc', float('nan')):.3f} "
              f"95% CI [{pm.get(f'{g}_auc_lo', float('nan')):.3f}, "
              f"{pm.get(f'{g}_auc_hi', float('nan')):.3f}] "
              f"over {int(pm.get(f'{g}_n', 0))} predictions")
        print("[agg]   ^ point estimate only. Its CI covers patient sampling, "
              "NOT rerun variance — quote it beside the sd, never instead of it.")
    if stalled_total:
        print(f"\n[agg] {stalled_total} stalled fold(s) excluded — they never "
              "trained; see docs/RESULTS.md")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[agg] wrote {a.out}")


if __name__ == "__main__":
    main()
