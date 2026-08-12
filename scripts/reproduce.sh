#!/usr/bin/env bash
# Reproduce every reported number from the archives. No data, no GPU, no
# downloads.
#
# This works because results/<run_id>.json stores each run's out-of-fold
# predictions, not just its summary metrics. The weights for this project's
# earlier best run no longer exist; its curves would otherwise be unrecoverable.
#
# Retraining is a different command and needs the cohort — see docs/DATA_REQUEST.md
# and scripts/multiseed.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

# Steps that need the cohort on disk are skipped rather than fatal. With
# `set -e` a missing data/processed/rows.jsonl killed the script at step 2 of 10
# on a fresh clone, taking the eight steps that DO work from the archives with
# it -- and this is the first command a new reader runs.
have_cohort() { [ -f data/processed/rows.jsonl ] && [ -f data/raw/nsclc_radiogenomics/clinical/clinical.csv ]; }
skip_note() { echo "  SKIPPED: needs the cohort on disk (see docs/DATA_REQUEST.md)."; }

echo "=== tests ==="
# python -m pytest, not bare pytest: the module form puts the repo on
# sys.path. conftest.py covers the bare form too, but this is one less thing
# that depends on how the caller's PATH resolves.
python -m pytest -q

echo
echo "=== 1. headline, with rerun variance ==="
# No --pattern: both read results/CANONICAL.json. Passing the old
# 'results/ms-s*.json' glob here re-introduced the stalled run and
# silently overwrote the regenerated archives with superseded numbers.
python -m theia.analysis.aggregate

echo
echo "=== 2. the comparison that decides the project ==="
if have_cohort; then python -m theia.analysis.incremental; else skip_note; fi

echo
echo "=== 3. baseline sensitivity to analytic choices ==="
python - <<'PY'
import json
d = json.load(open("results/radiomics_sensitivity.json"))
rad = [v["auc"] for k, v in d.items() if not k.startswith("clinical")]
cli = [v["auc"] for k, v in d.items() if k.startswith("clinical")]
print(f"radiomics spans {min(rad):.3f}-{max(rad):.3f} (spread {max(rad)-min(rad):.3f})")
print(f"clinical  spans {min(cli):.3f}-{max(cli):.3f} (spread {max(cli)-min(cli):.3f})")
PY

echo
echo "=== 4. grounding, against its own shuffled baseline ==="
python - <<'PY'
import json
import numpy as np
runs = json.load(open("results/CANONICAL.json"))["headline_runs"]
rows = [f["test"] for p in runs
        for f in json.load(open(p))["folds"] if not f.get("stalled")]
m = np.array([r["grounding_mass"] for r in rows])
s = np.array([r["grounding_mass_shuffled"] for r in rows])
print(f"{len(rows)} folds: mass {m.mean():.3f} +/- {m.std():.3f} vs {s.mean():.3f} shuffled")
print(f"  = {m.mean()/s.mean():.1f}x chance; beats its own baseline in {int((m>s).sum())}/{len(m)} folds")
PY

echo
echo "=== 5. THE PRIMARY ENDPOINT: grounding on an unseen cohort ==="
python - <<'PY'
import json
d = json.load(open("results/external_grounding.json"))
rows, cp, rnd = d["rows"], d["control_centre_prior"], d["control_random_init"]
mean = lambda k, rs=rows: sum(r[k] for r in rs) / len(rs)
live = [r for r in rows if r["grounding_pointing"] > 0.05]
print(f"{d['n_external']} external patients, {len(rows)} evaluations "
      f"({len(rows)-len(live)} dead)")
print(f"{'':<10}{'trained':>9}{'live only':>11}{'centre':>9}{'shuffle':>9}{'random':>9}")
for k in ("grounding_mass", "grounding_pointing", "grounding_iou"):
    n = k.split("_")[1]
    print(f"{n:<10}{mean(k):>9.3f}{mean(k, live):>11.3f}{cp[k]:>9.3f}"
          f"{mean(k+'_shuffled'):>9.3f}{rnd[k]:>9.3f}")
print(f"mass lift {d['mean_mass_lift']:+.3f}, beats shuffle "
      f"{d['folds_beating_shuffle']}/{d['n_folds']}  ->  "
      f"GATE D {'PASS' if d['gate_d_passed'] else 'FAIL'}")
PY

echo
echo "=== 6. calibration: the AUC is above chance, the probabilities are not useful ==="
python - <<'PY'
import json
d = json.load(open("results/calibration.json"))
for label, blk in d["subsets"].items():
    per = blk["per_run"]
    unc = per[0]["uncertainty"]
    print(f"{label}: base-rate Brier floor {unc:.3f}")
    for r in per:
        worse = "WORSE than a constant" if r["brier"] > unc else "better"
        print(f"   {r['run']:<26} slope {r['calibration_slope']:>6.3f}  "
              f"Brier {r['brier']:.3f}  {worse}")
PY

echo
echo "=== 7. no fusion topology beats the chart ==="
python - <<'PY'
import json
d = json.load(open("results/fusion.json"))
r = d["results"]
for k in d["arms"]:
    print(f"  {k:<34} {r[k]:.3f}")
for k, v in sorted((k, v) for k, v in r.items() if isinstance(v, dict)):
    print(f"  {k:<34} {v['auc']:.3f}   vs {d['best_single']} "
          f"{v['vs_best_single']:+.3f}  p={v['p']:.3f}")
PY

echo
echo "=== 8. sample size AND precision for the next cohort ==="
python - <<'PY'
import json
d = json.load(open("results/power.json"))
print("  n     img AUC 0.65   0.75   0.85   (power to show incremental AUC > 0)")
for n, row in d["table"].items():
    print(f"  {n:>5s}" + "".join(f"        {v['power']:.2f}" for v in row.values()))
if "precision_n_for_ci" in d:
    print("\n  precision: n needed for a 95% CI half-width, at AUC 0.63")
    for h, n in d["precision_n_for_ci"]["0.63"].items():
        print(f"    +/-{h:<6} {n}")
    print(f"  observed cohort half-width: +/-{d['observed_ci_halfwidth_at_063']:.3f}")
    print(f"  attainable external set:    "
          f"+/-{d['attainable_external_ci_halfwidth_at_063']:.3f}  "
          f"<- spans chance AND the published 0.80s")
    print(f"  Riley minimum n: {d['riley_min_n']}")
PY

echo
echo "=== 9. cohort flow and Table 1 ==="
if have_cohort; then python -m theia.analysis.cohort; else skip_note; fi

echo
echo "=== 10. figures ==="
python -m theia.analysis.figures || skip_note

echo
echo "Done. Provenance: docs/RESULTS.md   Scope: docs/MODEL_CARD.md"
echo "Plan: docs/ANALYSIS_PLAN.md   Reporting: docs/CLAIM_CHECKLIST.md"
