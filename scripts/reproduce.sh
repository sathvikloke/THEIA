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

echo "=== tests ==="
# python -m pytest, not bare pytest: the module form puts the repo on
# sys.path. conftest.py covers the bare form too, but this is one less thing
# that depends on how the caller's PATH resolves.
python -m pytest -q

echo
echo "=== 1. headline, with rerun variance ==="
python -m theia.analysis.aggregate --pattern 'results/ms-s*.json'

echo
echo "=== 2. the comparison that decides the project ==="
python -m theia.analysis.incremental --pattern 'results/ms-s*.json'

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
import glob, json
import numpy as np
rows = [f["test"] for p in sorted(glob.glob("results/ms-s*.json"))
        for f in json.load(open(p))["folds"] if not f.get("stalled")]
m = np.array([r["grounding_mass"] for r in rows])
s = np.array([r["grounding_mass_shuffled"] for r in rows])
print(f"{len(rows)} folds: mass {m.mean():.3f} +/- {m.std():.3f} vs {s.mean():.3f} shuffled")
print(f"  = {m.mean()/s.mean():.1f}x chance; beats its own baseline in {int((m>s).sum())}/{len(m)} folds")
PY

echo
echo "=== 5. sample size for the next cohort ==="
python - <<'PY'
import json
t = json.load(open("results/power.json"))["table"]
print("  n     img AUC 0.65   0.75   0.85   (power to show incremental AUC > 0)")
for n, row in t.items():
    print(f"  {n:>5s}" + "".join(f"        {v['power']:.2f}" for v in row.values()))
PY

echo
echo "=== 6. figures ==="
python -m theia.analysis.figures

echo
echo "Done. Provenance: docs/RESULTS.md   Scope: docs/MODEL_CARD.md"
