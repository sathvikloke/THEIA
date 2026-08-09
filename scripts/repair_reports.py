"""Recompute the `report` field in rows.jsonl from the AIM XML, in place.

Surgical on purpose. A full re-preprocess would also regenerate every .npz, and
segment_mask has changed since this cohort was last processed, so the masks --
and therefore every archived classification result -- would stop being
comparable. Only the text changes here; images and ROIs are untouched.

Why it is needed: the AIM parser read the readable term from `codeSystem` only,
which is where the AMC-* files put it. The R01-* files put it in a nested
iso:displayName, so 117 of 158 reports degraded to raw RadLex ids and the
generation head trained on them.

Run: python scripts/repair_reports.py [--dry-run]
"""
import argparse, json, os, re, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theia.config import load_config
from theia.data.aim import build_report, parse_aim

CODE = re.compile(r"\brid\d+\b|\blt\d+\b", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
path = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
aim_dir = os.path.join(cfg.paths.raw_dir, "aim")
rows = [json.loads(l) for l in open(path)]

before = sum(1 for r in rows if CODE.search(r.get("report", "")))
changed = 0
for r in rows:
    xml = os.path.join(aim_dir, f"{r['patient_id']}.xml")
    if not os.path.exists(xml):
        continue                      # pseudo-report from the clinical sheet
    try:
        new = build_report(parse_aim(xml))
    except Exception as exc:          # noqa: BLE001
        print(f"[repair] {r['patient_id']}: {type(exc).__name__}: {exc}")
        continue
    if new and new != r.get("report"):
        r["report"] = new
        changed += 1

after = sum(1 for r in rows if CODE.search(r.get("report", "")))
print(f"[repair] {len(rows)} rows | reports with raw codes: {before} -> {after} "
      f"| rewritten: {changed}")
if a.dry_run:
    print("[repair] dry run, nothing written")
else:
    shutil.copy(path, path + ".bak")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[repair] wrote {path} (backup at {path}.bak)")
