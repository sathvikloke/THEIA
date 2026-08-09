"""Frozen linear probe on THEIA's OWN nested splits, so the numbers compare.

diagnostics.py scores the probe under its own flat StratifiedKFold. THEIA is
scored under nested splits with checkpoint selection. Comparing those two
directly would confound the model with the protocol, so the probe is re-run here
through exactly the folds each archived seed used.
"""
import glob, json, os, sys
import numpy as np
import torch
from statistics import mean, stdev

from theia.analysis.baselines import oof_predictions
from theia.analysis.diagnostics import deep_features, load_rows
from theia.config import load_config
from theia.engine.evaluate import pooled_metrics
from theia.runtime import resolve_device


def main():
    cfg = load_config("configs/default.yaml")
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    print("[probe] extracting frozen features...", flush=True)
    X = deep_features(cfg, rows, resolve_device("auto"))
    print(f"[probe] {X.shape}", flush=True)

    aucs = []
    for p in sorted(glob.glob("results/ms-s*.json")):
        blob = json.load(open(p))
        seed = (blob.get("config") or {}).get("seed", cfg.seed)
        cfg["seed"] = seed
        oof = oof_predictions(X, rows, "EGFR", cfg)
        m = pooled_metrics(oof, ["EGFR"], cfg.eval.bootstrap_n)
        theia = (blob.get("pooled") or {}).get("egfr_auc")
        aucs.append(m["egfr_auc"])
        print(f"[probe] seed {seed:5d}  frozen probe {m['egfr_auc']:.3f} "
              f"[{m['egfr_auc_lo']:.3f}, {m['egfr_auc_hi']:.3f}]   "
              f"THEIA {theia:.3f}   diff {m['egfr_auc'] - theia:+.3f}", flush=True)
    if len(aucs) >= 2:
        print(f"\n[probe] frozen probe across seeds: {mean(aucs):.3f} +/- {stdev(aucs):.3f}")
    json.dump({"probe_aucs": aucs}, open("results/frozen_probe.json", "w"), indent=2)


if __name__ == "__main__":
    main()
