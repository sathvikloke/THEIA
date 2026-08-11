"""Are the whole-lung attributes visible inside THEIA's lesion crop?

`semantic_oracle` established that the radiologist's WHOLE-LUNG reads -- emphysema,
airway abnormality, fibrosis -- carry more EGFR signal than the nodule-morphology
reads (0.802-0.824 against 0.747-0.749, 10/10 and 9/10 seeds). That makes a field-
of-view argument tempting: THEIA crops to `(bbox_mm + 24) * 2.5` around the lesion,
so diffuse parenchymal disease elsewhere in the lung is outside the pixels it sees.

But "outside the anatomy I labelled whole-lung" is not the same as "outside the
crop". A 2.5x crop around a 30mm lesion still spans ~135mm of thorax, which is a
lot of lung. If emphysema is legible in that much context, the FOV is not the
binding constraint and widening it buys less than the oracle implies.

So this flips the target. Instead of predicting EGFR, it predicts each SEMANTIC
ATTRIBUTE from the frozen crop tokens, through the same nested folds:

    near chance -> the attribute is genuinely not in the crop. FOV is the problem.
    high AUC    -> it is in the crop already, and the ceiling is the model, not
                   the field of view.

The in-crop attributes are the control. They MUST be predictable -- they describe
the lesion, which is centred in the crop by construction. If they come out at
chance too, the probe is measuring nothing and the whole-lung numbers mean nothing
either.

This exists because the direct experiment is blocked: rebuilding whole-lung crops
needs the raw DICOM, and `data/raw/nsclc_radiogenomics/dicom/` holds 188 patient
directories containing zero files. This runs off the cached tokens instead.

Run: python -m theia.analysis.fov_probe
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, stdev

import numpy as np

# Attributes worth probing, with the anatomy they require. Kept small and binary
# on purpose: a 190-patient cohort cannot support probing 100 one-hot columns.
TARGETS = [
    # (display name, semantic label, value that counts as positive, where it lives)
    ("emphysema", "Emphysema", "present", "whole_lung"),
    ("fibrosis", "Fibrosis", "present", "whole_lung"),
    ("airway abnormality", "Lung Parencyma Features", "airway abnormality", "whole_lung"),
    ("bronchial wall thickening", "Lung Parencyma Features", "bronchial wall thickening", "whole_lung"),
    ("centrilobular nodules", "Centrilobular Nodules - Diffuse (RB type nodules)", "present", "whole_lung"),
    # Controls: these describe the lesion itself and must be predictable.
    ("spiculated margin", "Nodule Margins-Primary Pattern", "spiculated", "in_crop"),
    ("lobulated margin", "Nodule Margins-Primary Pattern", "lobulated", "in_crop"),
    ("round shape", "Nodule Shape", "round", "in_crop"),
    ("pleural attachment", "Nodule Associated Findings", "attachment to pleura", "in_crop"),
    ("peripheral location", "Axial Location", "peripheral", "in_crop"),
]


def probe(X: np.ndarray, y: np.ndarray, rows: list[dict], cfg, seeds: list[int]) -> tuple:
    """Pooled out-of-fold AUC for a binary target, through THEIA's own folds.

    Reuses `oof_predictions` by writing the target into a synthetic gene slot, so
    the fold construction, the inner tuning loop and the rank pooling are byte-for
    -byte the ones the headline number uses.
    """
    from theia.analysis.baselines import oof_predictions
    from theia.engine.evaluate import pooled_metrics

    shadow = [dict(r, labels=dict(r.get("labels", {}), PROBE=int(v))) for r, v in zip(rows, y)]
    aucs = []
    for s in seeds:
        cfg["seed"] = s
        oof = oof_predictions(X, shadow, "PROBE", cfg)
        aucs.append(pooled_metrics(oof, ["PROBE"], cfg.eval.bootstrap_n)["probe_auc"])
    aucs = [a for a in aucs if a is not None and not np.isnan(a)]
    if not aucs:
        return float("nan"), 0.0, []
    return mean(aucs), (stdev(aucs) if len(aucs) > 1 else 0.0), aucs


def main() -> None:
    from theia.analysis.diagnostics import load_rows
    from theia.analysis.semantic_oracle import load_semantics
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default="results/_token_cache.npz")
    ap.add_argument("--seeds", default="1337,7,42,3,11")
    ap.add_argument("--min-pos", type=int, default=15,
                    help="skip targets rarer than this; a 10-positive probe is noise")
    ap.add_argument("--out", default="results/fov_probe.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    sem = load_semantics(os.path.join(cfg.paths.raw_dir, "aim"))

    if not os.path.exists(a.cache):
        raise SystemExit(f"{a.cache} is missing. Run theia.analysis.pooling_sweep first "
                         "to build the frozen-token cache.")
    z = np.load(a.cache)
    tok = z["tok"]                       # [P, N, D]
    X = tok.mean(axis=1)                 # whole-crop mean pool, the current baseline
    print(f"[fov] {X.shape[0]} patients, {X.shape[1]}-d frozen crop features, seeds={seeds}\n")

    results = {}
    print(f"{'attribute':<28} {'where':<11} {'n+':>4}  {'AUC from crop pixels'}")
    for name, label, positive, where in TARGETS:
        y = np.array([1 if positive in sem.get(r["patient_id"], {}).get(label, []) else 0
                      for r in rows])
        if int(y.sum()) < a.min_pos or int(y.sum()) > len(y) - a.min_pos:
            print(f"{name:<28} {where:<11} {int(y.sum()):>4}  (skipped, too rare)")
            continue
        m, s, per = probe(X, y, rows, cfg, seeds)
        results[name] = {"label": label, "positive": positive, "where": where,
                         "n_pos": int(y.sum()), "mean": m, "sd": s, "per_seed": per}
        print(f"{name:<28} {where:<11} {int(y.sum()):>4}  {m:.3f} +/- {s:.3f}")

    wl = [v["mean"] for v in results.values() if v["where"] == "whole_lung"]
    ic = [v["mean"] for v in results.values() if v["where"] == "in_crop"]
    print()
    if ic:
        print(f"[fov] in-crop controls   : {mean(ic):.3f}  "
              f"(must be well above 0.5, else the probe measures nothing)")
    if wl:
        print(f"[fov] whole-lung targets : {mean(wl):.3f}")
    if wl and ic:
        print(f"[fov] gap                : {mean(ic) - mean(wl):+.3f}")
        print("\n[fov] reading: whole-lung near 0.5 with controls high => the information is "
              "outside the crop and widening the FOV is justified.\n"
              "      whole-lung also high => it is already in-crop; the ceiling is the model.")

    json.dump({"seeds": seeds, "n": int(X.shape[0]), "targets": results,
               "whole_lung_mean": mean(wl) if wl else None,
               "in_crop_mean": mean(ic) if ic else None},
              open(a.out, "w"), indent=2)
    print(f"\n[fov] wrote {a.out}")


if __name__ == "__main__":
    main()
