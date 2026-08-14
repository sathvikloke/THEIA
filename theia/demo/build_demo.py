"""Replace the demo's schematic drawing with real images and real predictions.

WHY THIS EXISTS. `demo/index.html` shipped a hand-drawn axial CT with invented
DICOM metadata and three confident genotype calls. It was labelled schematic, but
it read as a working viewer -- and this project's central finding is that a
plausible-looking interface is not evidence that anything works. A schematic demo
of a model whose paper says "plausible attention does not imply predictive
validity" is the single most misreadable artifact in the repository.

WHAT IS REAL HERE. Every pixel is a real CT section from NSCLC-Radiogenomics
(CC BY 3.0, redistributable with attribution) and every attention map is a real
forward pass through a canonical checkpoint. Every NUMBER is read from the
archived out-of-fold predictions rather than recomputed, so the demo cannot drift
from the manuscript: `results/ms-s7.json` for the model, `results/baselines.json`
for the clinical comparator.

WHAT THE DEMO MUST NOT DO. It must not present a genotype call. Pooled AUC is
0.618 against 0.794 for smoking status alone, the calibration slope is 0.19 and
the Brier score is worse than predicting the base rate -- so a confident
"EGFR-mutant" badge would be asserting exactly what the paper spent its length
disproving. Cases are chosen to include a confident miss for that reason.

Licence note: NSCLC-Radiogenomics is CC BY 3.0 and its images may be
redistributed with attribution. NSCLC-Radiomics is CC BY-NC 3.0 and appears
nowhere here.

Run: python -m theia.demo.build_demo --ckpt checkpoints/ms-s7/fold0/best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

OUT_DIR = "demo/assets"


def _archived(run_json: str, gene: str = "egfr") -> dict[str, dict]:
    blob = json.load(open(run_json))
    out = {}
    for fold in blob["folds"]:
        for r in fold.get("oof", []):
            out[r["patient_id"]] = {"prob": r.get(f"{gene}_prob"),
                                    "true": r.get(f"{gene}_true"),
                                    "fold": fold["fold"]}
    return out


def _clinical(baselines: str, gene: str = "egfr") -> dict[str, float]:
    blob = json.load(open(baselines)).get("oof", {}).get("clinical", [])
    return {r["patient_id"]: r[f"{gene}_prob"] for r in blob}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    from theia.config import load_config
    from theia.data.dataset import RadiogenomicsDataset, collate, nested_kfold_indices
    from theia.engine.evaluate import grounding_metrics
    from theia.models.theia_model import Theia
    from theia.runtime import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/ms-s7/fold0/best.pt")
    ap.add_argument("--run-json", default="results/ms-s7.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    device = resolve_device("auto")

    archived = _archived(a.run_json)
    clinical = _clinical(a.baselines)
    clin = pd.read_csv(os.path.join(cfg.paths.raw_dir, "clinical",
                                    "clinical.csv")).set_index("Case ID", drop=False)

    state = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = Theia(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)

    rp = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    seed = int((state.get("cfg") or {}).get("seed", cfg.seed))
    fold = int(state.get("fold", 0))
    splits = list(nested_kfold_indices(rp, cfg.split.stratify_on, cfg.split.n_folds,
                                       seed, float(cfg.split.inner_val_frac)))
    held = list(splits[fold][2])
    ds = RadiogenomicsDataset(rp, cfg.data.target_genes)

    # Candidates: held-out, segmented, with a known label AND an archived
    # prediction. Selection is by index order within those, so the panel is not
    # curated on score -- except that we deliberately keep at least one case the
    # model gets confidently wrong, because a demo of this model that shows only
    # hits would misrepresent it.
    cands = []
    for i in held:
        item = ds[i]
        pid = ds.rows[i]["patient_id"]
        rec = archived.get(pid)
        if float(item["roi"].sum()) <= 0 or not rec or rec["true"] not in (0, 1):
            continue
        cands.append((i, item, pid, rec))

    if not cands:
        raise SystemExit("[demo] no eligible held-out cases")

    def err(c):
        return abs(c[3]["prob"] - c[3]["true"])

    picked = cands[: a.n - 1]
    worst = max(cands, key=err)
    if worst[2] not in {c[2] for c in picked}:
        picked = picked[: a.n - 1] + [worst]

    cases = []
    for i, item, pid, rec in picked:
        with torch.no_grad():
            out = model(collate([item], genes=model.genes), device)
        attn = out["attn_maps"][0].amax(0).float().cpu().numpy()
        attn = (attn - attn.min()) / (np.ptp(attn) + 1e-6)

        g = grounding_metrics(out["attn_maps"][:1].cpu(), item["roi"][None], seed=cfg.seed)
        img, roi = item["images"].numpy(), item["roi"].numpy()
        k = int(roi.reshape(roi.shape[0], -1).sum(1).argmax())
        mid, msk = img[k, 0], roi[k, 0]

        ky = max(mid.shape[0] // attn.shape[0], 1)
        kx = max(mid.shape[1] // attn.shape[1], 1)
        big = np.kron(attn, np.ones((ky, kx)))[: mid.shape[0], : mid.shape[1]]

        base = f"{pid.replace('/', '_')}"
        for name, draw in (("ct", False), ("attn", True)):
            fig, ax = plt.subplots(figsize=(4, 4), dpi=170)
            ax.imshow(mid, cmap="gray")
            if draw:
                ax.imshow(big, cmap="inferno", alpha=0.45)
                ax.contour(msk, levels=[0.5], colors="#39d353", linewidths=1.0)
            ax.set_axis_off()
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(os.path.join(a.out, f"{base}_{name}.png"),
                        bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        row = clin.loc[pid] if pid in clin.index else None
        age = pd.to_numeric(row["Age at Histological Diagnosis"], errors="coerce") \
            if row is not None else None
        cases.append({
            "id": pid,
            "age": None if age is None or pd.isna(age) else int(age),
            "sex": (str(row["Gender"]).strip().lower() if row is not None else None),
            "histology": (str(row["Histology "]).strip() if row is not None else None),
            "egfr_true": int(rec["true"]),
            "model_prob": round(float(rec["prob"]), 3),
            "clinical_prob": (round(float(clinical[pid]), 3)
                              if pid in clinical else None),
            "grounding": {k2: round(float(v[0]), 3)
                          for k2, v in g.items() if v},
            "ct": f"assets/{base}_ct.png",
            "attn": f"assets/{base}_attn.png",
        })
        print(f"[demo] {pid}: true={rec['true']} model={rec['prob']:.3f} "
              f"clinical={clinical.get(pid, float('nan')):.3f} "
              f"pointing={cases[-1]['grounding'].get('grounding_pointing')}")

    meta = {
        "checkpoint": a.ckpt, "run": os.path.basename(a.run_json), "fold": fold,
        "collection": "NSCLC-Radiogenomics", "licence": "CC BY 3.0",
        "attribution": "Bakr S, Gevaert O, Echegaray S, et al. Sci Data 2018;5:180202",
        "cohort": {
            "model_auc": 0.618, "model_auc_sd": 0.025,
            "clinical_auc": 0.764, "smoking_auc": 0.794,
            "delta_auc": -0.029, "delta_sd": 0.015,
            "calibration_slope": 0.19, "brier": 0.236, "brier_floor": 0.193,
            "base_rate": 0.261,
            "external_pointing": 0.609, "external_centre_prior": 0.476,
        },
        "cases": cases,
    }
    json.dump(meta, open("demo/cases.json", "w"), indent=2)
    print(f"[demo] wrote demo/cases.json and {2*len(cases)} images to {a.out}")


if __name__ == "__main__":
    main()
