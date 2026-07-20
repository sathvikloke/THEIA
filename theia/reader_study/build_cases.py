"""Build the blinded reader-study case set.

For each sampled patient, emit three arms in randomized, de-identified order:
  theia_full   : predicted status + attention overlay + generated rationale
  label_only   : predicted status only (no explanation) — the control
  ground_truth : the assay-confirmed status (upper bound)

Readers score each presentation without knowing which arm it is. Arm identity and
patient id live only in the key file, which the analyst holds back until scoring
is done. This is what turns "we generated text" into "clinicians rated it."
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from theia.config import load_config
from theia.data.dataset import RadiogenomicsDataset, collate
from theia.models.theia_model import Theia


def _overlay_png(images: np.ndarray, attn: np.ndarray, path: str) -> None:
    """Write a mid-slice ROI with the attention union overlaid as a heat layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = images[len(images) // 2, 0]
    a = attn.max(axis=0)
    a = (a - a.min()) / (a.ptp() + 1e-6)
    a = np.array(np.kron(a, np.ones((mid.shape[0] // a.shape[0], mid.shape[1] // a.shape[1]))))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(mid, cmap="gray")
    ax.imshow(a[: mid.shape[0], : mid.shape[1]], cmap="inferno", alpha=0.45)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


def build(cfg, ckpt: str, out_dir: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Theia(cfg).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])

    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    ds = RadiogenomicsDataset(rows, cfg.data.target_genes)
    rng = random.Random(cfg.seed)
    picks = rng.sample(range(len(ds)), min(cfg.reader_study.n_cases, len(ds)))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_dir = Path(out_dir) / "img"
    img_dir.mkdir(exist_ok=True)
    presentations, key = [], []

    for pid_i in picks:
        item = ds[pid_i]
        batch = collate([item])
        with torch.no_grad():
            out = model(batch, device, generate=True)
        attn = out["attn_maps"][0].cpu().numpy()
        png = str(img_dir / f"{item['patient_id']}.png")
        _overlay_png(item["images"].numpy(), attn, png)

        pred = {g: int(out["logits"][g].argmax(1).item()) for g in model.genes}
        text = out["text"][0]
        truth = {g: int(item[g].item()) for g in model.genes}

        arms = {
            "theia_full": dict(image=os.path.basename(png), status=pred, rationale=text),
            "label_only": dict(status=pred),
            "ground_truth": dict(status=truth),
        }
        order = list(arms.keys())
        rng.shuffle(order)
        for arm in order:
            token = f"P{len(presentations):04d}"
            presentations.append(dict(token=token, arm_payload=arms[arm]))
            key.append(dict(token=token, patient_id=item["patient_id"], arm=arm))

    rng.shuffle(presentations)
    with open(Path(out_dir) / "presentations.json", "w") as fh:
        json.dump(presentations, fh, indent=2)
    with open(Path(out_dir) / "KEY_do_not_open.json", "w") as fh:
        json.dump(key, fh, indent=2)
    print(f"[reader] {len(presentations)} blinded presentations -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="theia/reader_study/cases")
    a = ap.parse_args()
    build(load_config(a.config), a.ckpt, a.out)
