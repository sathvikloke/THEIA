"""Does the attention follow the lesion, or does it just look at the middle?

THE CRITICISM THIS ANSWERS. Inputs are lesion-centred crops, so a Gaussian blob
pinned to the frame centre already scores 0.270 attention mass against the trained
model's 0.292. The manuscript answers that statistically -- pointing accuracy and
area-matched IoU are invariant to the prior's width, and the trained model beats it
on both -- but a statistical answer to "your task is too easy" is weaker than a
direct one. This is the direct one.

THE TEST. Translate the whole crop so the lesion is no longer centred, by a
fraction f of the frame, and translate the reference ROI with it. Nothing else
changes: same checkpoints, same patients, same metric code. A model that genuinely
localises should keep pointing at the tumour. A model that has learned "attend to
the middle" should fall off a cliff, and so should the centre prior, which is the
control that makes the comparison legible.

Padding is the per-volume minimum. These crops are HU-windowed to [0,1] at centre
-600 / width 1500, so air (-1000 HU) sits at ~0.22 and the minimum IS air -- the
vacated region is filled with something physically plausible rather than with black,
which would be a -1350 HU signal the encoder has never seen.

DIRECTION. Shifting every patient the same way would let the model exploit one
direction, and would confound the result with any anisotropy in the crop. Each
patient is displaced along a different angle, fixed by its index so the run is
deterministic.

CHOOSING f. Training used `crop_jitter_frac` 0.30, so lesions up to ~30% off centre
are IN distribution. The ROI covers ~4.9% of the frame, a radius of ~0.125, so past
f ~ 0.37 the lesion starts leaving the frame and the test stops being about
attention. f = 0.20 is inside the training regime and f = 0.35 is at the edge of it.
f = 0 is not recomputed here; it is the archived run, produced by the same code.

Run: python -m theia.analysis.fov_stress
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from theia.analysis.external_grounding import _loader, centre_prior, summarise


def shift_batch(images: torch.Tensor, roi: torch.Tensor, frac: float,
                index0: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate images and ROI together by `frac` of the frame.

    images, roi: [B, S, 1, H, W]. Each sample b is displaced along angle
    2*pi*((index0 + b) mod 8)/8, so directions are spread but reproducible.
    """
    b, s, c, h, w = images.shape
    out_i, out_r = images.clone(), roi.clone()
    for k in range(b):
        theta = 2 * math.pi * ((index0 + k) % 8) / 8.0
        dy = int(round(frac * h * math.sin(theta)))
        dx = int(round(frac * w * math.cos(theta)))
        if dy == 0 and dx == 0:
            continue
        fill = float(images[k].min())
        img = torch.full_like(images[k], fill)
        msk = torch.zeros_like(roi[k])
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        yd0, yd1 = max(0, -dy), min(h, h - dy)
        xd0, xd1 = max(0, -dx), min(w, w - dx)
        img[..., ys0:ys1, xs0:xs1] = images[k][..., yd0:yd1, xd0:xd1]
        msk[..., ys0:ys1, xs0:xs1] = roi[k][..., yd0:yd1, xd0:xd1]
        out_i[k], out_r[k] = img, msk
    return out_i, out_r


@torch.no_grad()
def attention_shifted(model, loader, device, frac: float):
    """Same forward pass as the external evaluation, on displaced crops."""
    model.eval()
    maps, rois = [], []
    seen = 0
    for batch in loader:
        images, roi = shift_batch(batch["images"], batch["roi"], frac, seen)
        seen += images.shape[0]
        images = images.to(device)
        sm = batch.get("slice_mask")
        tokens, grid = model.vision.encode(
            images, sm.to(device) if sm is not None else None)
        g = model.grounding(tokens, grid)
        maps.append(g["attn_maps"].detach().float().cpu())
        rois.append(roi.detach().float().cpu())
    return maps, rois


def main() -> None:
    from theia.config import load_config
    from theia.models.theia_model import Theia
    from theia.runtime import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default=None,
                    help="comma-separated run ids; default is CANONICAL.json")
    ap.add_argument("--rows", default="data/processed_pretrain/rows.jsonl")
    ap.add_argument("--offsets", default="0.20,0.35")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/fov_stress.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.runs:
        runs = a.runs.split(",")
    else:
        runs = [os.path.basename(p).replace(".json", "")
                for p in json.load(open("results/CANONICAL.json"))["headline_runs"]]
    offsets = [float(x) for x in a.offsets.split(",")]

    device = resolve_device("auto")
    loader = _loader(a.rows, cfg.data.target_genes, cfg.train.batch_size, a.limit)
    print(f"[fov] {len(loader.dataset)} external patients, runs={runs}, "
          f"offsets={offsets}, device={device}\n")

    ckpts = [c for run in runs
             for c in sorted(glob.glob(f"checkpoints/{run}/fold*/best.pt"))]
    if not ckpts:
        raise SystemExit("[fov] no checkpoints found")

    results: dict[str, list[dict]] = {}
    controls: dict[str, dict] = {}
    for frac in offsets:
        rows = []
        last_maps = last_rois = None
        for ck in ckpts:
            run = os.path.basename(os.path.dirname(os.path.dirname(ck)))
            fold = os.path.basename(os.path.dirname(ck))
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            model = Theia(cfg).to(device)
            want = {k: v for k, v in blob["model"].items()
                    if k.startswith("vision.") or k.startswith("grounding.")}
            missing, _ = model.load_state_dict(want, strict=False)
            if [k for k in missing if k.startswith(("vision.", "grounding."))]:
                print(f"[fov] skip {run}/{fold}")
                continue
            maps, rois = attention_shifted(model, loader, device, frac)
            r = summarise(maps, rois, seed=cfg.seed)
            r.update(run=run, fold=fold, offset=frac)
            rows.append(r)
            last_maps, last_rois = maps, rois
            print(f"[fov] f={frac:.2f} {run}/{fold}: mass {r['grounding_mass']:.3f} "
                  f"pointing {r['grounding_pointing']:.3f} iou {r['grounding_iou']:.3f}")
            del model

        # The centre prior is unchanged -- it is pinned to the frame centre by
        # definition. Scoring it against the DISPLACED ROI is the whole point.
        roi_frac = float(np.mean([r["grounding_roi_frac"] for r in rows]))
        cp = centre_prior(last_maps[0].shape, roi_frac)
        ctrl = summarise([cp[: m.shape[0]] for m in last_maps], last_rois,
                         seed=cfg.seed)
        results[f"{frac:.2f}"] = rows
        controls[f"{frac:.2f}"] = ctrl
        print(f"[fov] f={frac:.2f} CENTRE PRIOR: mass {ctrl['grounding_mass']:.3f} "
              f"pointing {ctrl['grounding_pointing']:.3f} "
              f"iou {ctrl['grounding_iou']:.3f}\n")

    print(f"{'offset':>7} {'trained pointing':>17} {'centre pointing':>16} "
          f"{'trained iou':>12} {'centre iou':>11}")
    summary = {}
    for k in results:
        tp = float(np.mean([r["grounding_pointing"] for r in results[k]]))
        ti = float(np.mean([r["grounding_iou"] for r in results[k]]))
        summary[k] = {"trained_pointing": tp, "trained_iou": ti,
                      "centre_pointing": controls[k]["grounding_pointing"],
                      "centre_iou": controls[k]["grounding_iou"],
                      "trained_mass": float(np.mean(
                          [r["grounding_mass"] for r in results[k]])),
                      "centre_mass": controls[k]["grounding_mass"],
                      "n_checkpoints": len(results[k])}
        print(f"{k:>7} {tp:>17.3f} {controls[k]['grounding_pointing']:>16.3f} "
              f"{ti:>12.3f} {controls[k]['grounding_iou']:>11.3f}")

    json.dump({"n_external": len(loader.dataset), "runs": runs,
               "offsets": offsets, "summary": summary,
               "per_checkpoint": results, "centre_prior": controls,
               "note": "offset 0.00 is the archived external_grounding_canonical "
                       "run, produced by the same code path and not recomputed"},
              open(a.out, "w"), indent=2)
    print(f"\n[fov] wrote {a.out}")


if __name__ == "__main__":
    main()
