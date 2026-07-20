"""Joint loss: classification + generation + grounding.

Grounding loss anchors the model's attention to the tumor. We take the max over
the Q region queries at each location (the union of what any query looks at),
downsample the ROI mask to the patch grid, and push them together with a soft
Dice + BCE. This is what makes the attention maps trustworthy instead of decorative.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -1


def classification_loss(logits: dict, batch, device) -> torch.Tensor:
    total, n = torch.zeros((), device=device), 0
    for gene, log in logits.items():
        if gene not in batch:
            continue
        y = batch[gene].to(device)
        mask = y != IGNORE
        if mask.any():
            total = total + F.cross_entropy(log[mask], y[mask])
            n += 1
    return total / max(n, 1)


def grounding_loss(attn_maps: torch.Tensor, roi: torch.Tensor, device) -> torch.Tensor:
    """attn_maps [B,Q,h,w] vs roi [B,S,1,H,W] -> Dice + BCE on the union map."""
    b, q, h, w = attn_maps.shape
    union = attn_maps.max(dim=1).values                         # [B,h,w]
    union = union / (union.amax(dim=(1, 2), keepdim=True) + 1e-6)

    target = roi.to(device).amax(dim=1)                         # [B,1,H,W] union over slices
    target = F.interpolate(target, size=(h, w), mode="area").squeeze(1)  # [B,h,w]
    target = (target > 0.5).float()

    bce = F.binary_cross_entropy(union.clamp(1e-6, 1 - 1e-6), target)
    inter = (union * target).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (union.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


def total_loss(out: dict, batch, weights, device) -> tuple:
    cls = classification_loss(out["logits"], batch, device)
    gen = out.get("gen_loss", torch.zeros((), device=device))
    grd = grounding_loss(out["attn_maps"], batch["roi"], device)
    loss = weights.cls * cls + weights.gen * gen + weights.ground * grd
    return loss, dict(cls=float(cls), gen=float(gen), ground=float(grd), total=float(loss))
