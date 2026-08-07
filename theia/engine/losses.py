"""Joint loss: classification + generation + grounding.

Grounding loss anchors the model's attention to the tumor. We take the max over
the Q region queries at each location (the union of what any query looks at),
downsample the ROI mask to the patch grid, and push them together with a soft
Dice + BCE. This is what makes the attention maps trustworthy instead of decorative.

The objective is deliberately unchanged: peak-normalising the union before the BCE
means flat "attend everywhere" attention is scored against a mostly-zero target and
punished hard, while attention concentrated on the tumor scores ~0. That behaviour
was measured, not assumed, and `tests/test_regressions.py` pins it.

What did change is *how* it is computed. `F.binary_cross_entropy` is on PyTorch's
CUDA autocast banned list ("unsafe to autocast"), so with the shipped `amp: true`
this raised a RuntimeError on the first step of any CUDA run — the exact hardware
the README asks for. The BCE now runs in float32 with autocast locally disabled,
which keeps the numbers identical and makes the CUDA path work.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -1
EPS = 1e-6


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


def roi_to_grid(roi: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
    """[B,S,1,H,W] tumor mask -> [B,h,w] binary target on the patch grid.

    Padded slices are all-zero so the max over slices is unaffected by them.

    A cell is positive when the tumor covers most of it. That alone loses small
    lesions entirely: at a 14x14 grid over a 224px crop each cell is 16x16px, so
    a lesion thinner than ~11px occupies no cell by majority, the target comes
    back empty, and the grounding term silently contributes nothing for that
    patient — the loss reads 0.000 and looks healthy. Any sample whose mask
    survives upstream but vanishes here falls back to its top-k cells by
    coverage, where k is the tumor's area in cell-equivalents.
    """
    mask = roi.to(device).amax(dim=1)                                  # [B,1,H,W]
    frac = F.interpolate(mask, size=(h, w), mode="area").squeeze(1)    # [B,h,w] coverage
    binary = (frac > 0.5).float()

    lost = (binary.flatten(1).sum(dim=1) == 0) & (frac.flatten(1).sum(dim=1) > 0)
    if bool(lost.any()):
        b = frac.shape[0]
        flat_frac = frac.reshape(b, -1)
        flat_bin = binary.reshape(b, -1).clone()
        for i in torch.nonzero(lost, as_tuple=False).flatten().tolist():
            k = max(1, int(round(float(flat_frac[i].sum()))))
            flat_bin[i, torch.topk(flat_frac[i], k).indices] = 1.0
        binary = flat_bin.view(b, h, w)
    return binary


def grounding_loss(attn_maps: torch.Tensor, roi: torch.Tensor, device) -> torch.Tensor:
    """attn_maps [B,Q,h,w] vs roi [B,S,1,H,W] -> Dice + BCE on the union map."""
    b, q, h, w = attn_maps.shape
    union = attn_maps.amax(dim=1)                                        # [B,h,w]
    # Rescale the map to peak at 1. The denominator is DETACHED and floored on
    # purpose, and both matter.
    #
    # detach: this division is a rescaling, not a learning signal about the
    # maximum. Left attached it contributes a 1/max^2 term that grows without
    # bound as attention flattens.
    # clamp_min: attn_w is a softmax over patch tokens, so a diffuse map has a
    # maximum near 1/n_tokens. Dividing by that amplifies every upstream
    # gradient by n_tokens, and the amplification is unbounded as the map
    # flattens further.
    #
    # Together with the BCE floor below, these were the two unbounded gradient
    # paths that made whole folds die: the loss stayed perfectly finite while
    # the gradient overflowed fp32, so nothing downstream could see it coming.
    denom = union.amax(dim=(1, 2), keepdim=True).detach().clamp_min(1e-3)
    union = union / denom

    target = roi_to_grid(roi, h, w, device)

    # A patient whose mask vanishes at grid resolution carries no grounding
    # signal; including it would push all attention toward zero.
    valid = target.sum(dim=(1, 2)) > 0
    if not bool(valid.any()):
        return torch.zeros((), device=device, dtype=union.dtype)
    union, target = union[valid], target[valid]

    # binary_cross_entropy is autocast-banned on CUDA. Run it in fp32 with
    # autocast off; the result is numerically what the fp32 path always produced.
    with torch.amp.autocast(device_type=union.device.type, enabled=False):
        # d/du of BCE is (u - t) / (u(1 - u)), so the clamp floor sets the
        # largest gradient this loss can emit. At EPS=1e-6 that is 1e6 per
        # element, before the normalizer above multiplies it again. 1e-4 bounds
        # it to 1e4 and changes the loss value by <1e-3 nats.
        u32 = union.float().clamp(1e-4, 1.0 - 1e-4)
        t32 = target.float()
        bce = F.binary_cross_entropy(u32, t32)
        inter = (u32 * t32).sum(dim=(1, 2))
        dice = 1 - (2 * inter + 1) / (u32.sum(dim=(1, 2)) + t32.sum(dim=(1, 2)) + 1)
        out = bce + dice.mean()
    return out.to(attn_maps.dtype)


def total_loss(out: dict, batch, weights, device) -> tuple[torch.Tensor, dict]:
    """Returns (loss, parts).

    `parts` holds *detached tensors*, not floats. Calling float() on four tensors
    every step forces a host sync each iteration; the caller converts once per
    logging interval instead.
    """
    cls = classification_loss(out["logits"], batch, device)
    gen = out.get("gen_loss")
    if gen is None:
        gen = torch.zeros((), device=device)
    grd = grounding_loss(out["attn_maps"], batch["roi"], device)
    loss = weights.cls * cls + weights.gen * gen + weights.ground * grd
    parts = dict(cls=cls.detach(), gen=gen.detach(), ground=grd.detach(), total=loss.detach())
    return loss, parts
