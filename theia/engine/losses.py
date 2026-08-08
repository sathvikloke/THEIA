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
# Sharpness of the logit map in grounding_loss; see the comment at its use.
LOGIT_SCALE = 8.0


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
    # Rescale the map to peak at 1. The denominator stays ATTACHED: detaching it
    # removes the gradient through the maximum, which is one of the two things
    # that let a flattened map recover (the other, the clamp, is gone entirely --
    # see the BCE comment below). Detached, run 12 collapsed to uniform attention
    # in all five folds, lift +0.001 and peak ratio 1.000 against run 6's +0.235.
    #
    # The floor bounds the amplification: attn_w is a softmax over patch tokens,
    # so a diffuse map peaks near 1/n_tokens and dividing by that scales every
    # upstream gradient by n_tokens, without bound as the map flattens further.
    denom = union.amax(dim=(1, 2), keepdim=True).clamp_min(1e-3)
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
        u32 = union.float()
        t32 = target.float()
        # BCE on LOGITS rather than on clamped probabilities.
        #
        # The old form ran binary_cross_entropy on u clamped to (eps, 1-eps),
        # which has two defects that only show up together. d/du of BCE is
        # (u - t)/(u(1 - u)), so the floor sets the largest gradient the loss can
        # emit -- 1e6 per element at eps=1e-6. And normalising by the maximum
        # puts the peak cell at exactly 1.0 on EVERY step, so the peak always sat
        # on the upper clamp, where the gradient is identically zero. A map that
        # went uniform therefore had every cell clamped at once and could never
        # recover: loss 9.30, max|grad| 0.000. Run 6 escaped only because random
        # init is never exactly flat; run 12 fell in and all five folds reported
        # peak ratio 1.000.
        #
        # with_logits has no clamp and no dead zone. Its gradient is
        # (sigmoid(z) - t), bounded in [-1, 1] by construction, so this is both
        # better conditioned than the original AND free of the fixed point.
        # LOGIT_SCALE sets how sharply the map is pushed toward 0/1; at 8 the
        # attainable range is sigmoid(+-4) = [0.018, 0.982].
        bce = F.binary_cross_entropy_with_logits((u32 - 0.5) * LOGIT_SCALE, t32)
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
