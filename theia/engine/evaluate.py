"""Evaluation: prediction, grounding, and rationale quality.

Prediction : ROC-AUC per gene with bootstrap CIs, plus sensitivity/specificity.
Grounding  : attention mass inside the ROI, a pointing game, and an area-matched
             IoU — each reported against a spatially-shuffled baseline.
Rationale  : BLEU + ROUGE-L against the pseudo-report AS A SANITY CHECK ONLY.
             The real rationale evaluation is the blinded reader study
             (theia/reader_study/), because NLG metrics do not measure clinical
             correctness — reviewers increasingly reject NLG-only evaluation.

Why the grounding metrics were rewritten
----------------------------------------
The previous `grounding_iou` peak-normalised the attention union and thresholded
at a fixed 0.1. After dividing by the per-sample max, almost every cell clears
0.1, so the prediction was effectively all-ones and the "IoU" just reported the
tumor's area fraction in the crop. Measured on synthetic data, uniform attention
and *random* attention scored identically to three decimals (0.327 / 0.510 /
0.735 as the tumor filled 30 / 50 / 70% of the frame). The metric could not
distinguish a trained model from noise, which matters because grounding is the
headline claim.

Every grounding number now ships next to `*_shuffled`, the same statistic on a
spatially permuted copy of the model's own attention. That is an assumption-free
chance baseline: if `grounding_mass` is not clearly above `grounding_mass_shuffled`,
the model has not localised anything, and the gap (`grounding_mass_lift`) is the
number worth reporting.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from theia.engine.losses import roi_to_grid

EPS = 1e-8


@torch.no_grad()
def evaluate(model, loader, cfg, device, full: bool = False) -> dict:
    """Compute validation metrics.

    full=False (per-epoch, for early stopping): AUC + grounding, no bootstrap.
    full=True  (final, for reporting): adds bootstrap CIs on every AUC.
    """
    model.eval()
    genes = [g.lower() for g in cfg.data.target_genes]
    probs = {g: [] for g in genes}
    ys = {g: [] for g in genes}
    ground: dict[str, list[float]] = {}

    for batch in loader:
        out = model(batch, device)
        for g in genes:
            if g not in batch:
                continue
            p = torch.softmax(out["logits"][g].float(), dim=1)[:, 1].cpu().numpy()
            y = batch[g].numpy()
            keep = y != -1
            probs[g].extend(p[keep].tolist())
            ys[g].extend(y[keep].tolist())
        for k, v in grounding_metrics(out["attn_maps"], batch["roi"]).items():
            ground.setdefault(k, []).extend(v)

    metrics: dict[str, float] = {}
    for g in genes:
        if len(set(ys[g])) < 2:
            # Undefined, not zero. train.py treats NaN as "no signal this epoch"
            # rather than letting it silently block checkpointing forever.
            metrics[f"{g}_auc"] = float("nan")
            metrics[f"{g}_n_pos"] = float(sum(ys[g]))
            metrics[f"{g}_n"] = float(len(ys[g]))
            continue
        auc = roc_auc_score(ys[g], probs[g])
        metrics[f"{g}_auc"] = float(auc)
        metrics[f"{g}_n_pos"] = float(sum(ys[g]))
        metrics[f"{g}_n"] = float(len(ys[g]))
        sens, spec = _sens_spec(np.array(ys[g]), np.array(probs[g]))
        metrics[f"{g}_sens"] = sens
        metrics[f"{g}_spec"] = spec
        if full:
            lo, hi = _bootstrap_auc(np.array(ys[g]), np.array(probs[g]), cfg.eval.bootstrap_n)
            metrics[f"{g}_auc_lo"] = float(lo)
            metrics[f"{g}_auc_hi"] = float(hi)

    for k, v in ground.items():
        metrics[k] = float(np.mean(v)) if v else float("nan")
    for base in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        real, sham = metrics.get(base), metrics.get(f"{base}_shuffled")
        if real is not None and sham is not None:
            metrics[f"{base}_lift"] = real - sham
    return metrics


def grounding_metrics(attn_maps: torch.Tensor, roi: torch.Tensor,
                      seed: int = 0) -> dict[str, list[float]]:
    """Per-sample grounding statistics, each paired with a shuffled baseline.

    grounding_mass     : share of attention mass falling inside the ROI.
    grounding_pointing : is the single most-attended cell inside the ROI?
    grounding_iou      : IoU when the attention is thresholded at the quantile
                         that makes |prediction| == |ROI|, so the score reflects
                         *where* the model looks, not how much of the frame the
                         tumor happens to occupy.
    """
    attn = attn_maps.detach().float().cpu()
    roi = roi.detach().float().cpu()
    b, q, h, w = attn.shape
    union = attn.amax(dim=1).reshape(b, -1)                     # [B, h*w]
    target = roi_to_grid(roi, h, w, torch.device("cpu")).reshape(b, -1)

    valid = target.sum(dim=1) > 0
    out: dict[str, list[float]] = {}
    if not bool(valid.any()):
        return out
    union, target = union[valid], target[valid]

    g = torch.Generator().manual_seed(seed)
    perm = torch.argsort(torch.rand(union.shape, generator=g), dim=1)
    shuffled = torch.gather(union, 1, perm)                     # same values, no structure

    for name, u in (("", union), ("_shuffled", shuffled)):
        out[f"grounding_mass{name}"] = _mass_in_mask(u, target)
        out[f"grounding_pointing{name}"] = _pointing(u, target)
        out[f"grounding_iou{name}"] = _area_matched_iou(u, target)
    out["grounding_roi_frac"] = (target.mean(dim=1)).tolist()
    return out


def _mass_in_mask(u: torch.Tensor, t: torch.Tensor) -> list[float]:
    return ((u * t).sum(dim=1) / u.sum(dim=1).clamp_min(EPS)).tolist()


def _pointing(u: torch.Tensor, t: torch.Tensor) -> list[float]:
    hit = t.gather(1, u.argmax(dim=1, keepdim=True)).squeeze(1)
    return hit.tolist()


def _area_matched_iou(u: torch.Tensor, t: torch.Tensor) -> list[float]:
    """Threshold each map at its own top-k, where k = |ROI| for that sample."""
    scores = []
    order = torch.argsort(u, dim=1, descending=True)
    for i in range(u.shape[0]):
        k = int(t[i].sum().item())
        pred = torch.zeros_like(t[i])
        pred[order[i, :k]] = 1.0
        inter = float((pred * t[i]).sum())
        union_sz = float(((pred + t[i]) > 0).float().sum())
        scores.append(inter / union_sz if union_sz else float("nan"))
    return scores


def _bootstrap_auc(y, p, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _sens_spec(y, p, thresh: float = 0.5) -> tuple[float, float]:
    pred = (p >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def score_rationale(pred: str, ref: str) -> dict:
    """Sanity-check NLG metrics only. Not the primary rationale evaluation."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
    rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred)["rougeL"].fmeasure
    return {"bleu": bleu, "rougeL": rl}


@torch.no_grad()
def score_rationales(model, loader, device, limit: int = 64) -> dict:
    """Generate rationales and score them against the pseudo-reports.

    `score_rationale` existed but nothing ever called it, so `nltk` and
    `rouge-score` were installed and unused. This wires it into the final
    evaluation path.
    """
    model.eval()
    bleu, rouge, n = [], [], 0
    for batch in loader:
        out = model(batch, device, generate=True)
        for pred, ref in zip(out["text"], batch["report"]):
            s = score_rationale(pred, ref)
            bleu.append(s["bleu"])
            rouge.append(s["rougeL"])
            n += 1
            if n >= limit:
                break
        if n >= limit:
            break
    if not bleu:
        return {}
    return {"rationale_bleu": float(np.mean(bleu)),
            "rationale_rougeL": float(np.mean(rouge)),
            "rationale_n": float(n)}
