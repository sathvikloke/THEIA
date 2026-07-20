"""Evaluation: prediction, grounding, and rationale quality.

Prediction : ROC-AUC per gene with bootstrap CIs, plus sensitivity/specificity.
Grounding  : IoU / pointing-game between the attention union and the tumor ROI.
Rationale  : BLEU + ROUGE-L against the pseudo-report AS A SANITY CHECK ONLY.
             The real rationale evaluation is the blinded reader study
             (theia/reader_study/), because NLG metrics do not measure clinical
             correctness — reviewers increasingly reject NLG-only evaluation.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> dict:
    model.eval()
    genes = [g.lower() for g in cfg.data.target_genes]
    probs = {g: [] for g in genes}
    ys = {g: [] for g in genes}
    ious = []

    for batch in loader:
        out = model(batch, device)
        for g in genes:
            if g not in batch:
                continue
            p = torch.softmax(out["logits"][g], dim=1)[:, 1].cpu().numpy()
            y = batch[g].numpy()
            keep = y != -1
            probs[g].extend(p[keep].tolist())
            ys[g].extend(y[keep].tolist())
        ious.extend(_grounding_iou(out["attn_maps"], batch["roi"],
                                   cfg.eval.grounding_iou_thresh))

    metrics = {}
    for g in genes:
        if len(set(ys[g])) < 2:
            metrics[f"{g}_auc"] = float("nan")
            continue
        auc = roc_auc_score(ys[g], probs[g])
        lo, hi = _bootstrap_auc(np.array(ys[g]), np.array(probs[g]), cfg.eval.bootstrap_n)
        metrics[f"{g}_auc"] = float(auc)
        metrics[f"{g}_auc_lo"] = float(lo)
        metrics[f"{g}_auc_hi"] = float(hi)
        sens, spec = _sens_spec(np.array(ys[g]), np.array(probs[g]))
        metrics[f"{g}_sens"] = sens
        metrics[f"{g}_spec"] = spec
    metrics["grounding_iou"] = float(np.mean(ious)) if ious else float("nan")
    return metrics


def _grounding_iou(attn_maps: torch.Tensor, roi: torch.Tensor, thresh: float) -> list[float]:
    import torch.nn.functional as F

    # keep everything on CPU: attn_maps come off the model device, roi is CPU.
    attn_maps = attn_maps.detach().cpu()
    roi = roi.detach().cpu()
    b, q, h, w = attn_maps.shape
    union = attn_maps.max(dim=1).values
    union = union / (union.amax(dim=(1, 2), keepdim=True) + 1e-6)
    target = F.interpolate(roi.amax(dim=1), size=(h, w), mode="area").squeeze(1)
    pred = (union > thresh).float()
    tgt = (target > 0.5).float()
    inter = (pred * tgt).sum(dim=(1, 2))
    uni = ((pred + tgt) > 0).float().sum(dim=(1, 2))
    return ((inter + 1e-6) / (uni + 1e-6)).cpu().tolist()


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
