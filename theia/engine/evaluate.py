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
        # A diverged epoch produces NaN logits. sklearn then raises
        # "Input contains NaN" and takes the whole multi-fold run down with it —
        # measured: run 5 died at fold 0 epoch 2 and lost the other 4 folds.
        # Treat it as an epoch with no signal instead: NaN is already handled
        # everywhere downstream (monitor, checkpointing, pooling).
        finite = np.isfinite(probs[g]).all() if probs[g] else True
        if not finite:
            n_bad = int((~np.isfinite(np.asarray(probs[g]))).sum())
            print(f"[eval] {g}: {n_bad}/{len(probs[g])} non-finite predictions "
                  "(model diverged this epoch); reporting NaN")
            metrics[f"{g}_auc"] = float("nan")
            metrics[f"{g}_n"] = float(len(ys[g]))
            metrics[f"{g}_n_pos"] = float(sum(ys[g]))
            continue
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


@torch.no_grad()
def collect_predictions(model, loader, cfg, device, fold: int = 0) -> list[dict]:
    """Per-patient predictions, for pooling out-of-fold across folds.

    Averaging per-fold AUCs is the wrong estimator at this cohort size. With ~23
    EGFR positives spread over 5 folds, a held-out fold holds ~5 positives, and an
    AUC from 5 positives has a 95% CI of roughly +/-0.27 even when the model is
    genuinely good — an interval that includes chance. Averaging those hides the
    instability behind a tidy mean.

    Pooling every fold's out-of-fold prediction into ONE ranking and bootstrapping
    that gives a single estimate over the whole cohort (+/-0.12 at the same
    sample size). Each patient still appears exactly once, and always from a model
    that never saw them in training, so it stays honest.
    """
    model.eval()
    genes = [g.lower() for g in cfg.data.target_genes]
    rows: list[dict] = []
    for batch in loader:
        out = model(batch, device)
        probs = {g: torch.softmax(out["logits"][g].float(), dim=1)[:, 1].cpu().numpy()
                 for g in genes if g in out["logits"]}
        for i, pid in enumerate(batch["patient_id"]):
            rec = {"patient_id": pid, "fold": fold}
            for g in genes:
                if g in probs and g in batch:
                    rec[f"{g}_prob"] = float(probs[g][i])
                    rec[f"{g}_true"] = int(batch[g][i])
            rows.append(rec)
    return rows


def _rank_normalize_by_fold(rows: list[dict], key: str) -> np.ndarray:
    """Map each fold's scores to within-fold ranks in [0,1] before pooling.

    Every fold is a different model, and their probability scales are not
    comparable — one may output 0.4-0.6 while another spans 0.1-0.9. Concatenating
    the raw numbers into one ranking mixes those scales and can drag the pooled
    AUC well below what any individual fold achieved, which looks like a real
    performance drop but is an artefact of the merge.

    AUC only depends on ordering, so converting to within-fold ranks preserves
    exactly what each model got right and makes the folds commensurable.
    """
    out = np.zeros(len(rows))
    folds: dict[object, list[int]] = {}
    for i, r in enumerate(rows):
        folds.setdefault(r.get("fold", 0), []).append(i)
    for idx in folds.values():
        vals = np.array([rows[i][key] for i in idx], dtype=float)
        order = vals.argsort().argsort().astype(float)      # 0..n-1 ties broken stably
        denom = max(len(idx) - 1, 1)
        for j, i in enumerate(idx):
            out[i] = order[j] / denom
    return out


def _warn_single_class_folds(rows: list[dict], gene: str) -> None:
    """A fold with only one class contributes noise to the pooled ranking.

    Within-fold ranks are what make folds commensurable, but a fold whose
    patients are all positive (or all negative) has no internal ordering that
    means anything against the label: its ranks still spread across [0,1] and
    enter the pooled AUC as if they were informative. The pooled number drifts
    toward chance with nothing in the output to say why.

    Not an error — with a rare gene it may be unavoidable, and the roadmap's
    ALK/STK11/TP53 will hit it long before EGFR does. But it must be visible,
    because it is indistinguishable from a genuinely weak model.
    """
    by_fold: dict[object, list[int]] = {}
    for r in rows:
        y = r.get(f"{gene}_true", -1)
        if y != -1:
            by_fold.setdefault(r.get("fold", 0), []).append(int(y))
    bad = [f for f, ys in by_fold.items() if len(ys) > 1 and len(set(ys)) < 2]
    if bad:
        print(f"[eval] WARNING: {gene.upper()} fold(s) {sorted(map(str, bad))} hold a "
              "single class; their within-fold ranks carry no signal and drag the "
              "pooled AUC toward chance. Treat the pooled estimate as a lower bound.")


def pooled_metrics(rows: list[dict], genes: list[str], bootstrap_n: int = 2000) -> dict:
    """One AUC per gene over the pooled out-of-fold predictions, with a CI."""
    out: dict[str, float] = {"n_patients": float(len(rows))}
    for g in [x.lower() for x in genes]:
        keep = [r for r in rows if r.get(f"{g}_true", -1) != -1 and f"{g}_prob" in r]
        if not keep:
            out[f"{g}_n"] = 0.0
            out[f"{g}_auc"] = float("nan")
            continue
        y = np.array([r[f"{g}_true"] for r in keep])
        raw = np.array([r[f"{g}_prob"] for r in keep], dtype=float)
        if not np.isfinite(raw).all():
            bad = int((~np.isfinite(raw)).sum())
            print(f"[eval] pooled {g}: dropping {bad} non-finite out-of-fold predictions")
            ok = np.isfinite(raw)
            keep = [r for r, k in zip(keep, ok) if k]
            y = y[ok]
            if len(set(y.tolist())) < 2:
                out[f"{g}_n"] = float(len(y)); out[f"{g}_auc"] = float("nan"); continue
        _warn_single_class_folds(keep, g)
        p = _rank_normalize_by_fold(keep, f"{g}_prob")
        out[f"{g}_n"] = float(len(y))
        out[f"{g}_n_pos"] = float(y.sum()) if len(y) else 0.0
        if len(set(y.tolist())) < 2:
            out[f"{g}_auc"] = float("nan")
            continue
        out[f"{g}_auc"] = float(roc_auc_score(y, p))
        lo, hi = _bootstrap_auc(y, p, bootstrap_n)
        out[f"{g}_auc_lo"], out[f"{g}_auc_hi"] = float(lo), float(hi)
        sens, spec = _sens_spec(y, p)
        out[f"{g}_sens"], out[f"{g}_spec"] = sens, spec
    return out


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
    # Peak-to-mean of the attention union. At 1.0 the map is uniform, the model
    # is pointing nowhere, and every localization number above is meaningless —
    # the argmax then falls wherever float noise puts it (in practice cell 0,
    # which reads as a confident, reproducible pointing score of exactly 0.000).
    # Measured 1.03 on an untrained head, so treat anything under ~1.5 as "no
    # localization yet" rather than "localizes badly".
    out["grounding_peak_ratio"] = (
        union.amax(dim=1) / union.mean(dim=1).clamp_min(EPS)).tolist()
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
