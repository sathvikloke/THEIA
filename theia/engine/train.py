"""Training loop — k-fold aware, LoRA fine-tune, AMP, early stopping.

Run: python -m theia.engine.train --config configs/default.yaml
Trains one model per fold and writes checkpoints + a metrics log per fold.

Fixed here:
  * Device selection was `cuda if available else cpu`, which ignored Apple
    Silicon entirely. Now goes through theia.runtime (cuda > mps > cpu).
  * `torch.cuda.amp` was requested unconditionally; off CUDA it disables itself
    with a warning, so `amp: true` was quietly a lie. AMP is now device-aware and
    the resolved state is printed.
  * A NaN monitor (single-class validation fold) made `nan > best` False for every
    epoch, so `best.pt` was NEVER written for that fold, `best` stayed -1.0 and
    polluted the CV mean, and run_eval.sh died on a missing checkpoint. NaN is now
    handled explicitly and a checkpoint is always written.
  * The scheduler stepped even when the GradScaler skipped the optimizer step.
  * The CV summary printed (max-min)/2 as "±", which is a half-range, not a
    standard deviation or a confidence interval.
  * Model selection ran on the same split that was reported. Nested splits are
    now the default: early stopping uses an inner validation set, and the outer
    fold is scored once, after the model is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from theia.config import load_config
from theia.data.dataset import (RadiogenomicsDataset, kfold_indices, make_collate,
                                nested_kfold_indices)
from theia.engine.evaluate import collect_predictions, evaluate, pooled_metrics
from theia.engine.losses import total_loss
from theia.models.theia_model import Theia
from theia.runtime import amp_settings, autocast, describe, make_grad_scaler, resolve_device


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(cfg, rows, genes, idx, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = RadiogenomicsDataset(rows, genes, idx)
    workers = int(getattr(cfg.train, "num_workers", 2))
    return DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
        collate_fn=make_collate(genes), num_workers=workers,
        drop_last=drop_last and len(ds) > cfg.train.batch_size,
        persistent_workers=workers > 0,
    )


def make_loaders(cfg, train_idx, val_idx, test_idx=None):
    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    genes = cfg.data.target_genes
    tl = _loader(cfg, rows, genes, train_idx, shuffle=True, drop_last=True)
    vl = _loader(cfg, rows, genes, val_idx, shuffle=False, drop_last=False)
    testl = _loader(cfg, rows, genes, test_idx, shuffle=False, drop_last=False) if test_idx else None
    return tl, vl, testl


def _is_better(candidate: float, best: float) -> bool:
    """NaN is 'no signal', never an improvement — and never a permanent block."""
    return not math.isnan(candidate) and candidate > best


def monitor_value(metrics: dict, monitor) -> float:
    """Resolve train.monitor, which may be one key or several to be summed.

    Selecting on `egfr_auc` alone and then *reporting* grounding means the
    grounding number comes from whichever epoch happened to win on AUC.
    Measured: fold 1 selected epoch 4 (lift +0.007) while its best grounding sat
    at epoch 12 (+0.069). A composite monitor keeps an epoch that does both.

    Accepts "egfr_auc", ["egfr_auc", "grounding_mass_lift"], or
    [["egfr_auc", 1.0], ["grounding_mass_lift", 0.5]].
    """
    if isinstance(monitor, str):
        return float(metrics.get(monitor, float("nan")))
    total = 0.0
    for item in monitor:
        key, weight = (item, 1.0) if isinstance(item, str) else (item[0], float(item[1]))
        v = metrics.get(key, float("nan"))
        if v != v:                      # NaN in any component invalidates the sum
            return float("nan")
        total += weight * float(v)
    return total


def _param_groups(model, cfg) -> list[dict]:
    """Discriminative learning rates: pretrained vision blocks train slower.

    With `unfreeze_last_n`, the re-enabled ViT blocks carry pretrained weights but
    were getting the same LR as randomly-initialised heads. That is far too high
    for them and is the likely cause of grounding's fold-to-fold instability —
    measured mass lift ranged 0.000 to 0.217 across five folds on identical
    settings, and the same fold swung 0.302 -> 0.004 between runs.

    Vision parameters get lr * train.backbone_lr_mult; heads keep the full rate.
    """
    mult = float(getattr(cfg.train, "backbone_lr_mult", 1.0))
    vision, heads = [], []
    vision_ids = {id(p) for p in model.vision.parameters()}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (vision if id(p) in vision_ids else heads).append(p)
    if not vision or mult == 1.0:
        return [{"params": vision + heads, "lr": cfg.train.lr}]
    return [
        {"params": vision, "lr": cfg.train.lr * mult},
        {"params": heads, "lr": cfg.train.lr},
    ]


def train_fold(cfg, fold: int, train_idx, val_idx, test_idx, device) -> dict:
    tl, vl, testl = make_loaders(cfg, train_idx, val_idx, test_idx)
    amp_on, amp_dtype = amp_settings(device, cfg.train.amp)
    model = Theia(cfg).to(device)
    opt = torch.optim.AdamW(_param_groups(model, cfg), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    # scheduler steps once per OPTIMIZER step (every grad_accum batches), so
    # total_steps must match that count or OneCycleLR raises past the end.
    opt_steps_per_epoch = max(len(tl) // cfg.train.grad_accum, 1)
    steps = cfg.train.epochs * opt_steps_per_epoch
    # max_lr must be per-group, or OneCycleLR overwrites the discriminative rates
    # set in _param_groups and drives the pretrained blocks at the head rate.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps,
        pct_start=cfg.train.warmup_ratio)
    scaler = make_grad_scaler(device, amp_on)

    best, best_epoch, patience = -float("inf"), -1, 0
    ckpt_dir = Path(cfg.paths.ckpt_dir) / f"fold{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def save(path: Path, metrics: dict) -> None:
        torch.save({"model": model.state_dict(), "cfg": dict(cfg), "metrics": metrics,
                    "fold": fold, "lora_applied": model.lora_applied}, path)

    for epoch in range(cfg.train.epochs):
        model.train()
        opt.zero_grad()
        pbar = tqdm(tl, desc=f"fold{fold} ep{epoch}")
        running, seen, skipped, overflowed = None, 0, 0, 0
        for step, batch in enumerate(pbar):
            with autocast(device, amp_on, amp_dtype):
                out = model(batch, device)
                loss, parts = total_loss(out, batch, cfg.train.loss_weights, device)
                loss = loss / cfg.train.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                prev_scale = scaler.get_scale()
                # Clip, and skip the step outright if the gradient is not finite.
                #
                # On CUDA, GradScaler already skips non-finite steps. Off CUDA
                # (MPS, CPU) AMP is disabled, the scaler is a no-op, and NOTHING
                # skipped them: one bad batch wrote NaN into the weights and the
                # fold was dead from that point on, still burning epochs until
                # early stopping. Fold 0 of run 10 went NaN at epoch 8 and kept
                # its epoch-0 checkpoint, i.e. contributed an untrained model to
                # the pooled estimate. The old code detected this and printed a
                # warning, but had no way to act on it.
                #
                # unscale_ is a no-op when the scaler is disabled, so the clip
                # sees true gradients on every backend.
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None],
                    float(getattr(cfg.train, "grad_clip", 1.0)))
                if torch.isfinite(gnorm):
                    scaler.step(opt)
                    scaler.update()
                else:
                    # Distinguish the two ways this happens, because they need
                    # opposite responses and look identical from the norm alone:
                    #   every grad finite -> the SUM OF SQUARES overflowed fp32
                    #     while each element was representable. The gradients are
                    #     usable and merely enormous.
                    #   some grad non-finite -> a true NaN/inf. Skipping is right.
                    # Without this the log says "non-finite" for both and there is
                    # no way to tell which you are looking at.
                    grads_ok = all(torch.isfinite(p.grad).all()
                                   for p in model.parameters() if p.grad is not None)
                    skipped += 1
                    if grads_ok:
                        overflowed += 1
                    scaler.update()
                opt.zero_grad()
                # A skipped step (inf/nan gradients) must not advance the LR
                # schedule, or OneCycleLR drifts out of sync with real progress.
                if torch.isfinite(gnorm) and scaler.get_scale() >= prev_scale:
                    sched.step()
            # parts are detached tensors; converting every step would force a
            # host sync per iteration. Accumulate and format occasionally.
            if not torch.isfinite(parts["total"]):
                print(f"[train] fold{fold} ep{epoch} step{step}: non-finite loss "
                      f"({ {k: float(v) for k, v in parts.items()} }); "
                      "the fold has diverged")
            running = parts if running is None else {k: running[k] + parts[k] for k in parts}
            seen += 1
            if step % 20 == 0:
                pbar.set_postfix({k: f"{float(v) / seen:.3f}" for k, v in running.items()})

        metrics = evaluate(model, vl, cfg, device)
        metrics["epoch"] = epoch
        metrics["split"] = "inner_val"
        if skipped:
            # Recorded, not just printed: a fold that skipped many steps trained
            # on less than it appears to, and that belongs in the archive.
            metrics["skipped_steps"] = skipped
            metrics["overflow_steps"] = overflowed
            how = (f"{overflowed} from fp32 norm overflow (gradients themselves "
                   f"finite), {skipped - overflowed} from true NaN/inf")
            print(f"[train] fold{fold} ep{epoch}: skipped {skipped} optimizer "
                  f"step(s): {how}")
        log.append(metrics)
        monitor = monitor_value(metrics, cfg.train.monitor)
        if _is_better(monitor, best):
            best, best_epoch, patience = monitor, epoch, 0
            save(ckpt_dir / "best.pt", metrics)
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[train] early stop fold{fold} @ ep{epoch} (best ep{best_epoch})")
                break

    # Always leave a usable checkpoint. If the monitor was NaN every epoch
    # (single-class validation split) there is no "best", but downstream tools
    # still need weights to load rather than a FileNotFoundError.
    save(ckpt_dir / "last.pt", log[-1] if log else {})
    if best_epoch < 0:
        print(f"[train] fold{fold}: monitor '{cfg.train.monitor}' was NaN every epoch "
              f"(validation split likely single-class); using last.pt as best.pt")
        save(ckpt_dir / "best.pt", log[-1] if log else {})

    result = dict(fold=fold, best_inner=None if best_epoch < 0 else best, best_epoch=best_epoch)

    # The outer fold is scored exactly once, here, after the model is frozen.
    if testl is not None:
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, testl, cfg, device, full=True)
        test_metrics["epoch"] = best_epoch
        test_metrics["split"] = "outer_test"
        log.append(test_metrics)
        result["test"] = test_metrics
        # Kept per-patient so the folds can be pooled into one ranking later.
        result["oof"] = collect_predictions(model, testl, cfg, device, fold=fold)

    with open(ckpt_dir / "log.jsonl", "w") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")
    return result


def _monitor_label(monitor) -> str:
    if isinstance(monitor, str):
        return monitor
    return " + ".join(m if isinstance(m, str) else f"{m[1]}*{m[0]}" for m in monitor)


def _summarize(summary: list[dict], monitor) -> str:
    """Per-fold spread of the monitored quantity.

    Goes through monitor_value so a composite monitor (a list) does not get used
    as a dict key — `s["test"][monitor]` raises "unhashable type: list".
    """
    label = _monitor_label(monitor)
    vals = [monitor_value(s["test"], monitor) for s in summary if s.get("test")]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return f"[train] no fold produced a finite {label}"
    spread = f" ± {stdev(vals):.3f} (sd)" if len(vals) > 1 else ""
    return (f"[train] per-fold {label}: {mean(vals):.3f}{spread} "
            f"over {len(vals)}/{len(summary)} folds")


def _report_pooled(summary: list[dict], cfg) -> dict:
    """Pool every fold's out-of-fold predictions into one estimate and print it.

    This is the headline number, not the per-fold mean. Each patient contributes
    exactly one prediction, made by a model that never trained on them.
    """
    rows = [r for s in summary for r in (s.get("oof") or [])]
    if not rows:
        return {}
    pooled = pooled_metrics(rows, cfg.data.target_genes, cfg.eval.bootstrap_n)
    print(f"\n[train] POOLED OUT-OF-FOLD over {int(pooled['n_patients'])} patients:")
    for g in [x.lower() for x in cfg.data.target_genes]:
        auc = pooled.get(f"{g}_auc", float("nan"))
        n, npos = int(pooled.get(f"{g}_n", 0)), int(pooled.get(f"{g}_n_pos", 0))
        if math.isnan(auc):
            print(f"[train]   {g.upper():6s} n={n:4d} pos={npos:3d}  AUC undefined "
                  "(single-class)")
            continue
        lo, hi = pooled.get(f"{g}_auc_lo", float("nan")), pooled.get(f"{g}_auc_hi", float("nan"))
        warn = "  <-- CI includes chance" if lo <= 0.5 else ""
        print(f"[train]   {g.upper():6s} n={n:4d} pos={npos:3d}  "
              f"AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{warn}")
    return pooled


def _archive(run_id: str, cfg, summary: list, pooled: dict) -> str:
    """Write the one artifact that must outlive the checkpoints.

    Everything heavy — best.pt, last.pt — is gitignored and namespaced by run id,
    so it survives only as long as the disk. This file is small, tracked, and
    holds the resolved config plus every per-fold metric AND the out-of-fold
    predictions, which is enough to regenerate the pooled AUC, its CI and every
    ROC curve without the weights.

    Written because run 6 — the best result this project has produced — was
    destroyed by runs 7, 8 and 9 writing to the same `checkpoints/fold{k}/` and
    `runs/cv_summary.json` paths. Its headline AUC survived only in a log file
    under /private/tmp, which the OS is free to delete. A result you cannot
    reproduce a figure from is not a result.
    """
    out = Path(getattr(cfg.paths, "results_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    with open(path, "w") as fh:
        # Config subclasses dict and its nested values stay plain dicts, so it
        # serialises as-is — the archive records what actually ran, not what the
        # YAML on disk says today.
        json.dump({"run_id": run_id, "config": dict(cfg), "pooled": pooled,
                   "folds": summary}, fh, indent=2, default=str)
    print(f"[train] archived -> {path}")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None, help="override cfg.train.device")
    ap.add_argument("--run_id", default=None,
                    help="names checkpoints/<run_id>/ and results/<run_id>.json; "
                         "defaults to a timestamp. Runs no longer overwrite each other.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    # Join from a remembered root, not from the current value: mutating cfg in
    # place means a second main() on the same object would nest run-b inside
    # run-a. setdefault makes this idempotent however many times it runs.
    root = cfg["paths"].setdefault("ckpt_root", cfg.paths.ckpt_dir)
    cfg["paths"]["ckpt_dir"] = str(Path(root) / run_id)
    print(f"[train] run_id={run_id}  checkpoints -> {cfg.paths.ckpt_dir}")

    device = resolve_device(args.device or getattr(cfg.train, "device", "auto"))
    amp_on, _ = amp_settings(device, cfg.train.amp)
    print(f"[train] {describe(device, amp_on)}")
    if device.type == "cpu":
        print("[train] WARNING: no GPU backend. This is a scaffold run, not a real train.")

    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    nested = bool(getattr(cfg.split, "nested", True))
    if nested:
        folds = nested_kfold_indices(rows, cfg.split.stratify_on, cfg.split.n_folds,
                                     cfg.seed, float(getattr(cfg.split, "inner_val_frac", 0.2)))
    else:
        print("[train] WARNING: split.nested is false — the checkpoint is selected on "
              "the same split that gets reported, which biases the result upward.")
        folds = ((tr, va, None) for tr, va in
                 kfold_indices(rows, cfg.split.stratify_on, cfg.split.n_folds, cfg.seed))

    summary = []
    for fold, (tr, va, te) in enumerate(folds):
        summary.append(train_fold(cfg, fold, tr, va, te, device))

    Path(cfg.paths.runs_dir).mkdir(parents=True, exist_ok=True)
    print(_summarize(summary, cfg.train.monitor))
    pooled = _report_pooled(summary, cfg)
    with open(os.path.join(cfg.paths.runs_dir, "cv_summary.json"), "w") as fh:
        json.dump({"folds": summary, "pooled": pooled, "run_id": run_id}, fh, indent=2)
    _archive(run_id, cfg, summary, pooled)


if __name__ == "__main__":
    main()
