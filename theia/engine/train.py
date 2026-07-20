"""Training loop — k-fold aware, LoRA fine-tune, AMP, early stopping.

Run: python -m theia.engine.train --config configs/default.yaml
Trains one model per fold and writes checkpoints + a metrics log per fold.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from theia.config import load_config
from theia.data.dataset import RadiogenomicsDataset, collate, kfold_indices
from theia.engine.evaluate import evaluate
from theia.engine.losses import total_loss
from theia.models.theia_model import Theia


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(cfg, train_idx, val_idx):
    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    genes = cfg.data.target_genes
    tr = RadiogenomicsDataset(rows, genes, train_idx)
    va = RadiogenomicsDataset(rows, genes, val_idx)
    tl = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=4)
    return tl, vl


def train_fold(cfg, fold: int, train_idx, val_idx, device) -> dict:
    tl, vl = make_loaders(cfg, train_idx, val_idx)
    model = Theia(cfg).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    # scheduler steps once per OPTIMIZER step (every grad_accum batches), so
    # total_steps must match that count or OneCycleLR raises past the end.
    opt_steps_per_epoch = max(len(tl) // cfg.train.grad_accum, 1)
    steps = cfg.train.epochs * opt_steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=steps,
        pct_start=cfg.train.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)

    best, best_epoch, patience = -1.0, 0, 0
    ckpt_dir = Path(cfg.paths.ckpt_dir) / f"fold{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = []

    for epoch in range(cfg.train.epochs):
        model.train()
        opt.zero_grad()
        pbar = tqdm(tl, desc=f"fold{fold} ep{epoch}")
        for step, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                out = model(batch, device)
                loss, parts = total_loss(out, batch, cfg.train.loss_weights, device)
                loss = loss / cfg.train.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            pbar.set_postfix(parts)

        metrics = evaluate(model, vl, cfg, device)
        metrics["epoch"] = epoch
        log.append(metrics)
        monitor = metrics.get(cfg.train.monitor, -1.0)
        if monitor > best:
            best, best_epoch, patience = monitor, epoch, 0
            torch.save({"model": model.state_dict(), "cfg": dict(cfg), "metrics": metrics},
                       ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[train] early stop fold{fold} @ ep{epoch} (best ep{best_epoch})")
                break

    with open(ckpt_dir / "log.jsonl", "w") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")
    return dict(fold=fold, best=best, best_epoch=best_epoch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] WARNING: no CUDA. This is a scaffold run, not a real train.")

    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    summary = []
    for fold, (tr, va) in enumerate(kfold_indices(
            rows, cfg.split.stratify_on, cfg.split.n_folds, cfg.seed)):
        summary.append(train_fold(cfg, fold, tr, va, device))

    Path(cfg.paths.runs_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cfg.paths.runs_dir, "cv_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    aucs = [s["best"] for s in summary]
    print(f"[train] CV {cfg.train.monitor}: {sum(aucs)/len(aucs):.3f} "
          f"± {(max(aucs)-min(aucs))/2:.3f} over {len(aucs)} folds")


if __name__ == "__main__":
    main()
