"""Grounding pretraining on cohorts that have masks but no mutation labels.

Motivation, measured: on the 158-patient labelled cohort, grounding works in 4
of 5 folds (mean mass lift +0.235) but fold 3 sits at -0.001. There are only 119
mask-supervised patients to learn localisation from, and the grounding head is
learning it from scratch inside a run whose loss is dominated by classification.

NSCLC-Radiomics contributes 421 more masked patients and LIDC-IDRI another 875 —
1,297 in total, none of which need a genomic label. Pretraining the vision
blocks and the grounding head on that, then loading the weights into the main
run, gives grounding a head start it currently has to earn from 119 cases.

This trains ONLY the vision encoder's unfrozen blocks and the grounding head, on
the grounding loss alone. No classifier, no language model — building BioGPT
here would cost 480M parameters to compute a loss that never touches it.

Run:
  python -m theia.engine.pretrain --rows data/processed_pretrain/rows.jsonl \
      --out checkpoints/grounding_pretrain.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from theia.config import load_config
from theia.data.dataset import RadiogenomicsDataset, make_collate
from theia.engine.evaluate import grounding_metrics
from theia.engine.losses import grounding_loss
from theia.models.backbone import VisionEncoder
from theia.models.heads import GroundingHead
from theia.runtime import amp_settings, autocast, describe, make_grad_scaler, resolve_device


class GroundingOnly(torch.nn.Module):
    """Vision encoder + grounding head. Deliberately no classifier or LM."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.vision = VisionEncoder(m.vision_encoder, m.vision_dim, m.freeze_vision)
        n = int(getattr(m, "unfreeze_last_n", 0) or 0)
        if m.freeze_vision and n:
            k = self.vision.unfreeze_last_blocks(n)
            print(f"[pretrain] unfroze last {n} vision block(s): {k / 1e6:.1f}M params")
        self.grounding = GroundingHead(m.vision_dim, m.grounding_tokens)

    def forward(self, batch, device):
        images = batch["images"].to(device)
        sm = batch.get("slice_mask")
        tokens, grid = self.vision.encode(images, sm.to(device) if sm is not None else None)
        return self.grounding(tokens, grid)


def _split(rows_path: str, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    with open(rows_path) as fh:
        n = sum(1 for _ in fh)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(int(round(n * val_frac)), 1)
    return sorted(idx[k:].tolist()), sorted(idx[:k].tolist())


@torch.no_grad()
def _eval(model, loader, device) -> dict:
    model.eval()
    agg: dict[str, list[float]] = {}
    for batch in loader:
        out = model(batch, device)
        for k, v in grounding_metrics(out["attn_maps"], batch["roi"]).items():
            agg.setdefault(k, []).extend(v)
    m = {k: float(np.mean(v)) for k, v in agg.items() if v}
    for base in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        if base in m and f"{base}_shuffled" in m:
            m[f"{base}_lift"] = m[base] - m[f"{base}_shuffled"]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--rows", default="data/processed_pretrain/rows.jsonl")
    ap.add_argument("--out", default="checkpoints/grounding_pretrain.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--val_frac", type=float, default=0.15)
    a = ap.parse_args()

    cfg = load_config(a.config)
    epochs = a.epochs or int(cfg.grounding_pretrain.epochs)
    lr = float(cfg.grounding_pretrain.lr)
    device = resolve_device(getattr(cfg.train, "device", "auto"))
    amp_on, amp_dtype = amp_settings(device, cfg.train.amp)
    print(f"[pretrain] {describe(device, amp_on)} | epochs={epochs} lr={lr}")

    genes = cfg.data.target_genes
    tr_idx, va_idx = _split(a.rows, a.val_frac, cfg.seed)
    workers = int(getattr(cfg.train, "num_workers", 2))
    mk = lambda idx, sh: DataLoader(  # noqa: E731
        RadiogenomicsDataset(a.rows, genes, idx), batch_size=cfg.train.batch_size,
        shuffle=sh, collate_fn=make_collate(genes), num_workers=workers,
        persistent_workers=workers > 0)
    tl, vl = mk(tr_idx, True), mk(va_idx, False)
    print(f"[pretrain] {len(tr_idx)} train / {len(va_idx)} val patients")

    model = GroundingOnly(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[pretrain] trainable: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(epochs * len(tl), 1), pct_start=0.1)
    scaler = make_grad_scaler(device, amp_on)

    best, log = -float("inf"), []
    Path(os.path.dirname(a.out) or ".").mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in tqdm(tl, desc=f"pretrain ep{ep}"):
            with autocast(device, amp_on, amp_dtype):
                out = model(batch, device)
                loss = grounding_loss(out["attn_maps"], batch["roi"], device)
            opt.zero_grad()
            scaler.scale(loss).backward()
            prev = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= prev:
                sched.step()
            run_loss += float(loss.detach())
            seen += 1

        m = _eval(model, vl, device)
        lift = m.get("grounding_mass_lift", float("nan"))
        m.update(epoch=ep, train_loss=run_loss / max(seen, 1))
        log.append(m)
        print(f"[pretrain] ep{ep}: loss {run_loss / max(seen, 1):.4f}  "
              f"val mass {m.get('grounding_mass', float('nan')):.3f} "
              f"(chance {m.get('grounding_mass_shuffled', float('nan')):.3f}, "
              f"lift {lift:+.3f})  pointing {m.get('grounding_pointing', float('nan')):.2f}",
              flush=True)

        if lift == lift and lift > best:
            best = lift
            torch.save({"vision": model.vision.state_dict(),
                        "grounding": model.grounding.state_dict(),
                        "metrics": m, "cfg": dict(cfg)}, a.out)
            print(f"[pretrain]   saved (best lift {best:+.3f}) -> {a.out}", flush=True)

    with open(str(a.out) + ".log.jsonl", "w") as fh:
        for r in log:
            fh.write(json.dumps(r) + "\n")
    print(f"[pretrain] done. best val grounding lift {best:+.3f}")


if __name__ == "__main__":
    main()
