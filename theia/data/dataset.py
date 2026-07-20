"""Torch Dataset + patient-level stratified k-fold splitting.

Each item is a dict:
  images: FloatTensor [n_slices, 1, H, W]   HU-windowed ROI stack
  roi:    FloatTensor [n_slices, 1, H, W]   ground-truth tumor mask (grounding target)
  report: str                               semantic-annotation pseudo-report
  egfr, kras: long                          {0,1} or -1 (unknown -> masked in loss)
  patient_id: str

Splitting is patient-level so no patient's slices leak across folds.
"""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


class RadiogenomicsDataset(Dataset):
    def __init__(self, rows_jsonl: str, genes: list[str], indices: list[int] | None = None):
        with open(rows_jsonl) as fh:
            self.rows = [json.loads(line) for line in fh]
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        self.genes = genes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        blob = np.load(r["npz"])
        images = torch.from_numpy(blob["images"]).float().unsqueeze(1)  # [S,1,H,W]
        roi = torch.from_numpy(blob["roi"]).float().unsqueeze(1)
        item = dict(
            images=images,
            roi=roi,
            report=r["report"],
            patient_id=r["patient_id"],
        )
        for g in self.genes:
            item[g.lower()] = torch.tensor(int(r["labels"].get(g, -1)), dtype=torch.long)
        return item


def collate(batch: list[dict]) -> dict:
    """Pad slice counts to the batch max so images stack cleanly."""
    max_s = max(b["images"].shape[0] for b in batch)

    def pad(x):
        if x.shape[0] == max_s:
            return x
        pad_shape = (max_s - x.shape[0], *x.shape[1:])
        return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)

    out = dict(
        images=torch.stack([pad(b["images"]) for b in batch]),
        roi=torch.stack([pad(b["roi"]) for b in batch]),
        report=[b["report"] for b in batch],
        patient_id=[b["patient_id"] for b in batch],
        n_slices=torch.tensor([b["images"].shape[0] for b in batch]),
    )
    for key in batch[0]:
        if key in ("egfr", "kras"):
            out[key] = torch.stack([b[key] for b in batch])
    return out


def kfold_indices(rows_jsonl: str, stratify_on: str, n_folds: int, seed: int) -> Iterator[tuple]:
    """Yield (train_idx, val_idx) per fold, stratified on a gene's label."""
    with open(rows_jsonl) as fh:
        rows = [json.loads(line) for line in fh]
    y = np.array([int(r["labels"].get(stratify_on, -1)) for r in rows])
    known = np.where(y >= 0)[0]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_k, val_k in skf.split(known, y[known]):
        yield known[train_k].tolist(), known[val_k].tolist()
