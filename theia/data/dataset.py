"""Torch Dataset + patient-level splitting.

Each item is a dict:
  images: FloatTensor [n_slices, 1, H, W]   HU-windowed ROI stack
  roi:    FloatTensor [n_slices, 1, H, W]   ground-truth tumor mask (grounding target)
  report: str                               semantic-annotation pseudo-report
  <gene>: long                              {0,1} or -1 (unknown -> masked in loss),
                                            one key per configured gene, lowercased
  patient_id: str

Splitting is patient-level so no patient's slices leak across folds.

Two things here used to be silently wrong and are now enforced:

1. `collate` carried a hard-coded ("egfr", "kras") tuple, so every other gene in
   the panel was dropped from the batch. The loss then skipped those genes via
   `if gene not in batch: continue` and trained the corresponding heads on
   nothing, with no error. Gene keys are now driven by the configured panel.
2. `kfold_indices` filtered the cohort down to patients with a known label for
   the stratification gene, discarding patients whose *other* gene labels were
   perfectly usable. Those patients are now kept and spread across folds.
"""
from __future__ import annotations

import json
from typing import Iterator, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset

UNKNOWN = -1


class RadiogenomicsDataset(Dataset):
    def __init__(self, rows_jsonl: str, genes: Sequence[str], indices: list[int] | None = None):
        with open(rows_jsonl) as fh:
            self.rows = [json.loads(line) for line in fh]
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        self.genes = [g.lower() for g in genes]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        # np.load returns a lazy NpzFile holding an open descriptor; without the
        # context manager these leak one fd per item and exhaust the limit on
        # hosts with a low `ulimit -n`.
        with np.load(r["npz"]) as blob:
            images = torch.from_numpy(blob["images"]).float().unsqueeze(1)  # [S,1,H,W]
            roi = torch.from_numpy(blob["roi"]).float().unsqueeze(1)
        item = dict(
            images=images,
            roi=roi,
            report=r["report"],
            patient_id=r["patient_id"],
            # False for patients located from an AIM annotation with no pixel
            # mask. Their ROI is all zeros and the grounding term skips them.
            has_mask=torch.tensor(bool(r.get("has_mask", True))),
        )
        labels = {str(k).lower(): v for k, v in r["labels"].items()}
        for g in self.genes:
            item[g] = torch.tensor(int(labels.get(g, UNKNOWN)), dtype=torch.long)
        return item


def collate(batch: list[dict], genes: Sequence[str] | None = None) -> dict:
    """Pad slice counts to the batch max so images stack cleanly.

    `genes` should be the configured panel. When omitted the gene keys are
    inferred from the batch (any scalar long tensor), so no gene can be dropped
    just because it was not in a hard-coded list.
    """
    max_s = max(b["images"].shape[0] for b in batch)

    def pad(x):
        if x.shape[0] == max_s:
            return x
        pad_shape = (max_s - x.shape[0], *x.shape[1:])
        return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)

    if genes is None:
        gene_keys = [
            k for k, v in batch[0].items()
            if torch.is_tensor(v) and v.ndim == 0 and v.dtype == torch.long
            and k != "has_mask"
        ]
    else:
        gene_keys = [g.lower() for g in genes]

    n_slices = torch.tensor([b["images"].shape[0] for b in batch])
    # Explicit validity mask so the encoder can ignore padded slices instead of
    # averaging real features against zeros.
    slice_mask = torch.zeros(len(batch), max_s)
    for i, n in enumerate(n_slices.tolist()):
        slice_mask[i, :n] = 1.0

    out = dict(
        images=torch.stack([pad(b["images"]) for b in batch]),
        roi=torch.stack([pad(b["roi"]) for b in batch]),
        report=[b["report"] for b in batch],
        patient_id=[b["patient_id"] for b in batch],
        n_slices=n_slices,
        slice_mask=slice_mask,
        has_mask=torch.stack([b.get("has_mask", torch.tensor(True)) for b in batch]),
    )
    for g in gene_keys:
        if g in batch[0]:
            out[g] = torch.stack([b[g] for b in batch])
    return out


def make_collate(genes: Sequence[str]):
    """Bind the gene panel for use as a DataLoader collate_fn."""
    def _collate(batch: list[dict]) -> dict:
        return collate(batch, genes=genes)
    return _collate


def _load_labels(rows_jsonl: str, stratify_on: str) -> np.ndarray:
    with open(rows_jsonl) as fh:
        rows = [json.loads(line) for line in fh]
    key = stratify_on.lower()
    out = []
    for r in rows:
        labels = {str(k).lower(): v for k, v in r["labels"].items()}
        out.append(int(labels.get(key, UNKNOWN)))
    return np.array(out)


def kfold_indices(rows_jsonl: str, stratify_on: str, n_folds: int,
                  seed: int) -> Iterator[tuple[list[int], list[int]]]:
    """Yield (train_idx, val_idx) per fold.

    Patients with a known `stratify_on` label are stratified. Patients whose
    label for that gene is unknown are still useful supervision for every other
    gene, so they are distributed across folds by a plain KFold rather than
    dropped. `evaluate` masks unknown labels out of the metrics anyway.
    """
    y = _load_labels(rows_jsonl, stratify_on)
    known = np.where(y >= 0)[0]
    unknown = np.where(y < 0)[0]
    if known.size == 0:
        raise ValueError(f"no patient has a known '{stratify_on}' label to stratify on")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    known_folds = list(skf.split(known, y[known]))

    if unknown.size >= n_folds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        unk_folds = list(kf.split(unknown))
    else:
        # Too few to split; keep them in train for every fold so they still
        # supervise the other genes without polluting a tiny validation set.
        all_unk = list(range(unknown.size))
        unk_folds = [(all_unk, []) for _ in range(n_folds)]

    for (k_tr, k_va), (u_tr, u_va) in zip(known_folds, unk_folds):
        train = known[k_tr].tolist() + unknown[list(u_tr)].tolist()
        val = known[k_va].tolist() + unknown[list(u_va)].tolist()
        yield sorted(train), sorted(val)


def nested_kfold_indices(rows_jsonl: str, stratify_on: str, n_folds: int, seed: int,
                         inner_val_frac: float = 0.2
                         ) -> Iterator[tuple[list[int], list[int], list[int]]]:
    """Yield (train_idx, val_idx, test_idx) per outer fold.

    The outer fold is the held-out test set and is touched exactly once, after
    the model is frozen. Early stopping and checkpoint selection run on `val`,
    which is carved out of the outer training portion. The flat `kfold_indices`
    selects the checkpoint on the same split it reports, which is selection on
    the test set and biases every reported number upward.
    """
    y = _load_labels(rows_jsonl, stratify_on)
    rng = np.random.default_rng(seed)

    for outer_train, test in kfold_indices(rows_jsonl, stratify_on, n_folds, seed):
        outer_train = np.array(outer_train)
        y_tr = y[outer_train]
        known = outer_train[y_tr >= 0]
        unknown = outer_train[y_tr < 0]

        val: list[int] = []
        # Stratify the inner validation split so both classes are represented;
        # a single-class val split makes the monitored AUC undefined.
        for cls in (0, 1):
            pool = known[y[known] == cls]
            if pool.size == 0:
                continue
            n_val = max(1, int(round(pool.size * inner_val_frac)))
            n_val = min(n_val, max(pool.size - 1, 0))  # never empty the train side
            if n_val:
                val.extend(rng.choice(pool, size=n_val, replace=False).tolist())
        if unknown.size:
            n_val = int(round(unknown.size * inner_val_frac))
            if n_val:
                val.extend(rng.choice(unknown, size=n_val, replace=False).tolist())

        val_set = set(val)
        train = [int(i) for i in outer_train if int(i) not in val_set]
        yield sorted(train), sorted(int(i) for i in val_set), sorted(test)
