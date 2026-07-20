"""Smoke tests that run without TCIA data, a GPU, or model weights.

They fabricate a tiny processed dataset and exercise the data + loss plumbing so
the pipeline's wiring is verifiable before any real training. Run: pytest -q
"""
import json
import os

import numpy as np
import torch


def _fake_dataset(tmp_path, n=6, slices=4, size=32):
    rows = []
    for i in range(n):
        npz = os.path.join(tmp_path, f"p{i}.npz")
        np.savez_compressed(
            npz,
            images=np.random.rand(slices, size, size).astype(np.float32),
            roi=(np.random.rand(slices, size, size) > 0.7).astype(np.float32),
        )
        rows.append(dict(patient_id=f"p{i}", npz=npz, report="a solid lesion.",
                         labels={"EGFR": i % 2, "KRAS": (i + 1) % 2}, n_slices=slices))
    idx = os.path.join(tmp_path, "rows.jsonl")
    with open(idx, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return idx


def test_dataset_and_collate(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset, collate

    idx = _fake_dataset(tmp_path)
    ds = RadiogenomicsDataset(idx, ["EGFR", "KRAS"])
    assert len(ds) == 6
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["images"].dim() == 5           # [B,S,1,H,W]
    assert batch["egfr"].shape[0] == 3
    assert len(batch["report"]) == 3


def test_kfold_is_patient_disjoint(tmp_path):
    from theia.data.dataset import kfold_indices

    idx = _fake_dataset(tmp_path)
    for tr, va in kfold_indices(idx, "EGFR", n_folds=3, seed=0):
        assert set(tr).isdisjoint(set(va))


def test_grounding_loss_shapes():
    from theia.engine.losses import grounding_loss

    attn = torch.rand(2, 8, 14, 14)
    roi = (torch.rand(2, 4, 1, 224, 224) > 0.8).float()
    loss = grounding_loss(attn, roi, "cpu")
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_classification_loss_masks_unknowns():
    from theia.engine.losses import classification_loss

    logits = {"egfr": torch.randn(3, 2)}
    batch = {"egfr": torch.tensor([1, -1, 0])}
    loss = classification_loss(logits, batch, "cpu")
    assert torch.isfinite(loss)
