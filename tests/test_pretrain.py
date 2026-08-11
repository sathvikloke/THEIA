"""Tests for label-free grounding pretraining.

This is the path LIDC-IDRI would run through, so it is worth working before
136 GB arrives rather than after. The failures it can have are quiet ones: a
split that leaks, a model that silently carries a classifier it must not have,
or a checkpoint whose keys do not match what loads it.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def stub_vision(monkeypatch):
    import theia.models.backbone as bb

    class TinyTrunk(nn.Module):
        def __init__(self, dim=32, n=64):
            super().__init__()
            self.n, self.dim = n, dim
            self.stem = nn.Conv2d(3, dim, kernel_size=4, stride=4)

        def forward_features(self, x):
            f = self.stem(x)
            return f.flatten(2).transpose(1, 2)[:, : self.n, :]

    monkeypatch.setattr(bb.VisionEncoder, "_build",
                        lambda self, name: (TinyTrunk(), 32, (8, 8),
                                            (0.5,) * 3, (0.5,) * 3))


def _cfg(tmp_path):
    from theia.config import load_config

    cfg = load_config("configs/default.yaml", validate=False)
    cfg["data"]["image_size"] = 32
    cfg["data"]["n_slices"] = 3
    cfg["model"]["vision_dim"] = 32
    cfg["model"]["grounding_tokens"] = 4
    cfg["model"]["grounding_pretrain_ckpt"] = None
    return cfg


def _rows(tmp_path, n=20):
    rows = []
    for i in range(n):
        npz = os.path.join(tmp_path, f"m{i}.npz")
        img = np.random.rand(3, 32, 32).astype(np.float32)
        roi = np.zeros((3, 32, 32), dtype=np.float32)
        roi[:, 8:20, 8:20] = 1.0
        np.savez_compressed(npz, images=img, roi=roi)
        # Label-free: an EMPTY labels dict, which is the whole point.
        rows.append(dict(patient_id=f"m{i}", npz=npz, report="", labels={},
                         n_slices=3, has_mask=True))
    p = os.path.join(tmp_path, "rows.jsonl")
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_pretrain_model_has_no_classifier_or_language_model(tmp_path, stub_vision):
    """It must carry vision + grounding ONLY.

    A classifier here would be trained on absent labels, and an LM would pull in
    a download this path does not need. Both would also break the strict
    state_dict transfer that theia_model.load_grounding_pretrain performs.
    """
    from theia.engine.pretrain import GroundingOnly

    m = GroundingOnly(_cfg(tmp_path))
    names = {n.split(".")[0] for n, _ in m.named_parameters()}
    assert names <= {"vision", "grounding"}, f"unexpected submodules: {names}"
    assert not hasattr(m, "classifier")
    assert not hasattr(m, "generation")


def test_split_is_disjoint_and_covers_everything(tmp_path):
    """Train and validation must partition the cohort exactly.

    An overlap would report a validation lift on patients the model trained on,
    which is precisely the number this whole path exists to produce.
    """
    from theia.engine.pretrain import _split

    p = _rows(tmp_path, n=37)
    tr, va = _split(p, val_frac=0.2, seed=1337)
    assert not set(tr) & set(va), "train and validation overlap"
    assert sorted(tr + va) == list(range(37)), "split does not cover the cohort"
    assert len(va) == round(37 * 0.2)


def test_split_is_deterministic_for_a_seed(tmp_path):
    from theia.engine.pretrain import _split

    p = _rows(tmp_path, n=25)
    assert _split(p, 0.2, 7) == _split(p, 0.2, 7)
    assert _split(p, 0.2, 7) != _split(p, 0.2, 8)


def test_validation_split_is_never_empty(tmp_path):
    """A tiny cohort must still yield a validation patient, not zero.

    With an empty split the reported lift would be NaN and checkpoint selection
    would have nothing to select on.
    """
    from theia.engine.pretrain import _split

    tr, va = _split(_rows(tmp_path, n=3), val_frac=0.01, seed=1)
    assert len(va) >= 1 and len(tr) >= 1


def test_eval_reports_lift_against_a_shuffled_baseline(tmp_path, stub_vision):
    """Every grounding number must arrive with its own chance level.

    A raw mass of 0.74 means nothing if shuffling scores 0.73; the lift is the
    result, and pretraining was reported at +0.735 on exactly this path.
    """
    from theia.engine.pretrain import GroundingOnly, _eval

    m = GroundingOnly(_cfg(tmp_path))
    roi = torch.zeros(2, 3, 1, 32, 32)
    roi[:, :, :, 8:20, 8:20] = 1.0
    loader = [{"images": torch.rand(2, 3, 1, 32, 32), "roi": roi}]
    out = _eval(m, loader, "cpu")
    for base in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        assert f"{base}_shuffled" in out, f"{base} reported with no chance baseline"
        assert f"{base}_lift" in out, f"{base} lift not computed"


def test_pretrain_weights_load_into_the_full_model(tmp_path, stub_vision, monkeypatch):
    """The checkpoint must transfer into Theia without key surgery.

    load_grounding_pretrain loads strictly, on purpose: a silent key mismatch
    looks exactly like pretraining that did not help, which is the one
    conclusion this experiment must not reach by accident.
    """
    import theia.models.heads as heads
    import theia.models.theia_model as tm
    from theia.engine.pretrain import GroundingOnly

    class StubGen(nn.Module):
        def __init__(self, dim, lm_name, max_new_tokens=96):
            super().__init__()
            self.lm = nn.Linear(4, 4)

    monkeypatch.setattr(heads, "GenerationHead", StubGen)
    monkeypatch.setattr(tm, "GenerationHead", StubGen)

    cfg = _cfg(tmp_path)
    cfg["lora"]["enabled"] = False
    pre = GroundingOnly(cfg)
    ckpt = tmp_path / "pre.pt"
    torch.save({"vision": pre.vision.state_dict(),
                "grounding": pre.grounding.state_dict(),
                "metrics": {"grounding_mass_lift": 0.735}}, ckpt)

    cfg["model"]["grounding_pretrain_ckpt"] = str(ckpt)
    cfg["model"]["warm_start_vision"] = True
    full = tm.Theia(cfg)          # raises if the keys do not match
    for a, b in zip(full.grounding.state_dict().values(),
                    pre.grounding.state_dict().values()):
        assert torch.allclose(a, b), "grounding weights did not transfer"
