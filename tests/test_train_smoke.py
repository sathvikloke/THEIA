"""End-to-end smoke test for the training loop.

Unit tests cover the pieces; this exercises the wiring — nested splits, epochs,
evaluation, checkpoint selection, early stopping, the frozen-model outer-test
pass, and the CV summary — on synthetic data.

The vision backbone and the language model are replaced with tiny random-init
stubs so the test needs no network, no pretrained weights, and no GPU. Everything
between them is the real code path.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def stub_model(monkeypatch):
    """Swap the two heavyweight downloads for stubs, keep the rest real."""
    import theia.models.backbone as bb
    import theia.models.heads as heads

    class TinyTrunk(nn.Module):
        """32x32 input -> 8x8 patch grid, so grounding has somewhere to point."""

        def __init__(self, dim=32, n=64):
            super().__init__()
            self.n, self.dim = n, dim
            self.stem = nn.Conv2d(3, dim, kernel_size=4, stride=4)

        def forward_features(self, x):
            f = self.stem(x)                              # [B, dim, h, w]
            return f.flatten(2).transpose(1, 2)[:, : self.n, :]

    def fake_build(self, name):
        return TinyTrunk(), 32, (8, 8), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    monkeypatch.setattr(bb.VisionEncoder, "_build", fake_build)

    class StubGeneration(nn.Module):
        def __init__(self, dim, lm_name, max_new_tokens=96):
            super().__init__()
            self.proj = nn.Linear(dim, 8)
            self.lm = nn.Linear(8, 8)     # something for LoRA-free state_dict parity

        def forward(self, region_emb, reports, device):
            return self.lm(self.proj(region_emb)).pow(2).mean()

        @torch.no_grad()
        def generate(self, region_emb, device):
            return ["a solid lesion."] * region_emb.shape[0]

    monkeypatch.setattr(heads, "GenerationHead", StubGeneration)
    import theia.models.theia_model as tm
    monkeypatch.setattr(tm, "GenerationHead", StubGeneration)


def _synthetic_cohort(tmp_path, n=40, slices=3, size=32):
    """Learnable signal: EGFR-positive patients get a brighter lesion."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        egfr = i % 2
        img = rng.random((slices, size, size)).astype(np.float32) * 0.2
        roi = np.zeros((slices, size, size), dtype=np.float32)
        roi[:, 10:22, 10:22] = 1.0
        img[:, 10:22, 10:22] += 0.5 + 0.4 * egfr
        npz = os.path.join(tmp_path, f"p{i}.npz")
        np.savez_compressed(npz, images=np.clip(img, 0, 1), roi=roi)
        rows.append(dict(patient_id=f"p{i}", npz=npz, report="a solid lesion.",
                         labels={"EGFR": egfr, "KRAS": (i // 2) % 2}, n_slices=slices))
    idx = os.path.join(tmp_path, "rows.jsonl")
    with open(idx, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return idx


def _cfg(tmp_path):
    from theia.config import load_config

    cfg = load_config("configs/default.yaml", validate=False)
    cfg["paths"]["processed_dir"] = str(tmp_path)
    cfg["paths"]["ckpt_dir"] = str(tmp_path / "ckpt")
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    cfg["data"]["image_size"] = 32
    cfg["data"]["n_slices"] = 3
    cfg["model"]["vision_dim"] = 32
    cfg["model"]["grounding_tokens"] = 4
    cfg["model"]["classifier_hidden"] = 16
    cfg["lora"]["enabled"] = False
    cfg["train"].update(epochs=2, batch_size=4, grad_accum=1, num_workers=0,
                        amp=False, device="cpu", early_stop_patience=5)
    cfg["split"].update(n_folds=3, nested=True, inner_val_frac=0.25)
    cfg["eval"]["bootstrap_n"] = 20
    return cfg


def test_train_fold_runs_and_writes_a_usable_checkpoint(tmp_path, stub_model):
    from theia.data.dataset import nested_kfold_indices
    from theia.engine.train import train_fold

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    result = train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    ckpt = tmp_path / "ckpt" / "fold0" / "best.pt"
    assert ckpt.exists(), "no checkpoint written"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model" in state and "cfg" in state

    assert "test" in result, "outer test fold was never scored"
    assert result["test"]["split"] == "outer_test"
    # Reported AUC must come from the held-out fold, not the selection split.
    log = [json.loads(l) for l in open(tmp_path / "ckpt" / "fold0" / "log.jsonl")]
    assert log[-1]["split"] == "outer_test"
    assert any(r["split"] == "inner_val" for r in log[:-1])


def test_checkpoint_reloads_into_a_fresh_model(tmp_path, stub_model):
    from theia.data.dataset import nested_kfold_indices
    from theia.engine.train import train_fold
    from theia.models.theia_model import Theia

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    state = torch.load(tmp_path / "ckpt" / "fold0" / "best.pt",
                       map_location="cpu", weights_only=False)
    fresh = Theia(cfg)
    missing, unexpected = fresh.load_state_dict(state["model"], strict=False)
    assert not missing and not unexpected, (
        f"state_dict does not round-trip: {len(missing)} missing, {len(unexpected)} unexpected"
    )


def test_grounding_beats_chance_on_a_signal_that_is_actually_there(tmp_path, stub_model):
    """The lesion is always in the same place, so grounding must exceed its
    shuffled baseline. If lift is ~0 on data this easy, grounding is not working."""
    from theia.data.dataset import nested_kfold_indices
    from theia.engine.train import train_fold

    rows = _synthetic_cohort(tmp_path, n=48)
    cfg = _cfg(tmp_path)
    cfg["train"]["epochs"] = 25
    cfg["train"]["lr"] = 1e-2
    cfg["train"]["early_stop_patience"] = 100
    cfg["train"]["loss_weights"]["ground"] = 5.0
    cfg["train"]["loss_weights"]["gen"] = 0.0
    # Select on grounding, since that is what this test is about. Monitoring
    # egfr_auc here would restore whichever epoch happened to win on AUC — often
    # epoch 0, before grounding has learned anything — and the grounding number
    # would report that checkpoint. Worth remembering for the real runs too:
    # every metric you report comes from the epoch your monitor picked.
    cfg["train"]["monitor"] = "grounding_mass_lift"
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    result = train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    lift = result["test"].get("grounding_mass_lift")
    assert lift is not None, "grounding lift was not reported"
    assert lift > 0.05, f"grounding did not beat its own chance baseline (lift={lift:.3f})"


def test_full_run_reports_held_out_metrics(tmp_path, stub_model, monkeypatch, capsys):
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(train_mod, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("sys.argv", ["train", "--device", "cpu"])
    train_mod.main()

    blob = json.load(open(tmp_path / "runs" / "cv_summary.json"))
    summary = blob["folds"]
    assert len(summary) == 3
    assert all("test" in s for s in summary)
    # the headline estimate is pooled across folds, not a mean of per-fold AUCs
    assert "pooled" in blob and blob["pooled"]["egfr_n"] > 0
    out = capsys.readouterr().out
    assert "POOLED OUT-OF-FOLD" in out
