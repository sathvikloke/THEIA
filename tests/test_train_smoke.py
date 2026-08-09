"""End-to-end smoke test for the training loop.

Unit tests cover the pieces; this exercises the wiring — nested splits, epochs,
evaluation, checkpoint selection, early stopping, the frozen-model outer-test
pass, and the CV summary — on synthetic data.

The vision backbone and the language model are replaced with tiny random-init
stubs so the test needs no network, no pretrained weights, and no GPU. Everything
between them is the real code path.
"""
import json
import math
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
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg["data"]["image_size"] = 32
    cfg["data"]["n_slices"] = 3
    cfg["model"]["vision_dim"] = 32
    cfg["model"]["grounding_tokens"] = 4
    cfg["model"]["classifier_hidden"] = 16
    cfg["lora"]["enabled"] = False
    # tiny stub dims; the real pretrain checkpoint would not match
    cfg["model"]["grounding_pretrain_ckpt"] = None
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


def test_a_nonfinite_gradient_does_not_kill_the_fold(tmp_path, stub_model, monkeypatch):
    """One poisoned batch must be skipped, not written into the weights.

    Off CUDA, AMP is disabled and GradScaler is a no-op, so before grad_clip
    landed there was nothing skipping non-finite steps: a single bad batch put
    NaN into every parameter and the fold trained on garbage for the rest of its
    epochs. Run 10's fold 0 did exactly this at epoch 8 and then contributed its
    epoch-0 checkpoint to the pooled estimate.
    """
    from theia.data.dataset import nested_kfold_indices
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    cfg["train"]["epochs"] = 2

    # Poison exactly one step's loss, on the step that triggers an optimizer update.
    real_total_loss, calls = train_mod.total_loss, {"n": 0}

    def poisoned(out, batch, weights, device):
        loss, parts = real_total_loss(out, batch, weights, device)
        calls["n"] += 1
        if calls["n"] == 1:
            return loss * float("inf"), parts
        return loss, parts

    monkeypatch.setattr(train_mod, "total_loss", poisoned)

    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    result = train_mod.train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    state = torch.load(tmp_path / "ckpt" / "fold0" / "best.pt",
                       map_location="cpu", weights_only=False)
    bad = [k for k, v in state["model"].items()
           if v.is_floating_point() and not torch.isfinite(v).all()]
    assert not bad, f"non-finite weights survived into the checkpoint: {bad[:3]}"
    auc = result["test"].get("egfr_auc")
    assert auc is None or not math.isnan(auc), "fold produced NaN metrics after one bad batch"


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


def test_each_run_is_archived_and_never_overwrites_the_last(tmp_path, stub_model,
                                                            monkeypatch):
    """Two runs must leave two results files, and each must be enough to rebuild
    the pooled AUC without any checkpoint.

    Run 6 — the best result this project produced — was destroyed by later runs
    writing to the same `checkpoints/fold{k}/best.pt` and `runs/cv_summary.json`.
    Its headline AUC survived only in a log under /private/tmp. This is the test
    that keeps that from happening twice.
    """
    from theia.engine import train as train_mod
    from theia.engine.evaluate import pooled_metrics

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(train_mod, "load_config", lambda *a, **k: cfg)

    for run_id in ("run-a", "run-b"):
        monkeypatch.setattr("sys.argv", ["train", "--device", "cpu", "--run_id", run_id])
        train_mod.main()

    results = sorted((tmp_path / "results").glob("*.json"))
    assert [p.stem for p in results] == ["run-a", "run-b"], (
        f"runs overwrote each other: {[p.name for p in results]}")
    # Weights are namespaced too, so run-a's model is still on disk.
    assert (tmp_path / "ckpt" / "run-a" / "fold0" / "best.pt").exists()
    assert (tmp_path / "ckpt" / "run-b" / "fold0" / "best.pt").exists()

    blob = json.load(open(results[0]))
    assert blob["config"]["train"]["epochs"] == cfg["train"]["epochs"], \
        "archive does not record what actually ran"
    # The archive alone reproduces the headline number — no checkpoint loaded.
    oof = [r for f in blob["folds"] for r in (f.get("oof") or [])]
    assert oof, "no out-of-fold predictions archived; ROC curves are unrecoverable"
    rebuilt = pooled_metrics(oof, cfg["data"]["target_genes"], cfg["eval"]["bootstrap_n"])
    assert rebuilt["egfr_auc"] == pytest.approx(blob["pooled"]["egfr_auc"])


def test_patience_does_not_run_during_warmup(tmp_path, stub_model, monkeypatch, capsys):
    """A lucky epoch-0 monitor must not stop a fold that is still improving.

    Under OneCycleLR the first epochs sit far below peak LR, so epoch 0 is an
    essentially untrained model and its inner-validation AUC is a draw from
    noise on ~24 patients. Run 13 lost 2 of 5 folds to exactly this: fold 2
    peaked at ep0 (0.620), then climbed 0.436 -> 0.571 over epochs 4-8 with
    grounding lift rising to +0.272 and still going, and was stopped because
    patience had been counting the whole time.
    """
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    cfg["train"].update(epochs=12, early_stop_patience=2, early_stop_min_epochs=8)

    # A monitor that peaks immediately and then never recovers: with patience 2
    # and no warmup floor this stops at epoch 2.
    seq = iter([0.9] + [0.1] * 20)
    monkeypatch.setattr(train_mod, "monitor_value", lambda m, mon: next(seq, 0.1))

    from theia.data.dataset import nested_kfold_indices
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    train_mod.train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    log = [json.loads(l) for l in open(tmp_path / "ckpt" / "fold0" / "log.jsonl")]
    ran = len([r for r in log if r.get("split") == "inner_val"])
    assert ran >= 8, (
        f"fold stopped after {ran} epochs; patience ran during warmup")


def test_a_frozen_monitor_is_reported_as_stuck(tmp_path, stub_model, monkeypatch, capsys):
    """An AUC repeating to full precision is a stuck fold, not a plateau.

    Run 13's fold 0 sat at exactly 0.556 with grounding lift exactly 0.000 for
    all nine epochs and was reported as an ordinary early stop, indistinguishable
    in the output from a fold that trained and levelled off.
    """
    from theia.engine import train as train_mod
    from theia.data.dataset import nested_kfold_indices

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    cfg["train"].update(epochs=6, early_stop_patience=99, early_stop_min_epochs=0)
    monkeypatch.setattr(train_mod, "monitor_value", lambda m, mon: 0.5560000)

    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    train_mod.train_fold(cfg, 0, tr, va, te, torch.device("cpu"))
    out = capsys.readouterr().out
    assert "not changing" in out and "stuck fold" in out, (
        f"a frozen monitor was not reported: {out[-400:]}")


def test_a_stalled_fold_is_retried_from_a_fresh_init(tmp_path, stub_model, monkeypatch):
    """A fold that skips every optimizer step must be re-run, not reported.

    There is an unresolved MPS backward instability: on a bad initialisation a
    fold skips every step from epoch 1 onward, so its weights never change and it
    contributes an untrained model to the pooled estimate. Runs 11, 13 and 14
    each lost a fold this way.
    """
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    calls = {"n": 0}
    real = train_mod.train_fold

    def stall_once(c, fold, tr, va, te, dev):
        calls["n"] += 1
        if calls["n"] == 1:
            raise train_mod._FoldStalled("fold0 skipped all 6 optimizer steps at epoch 1")
        return real(c, fold, tr, va, te, dev)

    monkeypatch.setattr(train_mod, "train_fold", stall_once)

    from theia.data.dataset import nested_kfold_indices
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    result = train_mod.train_fold_with_retry(cfg, 0, tr, va, te, torch.device("cpu"))

    assert calls["n"] == 2, "a stalled fold was not retried"
    assert not result.get("stalled"), "retry succeeded but the fold is still marked stalled"
    assert result.get("init_seed_offset") == 1000, (
        "the retry's seed offset must be recorded, or the archive misreports what ran")


def test_retry_never_fires_on_a_merely_bad_result(tmp_path, stub_model, monkeypatch):
    """Retrying on a poor AUC would be seed-shopping and would bias every number.

    The criterion must be numerical failure alone. A fold that trains normally and
    scores badly has to be kept exactly as it is.
    """
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    calls = {"n": 0}

    def terrible_but_healthy(c, fold, tr, va, te, dev):
        calls["n"] += 1
        return {"fold": fold, "test": {"egfr_auc": 0.11}}

    monkeypatch.setattr(train_mod, "train_fold", terrible_but_healthy)

    from theia.data.dataset import nested_kfold_indices
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    result = train_mod.train_fold_with_retry(cfg, 0, tr, va, te, torch.device("cpu"))

    assert calls["n"] == 1, "a healthy fold was retried because its score was low"
    assert result["test"]["egfr_auc"] == 0.11
    assert "init_seed_offset" not in result


def test_cli_overrides_are_applied_and_archived(tmp_path, stub_model, monkeypatch):
    """--set must reach the config AND be visible in the archive afterwards.

    An override that changes what ran but not what is recorded is how a result
    gets misattributed to settings it was never produced under. The archive
    stores the RESOLVED config for exactly this reason.
    """
    from theia.config import load_config
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    base = _cfg(tmp_path)

    def fake_load(path, overrides=None, **kw):
        cfg = base
        if overrides:
            for dotted, value in overrides.items():
                node, *rest = dotted.split(".")
                target = cfg[node]
                for k in rest[:-1]:
                    target = target[k]
                target[rest[-1]] = value
        return cfg

    monkeypatch.setattr(train_mod, "load_config", fake_load)
    monkeypatch.setattr("sys.argv", ["train", "--device", "cpu", "--run_id", "ovr",
                                     "--set", "train.loss_weights.gen=0.0",
                                     "--set", "train.epochs=1"])
    train_mod.main()

    blob = json.load(open(tmp_path / "results" / "ovr.json"))
    assert blob["config"]["train"]["loss_weights"]["gen"] == 0.0, (
        "override did not reach the archived config")
    assert blob["config"]["train"]["epochs"] == 1
    # YAML parsing, not raw strings — 0.0 must be a float, not "0.0".
    assert isinstance(blob["config"]["train"]["loss_weights"]["gen"], float)
    del load_config


def test_last_checkpoint_is_not_written_unless_asked(tmp_path, stub_model):
    """last.pt doubles storage and nothing loads it.

    Every tool opens best.pt. 33 accumulated last.pt files had reached 59 GB,
    which was most of the reason there was no disk left for a second pretraining
    cohort.
    """
    from theia.data.dataset import nested_kfold_indices
    from theia.engine.train import train_fold

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    d = tmp_path / "ckpt" / "fold0"
    assert (d / "best.pt").exists(), "no selected checkpoint written"
    assert not (d / "last.pt").exists(), "last.pt written without save_last"

    cfg["train"]["save_last"] = True
    train_fold(cfg, 1, tr, va, te, torch.device("cpu"))
    assert (tmp_path / "ckpt" / "fold1" / "last.pt").exists(), "save_last ignored"


def test_last_is_still_written_when_no_epoch_could_be_selected(tmp_path, stub_model,
                                                               monkeypatch):
    """The fallback path must survive the storage change.

    If the monitor is NaN every epoch there is no 'best'; something has to be
    loadable or downstream tools hit FileNotFoundError.
    """
    from theia.data.dataset import nested_kfold_indices
    from theia.engine import train as train_mod

    rows = _synthetic_cohort(tmp_path)
    cfg = _cfg(tmp_path)
    cfg["train"]["save_last"] = False
    monkeypatch.setattr(train_mod, "monitor_value", lambda m, mon: float("nan"))

    tr, va, te = next(nested_kfold_indices(rows, "EGFR", 3, 1337, 0.25))
    train_mod.train_fold(cfg, 0, tr, va, te, torch.device("cpu"))

    d = tmp_path / "ckpt" / "fold0"
    assert (d / "best.pt").exists(), "no fallback checkpoint written"
    assert (d / "last.pt").exists(), "fallback last.pt was suppressed"
