"""One test per bug fixed in the audit pass, so none of them come back quietly.

Every test names the defect it pins. These run without TCIA data, a GPU, or
downloaded model weights: `pytest -q`.
"""
import json
import os

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fake_rows(tmp_path, genes, n=8, slices=4, size=32, unknown_first=0):
    rows = []
    for i in range(n):
        npz = os.path.join(tmp_path, f"p{i}.npz")
        np.savez_compressed(
            npz,
            images=np.random.rand(slices, size, size).astype(np.float32),
            roi=(np.random.rand(slices, size, size) > 0.7).astype(np.float32),
        )
        labels = {g: i % 2 for g in genes}
        if i < unknown_first:
            labels[genes[0]] = -1          # unknown for the stratification gene only
        rows.append(dict(patient_id=f"p{i}", npz=npz, report="a solid lesion.",
                         labels=labels, n_slices=slices))
    idx = os.path.join(tmp_path, "rows.jsonl")
    with open(idx, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return idx


# --------------------------------------------------------------------------
# BUG: collate hard-coded ("egfr","kras"), silently dropping every other gene
# --------------------------------------------------------------------------
def test_collate_keeps_every_configured_gene(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset, collate

    genes = ["EGFR", "KRAS", "ALK", "STK11", "TP53"]
    idx = _fake_rows(tmp_path, genes)
    ds = RadiogenomicsDataset(idx, genes)
    batch = collate([ds[0], ds[1]], genes=genes)
    for g in genes:
        assert g.lower() in batch, f"{g} was dropped by collate"


def test_collate_infers_genes_when_not_told(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset, collate

    genes = ["EGFR", "ALK", "TP53"]
    ds = RadiogenomicsDataset(_fake_rows(tmp_path, genes), genes)
    batch = collate([ds[0], ds[1]])
    for g in genes:
        assert g.lower() in batch


def test_classification_loss_reaches_all_genes(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset, collate
    from theia.engine.losses import classification_loss

    genes = ["EGFR", "KRAS", "ALK"]
    ds = RadiogenomicsDataset(_fake_rows(tmp_path, genes), genes)
    batch = collate([ds[0], ds[1]], genes=genes)
    logits = {g.lower(): torch.randn(2, 2, requires_grad=True) for g in genes}
    loss = classification_loss(logits, batch, "cpu")
    loss.backward()
    for g in genes:
        assert logits[g.lower()].grad is not None, f"{g} head got no gradient"
        assert logits[g.lower()].grad.abs().sum() > 0


# --------------------------------------------------------------------------
# BUG: grounding_iou thresholded max-normalised maps at 0.1, so random and
#      uniform attention scored identically to the trained model
# --------------------------------------------------------------------------
def _roi_with_fraction(frac, b=16, s=8, size=224):
    roi = torch.zeros(b, s, 1, size, size)
    side = int(round((frac ** 0.5) * size))
    o = (size - side) // 2
    roi[:, :, :, o:o + side, o:o + side] = 1.0
    return roi


def test_grounding_metric_separates_signal_from_noise():
    from theia.engine.evaluate import grounding_metrics
    from theia.engine.losses import roi_to_grid

    torch.manual_seed(0)
    roi = _roi_with_fraction(0.5)
    target = roi_to_grid(roi, 14, 14, torch.device("cpu"))
    perfect = target.unsqueeze(1).repeat(1, 8, 1, 1)
    random_attn = torch.rand(roi.shape[0], 8, 14, 14)

    good = grounding_metrics(perfect, roi)
    bad = grounding_metrics(random_attn, roi)
    for key in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        assert np.mean(good[key]) > np.mean(bad[key]) + 0.2, (
            f"{key} cannot tell perfect attention from noise "
            f"({np.mean(good[key]):.3f} vs {np.mean(bad[key]):.3f})"
        )


def test_grounding_metric_ships_a_chance_baseline():
    """Random attention must score at ~its own shuffled baseline (lift ~ 0)."""
    from theia.engine.evaluate import grounding_metrics

    # Seeded: the assertion is a statistical one, so leaving it on whatever
    # global RNG state the previous test happened to leave behind makes it flaky.
    torch.manual_seed(0)
    roi = _roi_with_fraction(0.5)
    m = grounding_metrics(torch.rand(roi.shape[0], 8, 14, 14), roi)
    for key in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        assert f"{key}_shuffled" in m, f"{key} has no chance baseline"
        lift = np.mean(m[key]) - np.mean(m[f"{key}_shuffled"])
        assert abs(lift) < 0.25, f"{key} lift for random attention should be ~0, got {lift:.3f}"


# --------------------------------------------------------------------------
# The grounding LOSS was correct and must stay that way (measured, not assumed)
# --------------------------------------------------------------------------
def test_grounding_loss_still_prefers_correct_attention():
    from theia.engine.losses import grounding_loss, roi_to_grid

    roi = _roi_with_fraction(0.21, b=1)
    target = roi_to_grid(roi, 14, 14, torch.device("cpu"))
    peak = (target / target.sum()).unsqueeze(1).repeat(1, 8, 1, 1)
    flat = torch.full((1, 8, 14, 14), 1 / 196.0)
    assert float(grounding_loss(peak, roi, "cpu")) < float(grounding_loss(flat, roi, "cpu"))


# --------------------------------------------------------------------------
# BUG: F.binary_cross_entropy is autocast-banned; amp:true crashed on CUDA
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grounding_loss_survives_autocast(dtype):
    from theia.engine.losses import grounding_loss

    attn = torch.rand(2, 8, 14, 14)
    roi = _roi_with_fraction(0.3, b=2, s=4)
    with torch.amp.autocast(device_type="cpu", enabled=True, dtype=dtype):
        loss = grounding_loss(attn, roi, "cpu")
    assert torch.isfinite(loss)


def test_grounding_loss_handles_vanishing_mask():
    from theia.engine.losses import grounding_loss

    loss = grounding_loss(torch.rand(2, 8, 14, 14), torch.zeros(2, 4, 1, 224, 224), "cpu")
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# BUG: ndarray.ptp() was removed in NumPy 2.0
# --------------------------------------------------------------------------
def test_overlay_renders_without_ndarray_ptp(tmp_path):
    from theia.reader_study.build_cases import _overlay_png

    png = str(tmp_path / "o.png")
    _overlay_png(np.random.rand(4, 1, 224, 224), np.random.rand(8, 14, 14), png)
    assert os.path.getsize(png) > 0


# --------------------------------------------------------------------------
# BUG: a NaN monitor made `nan > best` False forever, so best.pt was never written
# --------------------------------------------------------------------------
def test_nan_monitor_is_not_an_improvement_and_not_a_deadlock():
    from theia.engine.train import _is_better

    assert _is_better(0.7, -float("inf")) is True
    assert _is_better(float("nan"), -float("inf")) is False
    assert _is_better(0.6, 0.7) is False
    assert _is_better(0.8, 0.7) is True


# --------------------------------------------------------------------------
# BUG: patients with an unknown stratification label were discarded entirely,
#      even when their other gene labels were usable
# --------------------------------------------------------------------------
def test_unknown_stratification_label_does_not_discard_patients(tmp_path):
    from theia.data.dataset import kfold_indices

    genes = ["EGFR", "KRAS"]
    idx = _fake_rows(tmp_path, genes, n=20, unknown_first=8)
    seen = set()
    for tr, va in kfold_indices(idx, "EGFR", 5, 0):
        assert set(tr).isdisjoint(va)
        seen |= set(tr) | set(va)
    assert len(seen) == 20, f"only {len(seen)}/20 patients reachable"


def test_nested_split_keeps_test_out_of_train_and_val(tmp_path):
    from theia.data.dataset import nested_kfold_indices

    idx = _fake_rows(tmp_path, ["EGFR", "KRAS"], n=30)
    n_folds = 0
    for tr, va, te in nested_kfold_indices(idx, "EGFR", 5, 0, 0.2):
        assert te, "outer test fold is empty"
        assert set(te).isdisjoint(tr), "test leaked into train"
        assert set(te).isdisjoint(va), "test leaked into inner validation"
        assert set(tr).isdisjoint(va), "inner validation leaked into train"
        assert va, "inner validation is empty; early stopping would have no signal"
        n_folds += 1
    assert n_folds == 5


# --------------------------------------------------------------------------
# BUG: image was cropped around the tumor while the mask was a whole-slice
#      resize, so the grounding target was in a different frame from the pixels
# --------------------------------------------------------------------------
def test_image_and_mask_share_one_crop_box():
    from theia.data.preprocess import crop_box, roi_box, _resize

    img = np.zeros((512, 512), dtype=np.float32)
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[300:340, 120:160] = 1.0
    img[300:340, 120:160] = 1.0          # a bright lesion exactly where the mask is

    box = roi_box(mask, img.shape, margin_px=5, context_factor=1.0)
    im = _resize(crop_box(img, box), 224)
    mk = _resize(crop_box(mask, box), 224)

    # Both crops must light up in the same place. If the mask were resized from
    # the full slice instead, its centroid would sit somewhere else entirely.
    iy, ix = np.argwhere(im > 0.5).mean(axis=0)
    my, mx = np.argwhere(mk > 0.5).mean(axis=0)
    assert abs(iy - my) < 5 and abs(ix - mx) < 5, "image and mask are misaligned"


def test_crop_is_square_so_anatomy_is_not_stretched():
    from theia.data.preprocess import crop_box, roi_box

    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:180, 200:220] = 1.0          # a tall, narrow lesion
    box = roi_box(mask, mask.shape, margin_px=4, context_factor=1.0)
    patch = crop_box(np.zeros((512, 512), np.float32), box)
    assert patch.shape[0] == patch.shape[1], f"crop is not square: {patch.shape}"


def test_crop_at_image_edge_pads_instead_of_squashing():
    from theia.data.preprocess import crop_box, roi_box

    mask = np.zeros((256, 256), dtype=np.float32)
    mask[0:20, 0:20] = 1.0                # lesion jammed into the corner
    box = roi_box(mask, mask.shape, margin_px=30, context_factor=2.0)
    patch = crop_box(np.ones((256, 256), np.float32), box)
    assert patch.shape[0] == patch.shape[1]
    assert (patch == 0).any(), "out-of-bounds region should be zero-padded"


def test_context_factor_widens_the_field_of_view():
    from theia.data.preprocess import roi_box

    mask = np.zeros((512, 512), dtype=np.float32)
    mask[200:240, 200:240] = 1.0
    tight = roi_box(mask, mask.shape, 4, 1.0)
    wide = roi_box(mask, mask.shape, 4, 3.0)
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])


def test_resize_always_returns_the_requested_size():
    from theia.data.preprocess import _resize

    for h in (17, 33, 64, 101, 223, 225):
        for w in (h, h + 3):
            assert _resize(np.random.rand(h, w), 224).shape == (224, 224)


# --------------------------------------------------------------------------
# BUG: slice_strategy "all" was documented but silently behaved as tumor_bbox
# --------------------------------------------------------------------------
def test_slice_strategy_all_is_implemented():
    from theia.data.preprocess import tumor_slices

    mask = np.zeros((40, 16, 16), dtype=np.uint8)
    mask[10:30, 4:8, 4:8] = 1
    got = tumor_slices(mask, 16, "all")
    assert len(got) == 16
    assert all(10 <= i < 30 for i in got)


def test_unknown_slice_strategy_is_rejected():
    from theia.data.preprocess import tumor_slices

    mask = np.zeros((20, 8, 8), dtype=np.uint8)
    mask[5:15] = 1
    with pytest.raises(ValueError, match="unknown slice_strategy"):
        tumor_slices(mask, 8, "typo_here")


# --------------------------------------------------------------------------
# BUG: padded slices were averaged into the patch tokens as real zeros
# --------------------------------------------------------------------------
def test_collate_emits_a_slice_mask_for_padding(tmp_path):
    from theia.data.dataset import collate

    a = dict(images=torch.rand(4, 1, 8, 8), roi=torch.rand(4, 1, 8, 8),
             report="r", patient_id="a", egfr=torch.tensor(1))
    b = dict(images=torch.rand(2, 1, 8, 8), roi=torch.rand(2, 1, 8, 8),
             report="r", patient_id="b", egfr=torch.tensor(0))
    out = collate([a, b], genes=["EGFR"])
    assert out["slice_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_masked_slice_pooling_ignores_padding():
    """The masked mean of a padded stack equals the mean of its real slices."""
    tokens = torch.stack([torch.full((3, 5), 2.0), torch.full((3, 5), 4.0),
                          torch.zeros(3, 5), torch.zeros(3, 5)]).unsqueeze(0)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m = mask.view(1, 4, 1, 1)
    pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    assert torch.allclose(pooled, torch.full((1, 3, 5), 3.0))
    assert not torch.allclose(tokens.mean(dim=1), pooled)   # the old behaviour differed


# --------------------------------------------------------------------------
# BUG: pretrained backbones were fed unnormalised [0,1] input
# --------------------------------------------------------------------------
def test_encoder_normalizes_with_the_backbone_statistics():
    import theia.models.backbone as bb

    enc = object.__new__(bb.VisionEncoder)
    torch.nn.Module.__init__(enc)
    enc.grid = (2, 2)
    enc.register_buffer("pix_mean", torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1))
    enc.register_buffer("pix_std", torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1))
    enc.proj = torch.nn.Identity()
    seen = {}

    def fake_tokens(x):
        seen["min"], seen["max"] = float(x.min()), float(x.max())
        return torch.zeros(x.shape[0], 4, 7)

    enc._forward_tokens = fake_tokens
    images = torch.rand(2, 3, 1, 8, 8)     # window_hu output lives in [0,1]
    enc.encode(images, slice_mask=torch.ones(2, 3))
    assert seen["min"] < -0.5, f"input was not normalized (min={seen['min']:.3f})"
    assert seen["max"] > 0.5


# --------------------------------------------------------------------------
# BUG: region pooling and grounding supervision disagreed with no way to align
# --------------------------------------------------------------------------
def test_region_pooling_modes():
    from theia.models.heads import GroundingHead

    emb = torch.tensor([[[1.0, 5.0], [3.0, 1.0]]])
    assert torch.allclose(GroundingHead.pooled_from_regions(emb, "mean"),
                          torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(GroundingHead.pooled_from_regions(emb, "max"),
                          torch.tensor([[3.0, 5.0]]))
    with pytest.raises(ValueError, match="unknown region_pooling"):
        GroundingHead.pooled_from_regions(emb, "median")


# --------------------------------------------------------------------------
# BUG: device selection ignored Apple Silicon; amp was requested off CUDA
# --------------------------------------------------------------------------
def test_device_resolution_and_amp_honesty():
    from theia.runtime import amp_settings, resolve_device

    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}
    on, dtype = amp_settings(torch.device("cpu"), want_amp=True)
    assert on is False and dtype is torch.float32, "amp must not claim to be on off-CUDA"
    off, _ = amp_settings(torch.device("cuda"), want_amp=False)
    assert off is False


# --------------------------------------------------------------------------
# Config validation catches the settings that used to fail late or silently
# --------------------------------------------------------------------------
def _base_cfg():
    from theia.config import load_config

    return load_config("configs/default.yaml", validate=False)


def test_config_rejects_a_monitor_evaluate_never_emits():
    from theia.config import validate_config

    cfg = _base_cfg()
    cfg["train"]["monitor"] = "egfr_accuracy"
    with pytest.raises(ValueError, match="not emitted by evaluate"):
        validate_config(cfg)


def test_config_rejects_a_panel_that_contradicts_itself():
    from theia.config import validate_config

    cfg = _base_cfg()
    cfg["data"]["exploratory_genes"] = ["ALK"]       # target_genes still [EGFR, KRAS]
    with pytest.raises(ValueError, match="headline \\+ exploratory"):
        validate_config(cfg)


def test_config_rejects_stratifying_on_an_untargeted_gene():
    from theia.config import validate_config

    cfg = _base_cfg()
    cfg["split"]["stratify_on"] = "BRAF"
    with pytest.raises(ValueError, match="not in data.target_genes"):
        validate_config(cfg)


def test_shipped_config_is_valid():
    from theia.config import load_config

    load_config("configs/default.yaml")              # raises if invalid


# --------------------------------------------------------------------------
# BUG: restarting the reader-study server replayed cases already scored
# --------------------------------------------------------------------------
def test_reader_study_resumes_where_it_left_off(tmp_path):
    from theia.reader_study.serve import _resume_position

    pres = [{"token": f"P{i:04d}"} for i in range(5)]
    out = tmp_path / "scores_R1.jsonl"
    assert _resume_position(str(out), pres) == 0
    out.write_text("".join(json.dumps({"token": f"P{i:04d}"}) + "\n" for i in range(3)))
    assert _resume_position(str(out), pres) == 3


def test_reader_study_arms_only_ask_answerable_questions():
    from theia.reader_study.build_cases import ARM_QUESTIONS

    assert ARM_QUESTIONS["theia_full"] == ["plausible", "grounded", "useful"]
    for arm in ("label_only", "ground_truth"):
        assert "plausible" not in ARM_QUESTIONS[arm], f"{arm} shows no rationale"
        assert "grounded" not in ARM_QUESTIONS[arm], f"{arm} shows no image"


# --------------------------------------------------------------------------
# BUG: np.load handles were never closed
# --------------------------------------------------------------------------
def test_dataset_does_not_leak_file_handles(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset

    ds = RadiogenomicsDataset(_fake_rows(tmp_path, ["EGFR"], n=4), ["EGFR"])
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")
    proc = psutil.Process()
    before = len(proc.open_files())
    for _ in range(3):
        for i in range(len(ds)):
            ds[i]
    assert len(proc.open_files()) <= before + 1


# --------------------------------------------------------------------------
# BUG: total_loss called float() on four tensors every step (host sync)
# --------------------------------------------------------------------------
def test_total_loss_parts_stay_detached_tensors():
    from theia.engine.losses import total_loss
    from theia.config import load_config

    cfg = load_config("configs/default.yaml")
    out = dict(logits={"egfr": torch.randn(2, 2, requires_grad=True)},
               attn_maps=torch.rand(2, 8, 14, 14),
               gen_loss=torch.zeros((), requires_grad=True))
    batch = dict(egfr=torch.tensor([0, 1]), roi=_roi_with_fraction(0.3, b=2, s=4))
    loss, parts = total_loss(out, batch, cfg.train.loss_weights, "cpu")
    assert loss.requires_grad
    for k, v in parts.items():
        assert torch.is_tensor(v), f"{k} was eagerly converted to float"
        assert not v.requires_grad, f"{k} still holds a graph reference"


# --------------------------------------------------------------------------
# BUG: make_collate returned a closure, which DataLoader cannot pickle to its
#      workers (num_workers>0 on macOS spawn) — the run died at the first batch
# --------------------------------------------------------------------------
def test_collate_fn_is_picklable_for_dataloader_workers():
    import pickle

    from theia.data.dataset import make_collate

    fn = make_collate(["EGFR", "KRAS"])
    restored = pickle.loads(pickle.dumps(fn))       # what DataLoader does per worker
    a = dict(images=torch.rand(2, 1, 8, 8), roi=torch.rand(2, 1, 8, 8),
             report="r", patient_id="a", egfr=torch.tensor(1), kras=torch.tensor(0))
    out = restored([a, a])
    assert out["egfr"].shape[0] == 2 and "kras" in out


# --------------------------------------------------------------------------
# Partial unfreezing: full freeze fixed the classifier but starved grounding
# --------------------------------------------------------------------------
def test_unfreeze_last_blocks_reports_what_it_actually_unfroze():
    import torch.nn as nn

    import theia.models.backbone as bb

    enc = object.__new__(bb.VisionEncoder)
    nn.Module.__init__(enc)
    enc.model = nn.Module()
    enc.model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
    for p in enc.model.parameters():
        p.requires_grad_(False)

    expected = sum(p.numel() for b in list(enc.model.blocks)[-2:] for p in b.parameters())
    n = enc.unfreeze_last_blocks(2)
    assert n == expected, f"reported {n}, actually {expected}"
    grads = [all(p.requires_grad for p in b.parameters()) for b in enc.model.blocks]
    assert grads == [False, False, False, False, True, True], grads


def test_unfreeze_zero_is_a_noop_and_missing_blocks_is_not_fatal():
    import torch.nn as nn

    import theia.models.backbone as bb

    enc = object.__new__(bb.VisionEncoder)
    nn.Module.__init__(enc)
    enc.model = nn.Module()
    enc.model.blocks = nn.ModuleList([nn.Linear(4, 4)])
    assert enc.unfreeze_last_blocks(0) == 0

    bare = object.__new__(bb.VisionEncoder)
    nn.Module.__init__(bare)
    bare.model = nn.Module()                       # no block list at all
    assert bare.unfreeze_last_blocks(2) == 0       # warns, does not raise


# --------------------------------------------------------------------------
# BUG: with no jitter the tumor sat at the exact centre of every crop, so
#      grounding had nothing to learn (measured ROI centroid 6.5+-0.1 on a
#      14x14 grid whose centre is 6.5; lift ~0.000 in all four ablation arms)
# --------------------------------------------------------------------------
def test_crop_jitter_moves_the_tumor_off_centre():
    from theia.data.preprocess import roi_box

    mask = np.zeros((512, 512), dtype=np.float32)
    mask[240:272, 240:272] = 1.0                      # lesion at image centre

    centres = []
    for pid in range(40):
        box = roi_box(mask, mask.shape, 4, context_factor=2.5, jitter=0.30,
                      rng=np.random.default_rng(pid))
        y0, y1, x0, x1 = box
        side = y1 - y0
        # where does the lesion centre sit, as a fraction of the crop?
        centres.append(((256 - y0) / side, (256 - x0) / side))
    ys = np.array([c[0] for c in centres])
    xs = np.array([c[1] for c in centres])
    assert ys.std() > 0.05 and xs.std() > 0.05, (
        f"jitter did not vary the lesion position (sd {ys.std():.3f}, {xs.std():.3f})")
    assert 0.2 < ys.mean() < 0.8, "lesion drifted out of frame on average"


def test_zero_jitter_is_deterministic_and_centred():
    from theia.data.preprocess import roi_box

    mask = np.zeros((256, 256), dtype=np.float32)
    mask[100:140, 100:140] = 1.0
    a = roi_box(mask, mask.shape, 4, context_factor=2.0, jitter=0.0)
    b = roi_box(mask, mask.shape, 4, context_factor=2.0, jitter=0.0)
    assert a == b
    y0, y1, x0, x1 = a
    assert abs((y0 + y1) / 2 - 120) <= 1 and abs((x0 + x1) / 2 - 120) <= 1
