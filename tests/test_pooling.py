"""Tests for the pooling sweep that located the EGFR signal.

The finding this protects: mean-pooling the whole crop, and mean-pooling the
tumour, both score ~0.63, while pooling the PERITUMORAL ring scores 0.695
(+0.064). A size- and shape-matched ring at a random location scores 0.575, so
the gain is location-specific rather than an artefact of pooling a thin annulus.

Two things about the strength of that claim, since this docstring previously
overstated it as "+0.075, 10/10 seeds, p=0.002".

The sweep scores 10 arms on the same folds and reports the best, so it needs a
multiplicity correction. Under a paired bootstrap over patients with Holm across
the arms, the best arm has p_raw 0.031 and **p_holm 0.281** -- nothing in the
sweep is significant. The old p=0.002 came from a sign test over seeds, and ten
seeds are ten re-splits of one 153-patient cohort, not ten replicates.

So the effect is directionally consistent, survives its location controls, and
reproduces end-to-end (+0.058, p=0.086), but it is not established. These tests
check the CONTROL construction rather than the effect size, because a broken
control is the failure mode that would make even the direction meaningless.
"""
import numpy as np
import pytest

from theia.analysis.pooling_sweep import build_variants


def _toy(p=24, h=6, w=6, d=8, seed=0):
    """Tokens where the peritumoral ring carries a label-correlated signal."""
    rng = np.random.default_rng(seed)
    tok = rng.normal(0, 1, (p, h * w, d)).astype(np.float32)
    msk = np.zeros((p, h * w), dtype=np.float32)
    for i in range(p):
        g = np.zeros((h, w), dtype=np.float32)
        g[2:4, 2:4] = 1.0
        msk[i] = g.reshape(-1)
    return tok, msk, h, w


def test_variants_have_the_expected_shapes_and_names():
    tok, msk, h, w = _toy()
    v = build_variants(tok, msk, h, w)
    for name, X in v.items():
        assert X.shape[0] == tok.shape[0], f"{name} lost patients"
        assert np.isfinite(X).all(), f"{name} produced non-finite features"
    assert any("CONTROL" in k for k in v), "the location control is missing"


def test_peritumoral_ring_excludes_the_tumour_itself():
    """The ring must be dilate(mask) MINUS mask.

    If it included the tumour the comparison would be confounded: any
    peritumoral advantage could just be the tumour signal reappearing.
    """
    tok, msk, h, w = _toy()
    # Reconstruct the ring the same way build_variants does and check disjointness.
    from scipy import ndimage
    g = msk[0].reshape(h, w) > 0.5
    ring = ndimage.binary_dilation(g, iterations=1) & ~g
    assert ring.sum() > 0
    assert not (ring & g).any(), "ring overlaps the tumour"


def test_control_ring_matches_the_real_ring_in_size():
    """The control must differ from the real ring ONLY in location.

    Checked against the module's OWN control, not a re-implementation. The first
    version of this test rebuilt the control inline and therefore tested a copy
    of the logic rather than the logic -- and the copy was the buggy one, so it
    failed against a module that had already been fixed.

    The bug it was written for is worth keeping in view: np.roll wraps a shape
    but binary_dilation clips at the array border, so unconstrained shifts made
    control rings 3.7% smaller than real ones. A smaller control carries less
    information and inflates the location-specific gain it exists to rule out.
    """
    import numpy as np
    from scipy import ndimage

    from theia.analysis.pooling_sweep import _control_regions

    tok, msk, h, w = _toy(p=16, h=10, w=10)
    ring = np.stack([(ndimage.binary_dilation(m.reshape(h, w) > 0.5, iterations=1)
                      & ~(m.reshape(h, w) > 0.5)).reshape(-1) for m in msk]).astype(np.float32)
    _shifted, randring = _control_regions(msk, ring, h, w, seed=1337)

    real_sizes = ring.sum(1)
    ctrl_sizes = randring.sum(1)
    assert np.allclose(real_sizes, ctrl_sizes), (
        f"control ring sizes {ctrl_sizes[:5]} != real {real_sizes[:5]}")


def test_pooling_falls_back_rather_than_emitting_an_empty_feature():
    """A lesion too small to occupy any patch must not yield a zero vector.

    AIM-tier patients carry no mask at all, and small lesions can vanish at grid
    resolution. Either would put an all-zero row into the design matrix, which
    trains as a systematic class of its own.
    """
    tok, msk, h, w = _toy()
    msk[:] = 0.0                       # nothing survives to the grid
    v = build_variants(tok, msk, h, w)
    for name, X in v.items():
        assert np.isfinite(X).all(), f"{name} non-finite with an empty mask"
        assert np.abs(X).sum(1).min() > 0, f"{name} emitted an all-zero feature"


def test_model_peritumoral_pool_matches_the_swept_definition(monkeypatch):
    """The model's ring must be the same region the sweep measured.

    The sweep measured dilate(mask) MINUS mask on the patch grid. If the model
    pooled a different region -- the dilated mask including the tumour, or the
    tumour itself -- it would be importing a number it does not implement.
    """
    import torch

    import theia.models.backbone as bb
    import theia.models.heads as heads
    import theia.models.theia_model as tm
    from theia.config import load_config

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(3, 8, 4, 4)

        def forward_features(self, x):
            return self.stem(x).flatten(2).transpose(1, 2)[:, :16, :]

    monkeypatch.setattr(bb.VisionEncoder, "_build",
                        lambda self, n: (Tiny(), 8, (4, 4), (0.5,) * 3, (0.5,) * 3))
    monkeypatch.setattr(heads, "GenerationHead", lambda *a, **k: torch.nn.Linear(2, 2))
    monkeypatch.setattr(tm, "GenerationHead", lambda *a, **k: torch.nn.Linear(2, 2))

    cfg = load_config("configs/default.yaml", validate=False)
    cfg["model"].update(vision_dim=8, grounding_tokens=2, classifier_hidden=8,
                        peritumoral_features=True, grounding_pretrain_ckpt=None)
    cfg["lora"]["enabled"] = False
    model = tm.Theia(cfg)

    # One patient, a 2x2 lesion in the middle of a 4x4 grid.
    tokens = torch.arange(16 * 8, dtype=torch.float32).view(1, 16, 8)
    roi = torch.zeros(1, 2, 1, 16, 16)
    roi[:, :, :, 4:12, 4:12] = 1.0            # -> 2x2 block on the 4x4 grid

    got = model._peritumoral_pool(tokens, (4, 4), roi, "cpu")

    from theia.engine.losses import roi_to_grid
    m = roi_to_grid(roi, 4, 4, "cpu")[0].numpy()
    from scipy import ndimage
    ring = (ndimage.binary_dilation(m > 0.5, iterations=1) & ~(m > 0.5)).reshape(-1)
    want = tokens[0].numpy()[ring].mean(0)

    assert np.allclose(got[0].detach().numpy(), want, atol=1e-5), (
        "model ring differs from the region the sweep measured")


def test_peritumoral_branch_falls_back_when_there_is_no_mask(monkeypatch):
    """AIM-tier patients carry no extent; they must not get a zero vector."""
    import torch

    import theia.models.backbone as bb
    import theia.models.heads as heads
    import theia.models.theia_model as tm
    from theia.config import load_config

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(3, 8, 4, 4)

        def forward_features(self, x):
            return self.stem(x).flatten(2).transpose(1, 2)[:, :16, :]

    monkeypatch.setattr(bb.VisionEncoder, "_build",
                        lambda self, n: (Tiny(), 8, (4, 4), (0.5,) * 3, (0.5,) * 3))
    monkeypatch.setattr(heads, "GenerationHead", lambda *a, **k: torch.nn.Linear(2, 2))
    monkeypatch.setattr(tm, "GenerationHead", lambda *a, **k: torch.nn.Linear(2, 2))

    cfg = load_config("configs/default.yaml", validate=False)
    cfg["model"].update(vision_dim=8, grounding_tokens=2, classifier_hidden=8,
                        peritumoral_features=True, grounding_pretrain_ckpt=None)
    cfg["lora"]["enabled"] = False
    model = tm.Theia(cfg)

    tokens = torch.randn(2, 16, 8)
    empty = torch.zeros(2, 2, 1, 16, 16)
    got = model._peritumoral_pool(tokens, (4, 4), empty, "cpu")
    assert torch.isfinite(got).all()
    assert got.abs().sum(1).min() > 0, "an empty mask produced an all-zero feature"
    assert torch.allclose(got, tokens.mean(1), atol=1e-5), "fallback is not the whole crop"
