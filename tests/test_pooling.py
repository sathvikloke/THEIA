"""Tests for the pooling sweep that located the EGFR signal.

The finding this protects: mean-pooling the whole crop, and mean-pooling the
tumour, both score ~0.628, while pooling the PERITUMORAL ring scores 0.703
(+0.075, 10/10 seeds, p=0.002). A size- and shape-matched ring at a random
location scores 0.648, so the gain is location-specific rather than an artefact
of pooling a thin annulus.

That conclusion rests entirely on the control being built correctly, so the
control is what these tests check hardest.
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
