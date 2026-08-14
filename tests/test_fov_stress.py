"""The off-centre stress test only means anything if the shift is correct.

If the ROI were translated and the image were not, the "trained model follows the
lesion" result would be an artifact of a broken transform rather than evidence
about attention -- and it would look like a strong positive result either way. So
the geometry is asserted here rather than eyeballed on an overlay.
"""
from __future__ import annotations

import math

import pytest
import torch

from theia.analysis.fov_stress import shift_batch

H = W = 64


def _fixture(b: int = 8):
    img = torch.rand(b, 2, 1, H, W) * 0.5 + 0.22   # air ~0.22, as HU-windowed
    roi = torch.zeros(b, 2, 1, H, W)
    roi[..., H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4] = 1.0
    return img, roi


def test_zero_offset_is_the_identity():
    img, roi = _fixture()
    i2, r2 = shift_batch(img, roi, 0.0, 0)
    assert torch.equal(i2, img)
    assert torch.equal(r2, roi)


@pytest.mark.parametrize("frac", [0.10, 0.20, 0.35])
def test_roi_centroid_lands_where_the_geometry_says(frac):
    img, roi = _fixture()
    _, r2 = shift_batch(img, roi, frac, 0)
    for k in range(r2.shape[0]):
        theta = 2 * math.pi * (k % 8) / 8.0
        want_y = H / 2 + frac * H * math.sin(theta)
        want_x = W / 2 + frac * W * math.cos(theta)
        idx = r2[k, 0, 0].nonzero().float()
        assert idx.numel() > 0, f"ROI left the frame at f={frac}, k={k}"
        cy, cx = idx.mean(0).tolist()
        # +0.5 because a pixel's centroid sits at its centre, and the source
        # block is even-sized; 1.5 px of slack covers the integer rounding.
        assert abs(cy + 0.5 - want_y) <= 1.5, f"y {cy} vs {want_y}"
        assert abs(cx + 0.5 - want_x) <= 1.5, f"x {cx} vs {want_x}"


@pytest.mark.parametrize("frac", [0.10, 0.20, 0.35])
def test_image_and_roi_move_together(frac):
    """The whole point: the lesion moves WITH its pixels, not away from them."""
    img, roi = _fixture()
    # Stamp a bright marker exactly over the ROI so the two can be compared.
    img = img.clone()
    img[roi > 0] = 1.0
    i2, r2 = shift_batch(img, roi, frac, 0)
    for k in range(i2.shape[0]):
        inside = i2[k][r2[k] > 0]
        assert inside.numel() > 0
        assert torch.allclose(inside, torch.ones_like(inside)), (
            f"image and ROI are out of register at f={frac}, k={k}")


@pytest.mark.parametrize("frac", [0.20, 0.35])
def test_vacated_region_is_filled_with_air_not_black(frac):
    """Fill is PER SAMPLE, so this must be checked per sample.

    Comparing the batch minimum against one sample's minimum fails for the
    boring reason that other samples are darker.
    """
    img, roi = _fixture()
    i2, _ = shift_batch(img, roi, frac, 0)
    for k in range(img.shape[0]):
        assert float(i2[k].min()) >= float(img[k].min()) - 1e-6, (
            f"sample {k}: padding must be that volume's own minimum (air), not "
            f"zero -- zero is -1350 HU after this window and the encoder has "
            f"never seen it")


def test_direction_varies_across_the_batch():
    """A single shared direction would confound the result with crop anisotropy."""
    img, roi = _fixture()
    _, r2 = shift_batch(img, roi, 0.25, 0)
    centroids = {tuple(round(v) for v in r2[k, 0, 0].nonzero().float().mean(0).tolist())
                 for k in range(r2.shape[0])}
    assert len(centroids) >= 6, f"only {len(centroids)} distinct directions"


def test_index_offset_shifts_the_direction_schedule():
    """Batches must continue the schedule, not restart it every batch."""
    img, roi = _fixture(b=1)
    a = shift_batch(img, roi, 0.25, 0)[1]
    b = shift_batch(img, roi, 0.25, 1)[1]
    assert not torch.equal(a, b)
