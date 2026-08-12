"""Tests for calibration reporting.

The headline finding these support is uncomfortable enough that the arithmetic
had better be right: THEIA's Brier score exceeds the uncertainty floor, meaning
its raw probabilities are worse than predicting the base rate for every patient.
That is a claim about a number, so the number gets tested.
"""
from __future__ import annotations

import numpy as np
import pytest

from theia.analysis.calibration import (brier_decomposition, calibration_report,
                                        calibration_slope_intercept, curve)

rng = np.random.default_rng(0)


def _well_calibrated(n=4000, base=0.3):
    """Probabilities that are true by construction: draw p, then draw y ~ Bern(p)."""
    p = rng.uniform(0.02, 0.98, n)
    p = p * (base / p.mean())
    p = np.clip(p, 1e-3, 1 - 1e-3)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_perfect_calibration_gives_slope_one_and_intercept_zero():
    y, p = _well_calibrated()
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_overconfident_predictions_give_slope_below_one():
    """Pushing probabilities toward 0 and 1 must be detected as slope < 1.

    This is the signature the report exists to catch, and it is what a
    40-positive cohort is expected to produce.
    """
    y, p = _well_calibrated()
    sharp = 1 / (1 + np.exp(-3.0 * np.log(p / (1 - p))))     # triple the logit
    slope, _ = calibration_slope_intercept(y, sharp)
    assert slope < 0.6, f"overconfidence not detected, slope={slope}"


def test_systematic_overprediction_moves_the_intercept_negative():
    y, p = _well_calibrated()
    shifted = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + 1.5)))  # predict too high
    _, intercept = calibration_slope_intercept(y, shifted)
    assert intercept < -1.0
    rep = calibration_report(y, shifted)
    assert rep["oe_ratio"] < 1.0, "observed should fall short of expected"


def test_murphy_decomposition_reconstructs_the_brier_score():
    """brier = reliability - resolution + uncertainty, up to binning error."""
    y, p = _well_calibrated(n=3000)
    d = brier_decomposition(y, p, n_bins=10)
    assert (d["reliability"] - d["resolution"] + d["uncertainty"]) == pytest.approx(
        d["brier"], abs=0.01)


def test_uncertainty_is_the_base_rate_variance():
    """The floor a constant predictor achieves. THEIA is measured against it."""
    y, p = _well_calibrated(base=0.26)
    d = brier_decomposition(y, p)
    assert d["uncertainty"] == pytest.approx(d["base_rate"] * (1 - d["base_rate"]))


def test_constant_base_rate_predictor_scores_exactly_the_uncertainty():
    """Anchors the comparison the results section makes."""
    y = np.array([1] * 26 + [0] * 74)
    p = np.full(100, 0.26)
    d = brier_decomposition(y, p)
    assert d["brier"] == pytest.approx(d["uncertainty"], abs=1e-3)
    assert d["resolution"] == pytest.approx(0.0, abs=1e-9), "a constant resolves nothing"


def test_single_class_input_returns_nan_not_a_number():
    y = np.ones(20, dtype=int)
    p = rng.uniform(0.1, 0.9, 20)
    slope, intercept = calibration_slope_intercept(y, p)
    assert np.isnan(slope) and np.isnan(intercept)


def test_curve_uses_equal_count_bins():
    """At n=153 equal-width bins go empty; equal-count bins do not."""
    y, p = _well_calibrated(n=153)
    pts = curve(y, p, n_bins=5)
    assert len(pts) == 5
    assert all(pt["n"] > 0 for pt in pts)
    assert sum(pt["n"] for pt in pts) == 153
    # Monotone in predicted probability, since bins are ordered by p.
    means = [pt["mean_predicted"] for pt in pts]
    assert means == sorted(means)
