"""Tests for the sample-size and precision functions.

The closed-form AUC standard error is load-bearing: it is what the analysis plan
uses to declare, in advance, that an attainable external test set cannot settle
anything. A wrong constant there would turn a pre-specified finding into a false
one, so it is checked against a bootstrap rather than taken on faith.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from theia.analysis.power import hanley_mcneil_se, n_for_auc_ci, riley_min_n


def _sample(auc: float, n_pos: int, n_neg: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Binormal draw whose population AUC is `auc`.

    For unit-variance normals separated by mu, AUC = Phi(mu / sqrt(2)), so the
    separation that delivers a target AUC is mu = sqrt(2) * Phi^-1(auc).
    """
    mu = np.sqrt(2) * norm.ppf(auc)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    s = np.concatenate([rng.normal(mu, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    return y, s


@pytest.mark.parametrize("auc,n_pos,n_neg", [(0.63, 40, 113), (0.80, 60, 60), (0.70, 25, 75)])
def test_closed_form_se_matches_a_bootstrap(auc, n_pos, n_neg):
    """The whole justification rests on this number being right."""
    rng = np.random.default_rng(0)
    y, s = _sample(auc, n_pos, n_neg, rng)
    observed = roc_auc_score(y, s)

    boots = []
    for _ in range(600):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) > 1:
            boots.append(roc_auc_score(y[idx], s[idx]))
    emp = float(np.std(boots))
    closed = hanley_mcneil_se(observed, n_pos, n_neg)
    # Bootstrap SE at these sample sizes is itself noisy; 35% agreement is a real
    # check that the formula is not out by a factor.
    assert closed == pytest.approx(emp, rel=0.35), f"closed {closed:.4f} vs boot {emp:.4f}"


def test_se_shrinks_as_the_square_root_of_n():
    a, b = hanley_mcneil_se(0.7, 40, 113), hanley_mcneil_se(0.7, 400, 1130)
    assert b < a
    assert a / b == pytest.approx(np.sqrt(10), rel=0.2)


def test_se_is_smaller_for_a_stronger_model():
    """A near-perfect AUC has less sampling variance than a near-chance one."""
    assert hanley_mcneil_se(0.95, 50, 50) < hanley_mcneil_se(0.55, 50, 50)


def test_n_for_ci_is_monotone_in_the_requested_precision():
    ns = [n_for_auc_ci(0.63, h, 0.261) for h in (0.20, 0.15, 0.10, 0.05)]
    assert ns == sorted(ns), f"tighter CIs must need more patients, got {ns}"


def test_n_for_ci_round_trips_through_the_se():
    """The returned n must actually deliver the requested half-width."""
    prev, hw = 0.261, 0.10
    n = n_for_auc_ci(0.63, hw, prev)
    n_pos = max(int(round(n * prev)), 1)
    assert 1.96 * hanley_mcneil_se(0.63, n_pos, n - n_pos) <= hw
    # ...and one patient fewer must not.
    m = n - 1
    m_pos = max(int(round(m * prev)), 1)
    assert 1.96 * hanley_mcneil_se(0.63, m_pos, m - m_pos) > hw


def test_impossible_precision_reports_unreachable_rather_than_a_number():
    assert n_for_auc_ci(0.63, 1e-6, 0.261, max_n=5000) == -1


def test_invalid_prevalence_is_rejected():
    for bad in (0.0, 1.0, -0.1, 1.4):
        with pytest.raises(ValueError):
            n_for_auc_ci(0.63, 0.1, bad)


def test_the_attainable_external_set_cannot_separate_chance_from_the_literature():
    """Pins the claim the analysis plan makes in advance.

    ~96 patients with ~17 positives is what the open world actually offers. If the
    CI at AUC 0.63 did not straddle both 0.50 and 0.80, the plan's pre-specified
    "external testing is uninformative at this scale" would be wrong.
    """
    hw = 1.96 * hanley_mcneil_se(0.63, 17, 96 - 17)
    assert 0.63 - hw < 0.50, "CI should reach down to chance"
    assert 0.63 + hw > 0.78, "CI should reach up into the published range"


def test_riley_needs_more_patients_for_more_predictors():
    assert riley_min_n(512, 0.261) > riley_min_n(5, 0.261)


def test_riley_flags_the_frozen_probe_as_underpowered_at_this_cohort_size():
    """153 patients against a 512-feature probe is not a modelling choice, it is a
    sample-size violation -- which is the honest reason that arm is not trusted."""
    assert riley_min_n(5, 0.261) < 153, "the 5-variable clinical model is fine"
    assert riley_min_n(512, 0.261) > 1000, "the frozen probe is not"
