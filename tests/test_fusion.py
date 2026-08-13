"""Tests for decision-level fusion and the pooling-sweep multiplicity correction.

The claim these protect is that no fusion topology beats the clinical model.
That claim is only worth anything if the stacker is genuinely cross-fitted -- a
stacker fitted on the same out-of-fold predictions it scores would beat
everything, and would mean nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from theia.analysis.fusion import (align, cross_fitted_stack, hard_vote,
                                   rank_within_fold, soft_vote)
from theia.analysis.pooling_sweep import holm_paired

rng = np.random.default_rng(7)


def _arm(ids, probs, y, folds, gene="egfr"):
    return [{"patient_id": i, "fold": f, f"{gene}_prob": p, f"{gene}_true": t}
            for i, p, t, f in zip(ids, probs, y, folds)]


def test_cross_fitted_stacker_cannot_memorise_its_own_fold():
    """The load-bearing property. Pure noise must not stack to a high AUC.

    A stacker fitted on all the data and evaluated on the same data would find
    spurious weights and report an inflated AUC. Cross-fitting must leave pure
    noise near chance.
    """
    n = 200
    y = (rng.uniform(size=n) < 0.3).astype(int)
    folds = np.arange(n) % 5
    R = rng.uniform(size=(n, 3))                    # arms carrying zero signal
    s = cross_fitted_stack(R, y, folds)
    from sklearn.metrics import roc_auc_score
    assert 0.35 < roc_auc_score(y, s) < 0.65, "cross-fitting is leaking"


def test_cross_fitted_stacker_still_finds_real_signal():
    """...and it must not be so conservative that it misses a genuine arm."""
    from sklearn.metrics import roc_auc_score
    n = 300
    y = (rng.uniform(size=n) < 0.4).astype(int)
    folds = np.arange(n) % 5
    good = y + rng.normal(0, 0.6, n)                # informative
    noise = rng.uniform(size=n)
    R = np.column_stack([good, noise])
    s = cross_fitted_stack(R, y, folds)
    assert roc_auc_score(y, s) > 0.75


def test_single_class_training_fold_falls_back_rather_than_emitting_a_constant():
    y = np.array([0] * 20 + [1] * 20)
    folds = np.array([0] * 20 + [1] * 20)           # every fold is single-class
    R = rng.uniform(size=(40, 2))
    s = cross_fitted_stack(R, y, folds)
    assert len(np.unique(s)) > 1, "must not collapse to a constant"


def test_soft_vote_weights_are_normalised():
    R = np.array([[0.0, 1.0], [1.0, 0.0]])
    a = soft_vote(R, [1, 1])
    b = soft_vote(R, [10, 10])
    assert a == pytest.approx(b), "scaling all weights must change nothing"


def test_soft_vote_respects_asymmetric_weights():
    R = np.array([[0.0, 1.0], [1.0, 0.0]])
    heavy_first = soft_vote(R, [9, 1])
    assert heavy_first[1] > heavy_first[0]


def test_hard_vote_is_coarse_by_construction():
    """Documents why the published pipelines land on soft voting instead."""
    R = rng.uniform(size=(100, 3))
    v = hard_vote(R)
    assert set(np.unique(v)) <= {0.0, 1.0, 2.0, 3.0}


def test_align_keeps_only_patients_scored_by_every_arm():
    y = np.array([1, 0, 1, 0])
    folds = np.array([0, 0, 1, 1])
    a = _arm(["P0", "P1", "P2", "P3"], [0.9, 0.1, 0.8, 0.2], y, folds)
    b = _arm(["P0", "P1", "P2"], [0.7, 0.2, 0.6], y[:3], folds[:3])
    ids, yy, ff, R = align({"a": a, "b": b})
    assert ids == ["P0", "P1", "P2"]
    assert R.shape == (3, 2)


def test_ranks_are_computed_within_fold():
    """Folds are separately fitted models, so raw scores are not comparable."""
    recs = _arm(["P0", "P1", "P2", "P3"], [0.9, 0.8, 0.2, 0.1],
                [1, 0, 1, 0], [0, 0, 1, 1])
    r = rank_within_fold(recs, "egfr_prob")
    # Within each fold the pair spans the full 0..1 range.
    assert r["P1"] == pytest.approx(0.0) and r["P0"] == pytest.approx(1.0)
    assert r["P3"] == pytest.approx(0.0) and r["P2"] == pytest.approx(1.0)


def test_holm_is_monotone_and_never_below_the_raw_p():
    truth = {f"P{i}": int(i % 3 == 0) for i in range(120)}
    ids = list(truth)
    base = {i: rng.uniform() for i in ids}
    arms = {"base": base}
    for k in range(5):
        arms[f"arm{k}"] = {i: rng.uniform() for i in ids}
    out = holm_paired(arms, truth, "base", n_boot=400)

    for v in out.values():
        assert v["p_holm"] >= v["p_raw"] - 1e-12, "adjusted p must not be smaller"
        assert v["p_holm"] <= 1.0
    ordered = sorted(out.values(), key=lambda v: v["p_raw"])
    holms = [v["p_holm"] for v in ordered]
    assert holms == sorted(holms), "Holm must be monotone in the raw p order"


def test_holm_excludes_the_baseline_from_the_comparisons():
    truth = {f"P{i}": int(i % 2) for i in range(60)}
    arms = {"base": {i: rng.uniform() for i in truth},
            "other": {i: rng.uniform() for i in truth}}
    out = holm_paired(arms, truth, "base", n_boot=200)
    assert set(out) == {"other"}


def test_holm_penalises_a_wide_sweep_more_than_a_narrow_one():
    """The whole point: scoring 10 arms and reporting the best needs a correction."""
    truth = {f"P{i}": int(i % 3 == 0) for i in range(150)}
    ids = list(truth)
    base = {i: rng.uniform() for i in ids}
    # One arm with a mild real edge, plus a variable number of null arms.
    edge = {i: rng.uniform() + 0.35 * truth[i] for i in ids}

    narrow = holm_paired({"base": base, "edge": edge}, truth, "base", n_boot=800)
    wide_arms = {"base": base, "edge": edge}
    for k in range(9):
        wide_arms[f"null{k}"] = {i: rng.uniform() for i in ids}
    wide = holm_paired(wide_arms, truth, "base", n_boot=800)

    assert wide["edge"]["p_holm"] >= narrow["edge"]["p_holm"]


# --- provenance ------------------------------------------------------------

def test_fusion_was_built_from_the_canonical_run_set():
    """The published fusion table was once computed on `peri-s42.json`.

    That is the PERITUMORAL VARIANT, not the headline model, because
    fusion.py's --theia default pointed at it. The symptom in the manuscript was
    a row printing the headline AUC 0.618 beside a difference of -0.108, which is
    0.656 - 0.764 -- the variant's number. The conclusion survived, but only by
    luck. This asserts the artifact on disk came from CANONICAL.json.
    """
    import json
    import os
    import pytest

    if not os.path.exists("results/fusion.json"):
        pytest.skip("fusion.json not generated")
    d = json.load(open("results/fusion.json"))
    canonical = {os.path.basename(p)
                 for p in json.load(open("results/CANONICAL.json"))["headline_runs"]}
    assert set(d.get("runs", [])) == canonical, (
        f"fusion.json was built from {d.get('runs')}, "
        f"canonical is {sorted(canonical)}")
    assert d.get("from_canonical") is True

    # The THEIA arm must equal the published headline, or the table is again
    # pairing an AUC from one estimand with a difference from another.
    head = json.load(open("results/CANONICAL.json"))["headline"]
    got = d["aggregate"]["THEIA"]["auc_mean"]
    assert abs(got - head["mean"]) < 0.001, (
        f"fusion THEIA arm {got:.3f} != headline {head['mean']:.3f}")
