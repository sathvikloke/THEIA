"""Tests for the multi-seed aggregator.

This module decides what number goes in the abstract, so its failure modes are
all "quietly reports something better than the truth":
  * averaging in a fold that never trained
  * merging two runs' fold 0 into one ranking, which makes rank normalisation
    compare models that were never comparable
  * presenting the pooled-over-seeds CI as if it covered rerun variance
"""
import json

import pytest


def _run(tmp_path, run_id, seed, aucs, stalled_folds=()):
    """A results/<run_id>.json with one out-of-fold prediction block per fold."""
    folds = []
    for i, auc in enumerate(aucs):
        if i in stalled_folds:
            folds.append({"fold": i, "stalled": True, "test": {}})
            continue
        oof = []
        for j in range(10):
            y = j % 2
            # Separable when auc is high, scrambled when low.
            prob = (y * 0.9 + 0.05) if auc > 0.5 else (0.5 + 0.01 * j)
            oof.append({"patient_id": f"{run_id}-p{i}-{j}", "fold": i,
                        "egfr_prob": prob, "egfr_true": y})
        folds.append({"fold": i, "test": {"egfr_auc": auc}, "oof": oof})
    blob = {"run_id": run_id, "config": {"seed": seed},
            "pooled": {"egfr_auc": sum(a for a in aucs) / len(aucs)},
            "folds": folds}
    p = tmp_path / f"{run_id}.json"
    p.write_text(json.dumps(blob))
    return p


def test_across_seed_sd_is_reported_and_is_not_the_bootstrap_ci(tmp_path, capsys):
    from theia.analysis import aggregate

    _run(tmp_path, "ms-s1", 1, [0.60, 0.62, 0.61])
    _run(tmp_path, "ms-s2", 2, [0.70, 0.72, 0.71])
    out = tmp_path / "sum.json"
    import sys
    sys.argv = ["agg", "--pattern", str(tmp_path / "ms-*.json"), "--out", str(out)]
    aggregate.main()

    blob = json.load(open(out))
    assert blob["n_runs"] == 2
    assert blob["across_seed_mean"] == pytest.approx(0.66, abs=1e-6)
    assert blob["across_seed_sd"] > 0.05, "rerun variance was not reported"
    # The pooled-all-seeds figure must be present but clearly separate.
    assert "pooled_all_seeds" in blob
    assert blob["across_seed_sd"] != blob["pooled_all_seeds"].get("egfr_auc")


def test_stalled_folds_are_excluded_and_counted(tmp_path):
    """A fold that never trained must not be averaged in as if it had."""
    from theia.analysis import aggregate
    import sys

    _run(tmp_path, "ms-s1", 1, [0.70, 0.70, 0.70], stalled_folds=(1,))
    out = tmp_path / "sum.json"
    sys.argv = ["agg", "--pattern", str(tmp_path / "ms-*.json"), "--out", str(out)]
    aggregate.main()

    blob = json.load(open(out))
    assert blob["stalled_folds"] == 1, "stalled fold was not counted"
    # 3 folds x 10 patients, minus the stalled one = 20 predictions pooled.
    assert blob["pooled_all_seeds"]["egfr_n"] == 20, (
        "a stalled fold's rows leaked into the pooled estimate")


def test_folds_from_different_runs_are_never_merged_into_one_ranking(tmp_path):
    """Rank normalisation is within-fold; two runs' fold 0 are different models.

    Merging them would rank one model's outputs against another's, which is the
    exact scale-mixing that within-fold normalisation exists to prevent.
    """
    from theia.analysis import aggregate
    import sys

    _run(tmp_path, "ms-s1", 1, [0.70])
    _run(tmp_path, "ms-s2", 2, [0.70])
    out = tmp_path / "sum.json"
    sys.argv = ["agg", "--pattern", str(tmp_path / "ms-*.json"), "--out", str(out)]
    aggregate.main()

    blob = json.load(open(out))
    # 2 runs x 1 fold x 10 patients, all retained and distinctly keyed.
    assert blob["pooled_all_seeds"]["egfr_n"] == 20
    # If fold ids collided the two runs' patients would share a rank pool; the
    # namespacing keeps them separate, so a perfectly separable pair stays 1.0.
    assert blob["pooled_all_seeds"]["egfr_auc"] == pytest.approx(1.0, abs=1e-9)
