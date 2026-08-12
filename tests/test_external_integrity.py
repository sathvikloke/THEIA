"""The external grounding result is the paper's primary endpoint. These guard it.

A held-out claim is worth exactly as much as the guarantee that the model never
saw the data. THEIA has a pretraining path (`model.grounding_pretrain_ckpt`,
`theia.engine.pretrain`) whose corpus IS NSCLC-Radiomics -- the same collection
used as the external cohort. If any evaluated checkpoint had been warm-started
from it, "420 held-out patients" would be false and the primary endpoint would be
void. Nothing in the training code prevents that combination, so it is asserted
here instead.
"""
from __future__ import annotations

import json
import os

import pytest

EXTERNAL = "results/external_grounding.json"
EXTERNAL_ROWS = "data/processed_pretrain/rows.jsonl"
INTERNAL_ROWS = "data/processed/rows.jsonl"


def _ids(path):
    return {json.loads(line)["patient_id"] for line in open(path)}


@pytest.mark.skipif(not os.path.exists(EXTERNAL), reason="external grounding not run")
def test_no_evaluated_run_was_warm_started_from_the_external_collection():
    """The one that would void the primary endpoint."""
    runs = {r["run"] for r in json.load(open(EXTERNAL))["rows"]}
    assert runs, "external result names no runs"
    for run in sorted(runs):
        path = f"results/{run}.json"
        if not os.path.exists(path):
            # run16-nogen-s1337 is archived under results/ms-s1337.json
            path = "results/ms-s1337.json" if run == "run16-nogen-s1337" else path
        if not os.path.exists(path):
            pytest.skip(f"no archived config for {run}")
        model = json.load(open(path))["config"]["model"]
        assert not model.get("grounding_pretrain_ckpt"), (
            f"{run} was warm-started from {model['grounding_pretrain_ckpt']}; "
            "if that checkpoint came from NSCLC-Radiomics the external result is void")
        assert not model.get("warm_start_vision"), (
            f"{run} warm-started its vision encoder, which is the other route to "
            "contamination")


@pytest.mark.skipif(not (os.path.exists(EXTERNAL_ROWS) and os.path.exists(INTERNAL_ROWS)),
                    reason="cohorts not preprocessed")
def test_the_two_cohorts_share_no_patients():
    internal, external = _ids(INTERNAL_ROWS), _ids(EXTERNAL_ROWS)
    overlap = internal & external
    assert not overlap, f"{len(overlap)} patients appear in both cohorts: {sorted(overlap)[:5]}"
    # And they are from visibly different collections, so an id-format change
    # cannot make a real overlap look empty.
    assert all(i.startswith(("AMC-", "R01-")) for i in internal)
    assert all(i.startswith("LUNG1-") for i in external)


@pytest.mark.skipif(not os.path.exists(EXTERNAL), reason="external grounding not run")
def test_the_external_evaluation_covers_the_whole_cohort():
    """A quietly truncated evaluation would inflate the result without looking wrong."""
    blob = json.load(open(EXTERNAL))
    assert blob["n_external"] >= 420, (
        f"only {blob['n_external']} external patients evaluated; a --limit was left on")
    assert blob["n_folds"] >= 15, f"only {blob['n_folds']} checkpoints evaluated"


@pytest.mark.skipif(not os.path.exists(EXTERNAL), reason="external grounding not run")
def test_gate_d_is_recomputed_not_trusted():
    """Recompute the pre-specified gate from the raw rows rather than reading the flag."""
    import numpy as np
    b = json.load(open(EXTERNAL))
    rows, cp, rnd = b["rows"], b["control_centre_prior"], b["control_random_init"]

    lift = float(np.mean([r["grounding_mass_lift"] for r in rows]))
    beat = sum(1 for r in rows if r["grounding_mass"] > r["grounding_mass_shuffled"])
    pointing = float(np.mean([r["grounding_pointing"] for r in rows]))

    assert lift >= 0.20, f"mass lift {lift:.3f} below the pre-specified 0.20"
    assert beat / len(rows) >= 10 / 14, f"beats shuffle in only {beat}/{len(rows)}"
    assert abs(rnd["grounding_mass_lift"]) < 0.10, "random-init head has a real lift"
    assert pointing > cp["grounding_pointing"], "does not beat the centre prior"
    assert b["gate_d_passed"] is True, "stored verdict disagrees with the recomputation"


def test_monitoring_an_untrained_term_is_rejected():
    """A monitor may not select on a loss the config is not optimising.

    This fired for real: base-s1337-rerun was trained with gen_loss in its monitor
    while loss_weights.gen was 0.0, and the other two canonical runs were not, so
    the three "seeds" of the headline differed by more than the seed.
    """
    import pytest as _pytest

    from theia.config import load_config

    with _pytest.raises(Exception) as exc:
        load_config("configs/default.yaml", {
            "train.loss_weights.gen": 0.0,
            "train.monitor": [["egfr_auc", 1.0], ["gen_loss", -0.01]],
        })
    assert "gen_loss" in str(exc.value)


def test_the_shipped_default_does_not_monitor_an_untrained_term():
    from theia.config import load_config
    cfg = load_config("configs/default.yaml")
    keys = [m if isinstance(m, str) else m[0] for m in cfg.train.monitor]
    if float(cfg.train.loss_weights.gen) == 0.0:
        assert "gen_loss" not in keys


def test_the_shipped_default_reproduces_the_canonical_headline_config():
    """configs/default.yaml must match what the reported runs actually used.

    ANALYSIS_PLAN section 3 fixes "training config: configs/default.yaml at the
    commit that seals the model". That is meaningless if the default produces a
    model no reported number corresponds to -- which it did, with
    peritumoral_features true and gen 1.0 against canonical runs using false
    and 0.0.
    """
    from theia.config import load_config
    cfg = load_config("configs/default.yaml")
    canon = json.load(open("results/CANONICAL.json"))["headline_runs"]
    for path in canon:
        if not os.path.exists(path):
            continue
        run = json.load(open(path))["config"]
        assert bool(cfg.model.peritumoral_features) == bool(
            run["model"].get("peritumoral_features", False)), (
            f"default peritumoral_features disagrees with {path}")
        assert float(cfg.train.loss_weights.gen) == float(
            run["train"]["loss_weights"]["gen"]), (
            f"default gen weight disagrees with {path}")
