"""The demo must show real output, and must not overclaim.

The page it replaced was a hand-drawn CT with invented DICOM metadata and three
confident genotype calls. It was labelled schematic, but it read as a working
viewer -- and this project's whole finding is that a plausible-looking artifact is
not evidence anything works. These assert the replacement stays honest.
"""
from __future__ import annotations

import json
import os
import re

import pytest

DEMO = "demo/index.html"
CASES = "demo/cases.json"

pytestmark = pytest.mark.skipif(not os.path.exists(CASES),
                                reason="demo not built")


@pytest.fixture(scope="module")
def cases():
    return json.load(open(CASES))


def test_every_number_traces_to_the_archived_predictions(cases):
    """cases.json must quote results/*.json, not a re-run."""
    run = json.load(open(f"results/{cases['run']}"))
    archived = {r["patient_id"]: r for f in run["folds"] for r in f.get("oof", [])}
    for c in cases["cases"]:
        rec = archived.get(c["id"])
        assert rec, f"{c['id']} is not in {cases['run']}"
        assert abs(rec["egfr_prob"] - c["model_prob"]) < 5e-4, (
            f"{c['id']}: demo says {c['model_prob']}, archive says "
            f"{rec['egfr_prob']:.3f}")
        assert rec["egfr_true"] == c["egfr_true"]


def test_clinical_comparator_traces_to_baselines(cases):
    blob = json.load(open("results/baselines.json"))["oof"]["clinical"]
    clin = {r["patient_id"]: r["egfr_prob"] for r in blob}
    for c in cases["cases"]:
        if c["clinical_prob"] is None:
            continue
        assert abs(clin[c["id"]] - c["clinical_prob"]) < 5e-4


def test_cohort_strip_matches_the_manuscript(cases):
    co = cases["cohort"]
    assert co["model_auc"] == 0.618 and co["clinical_auc"] == 0.764
    assert co["smoking_auc"] == 0.794 and co["delta_auc"] == -0.029
    assert co["brier"] > co["brier_floor"], "Brier must exceed its floor"
    assert co["external_pointing"] > co["external_centre_prior"]


def test_a_failure_case_is_shown(cases):
    """A demo of this model that only shows hits misrepresents it."""
    misses = [c for c in cases["cases"]
              if (c["model_prob"] >= .5) != (c["egfr_true"] == 1)]
    assert misses, "at least one case the model gets wrong must be displayed"


def test_no_schematic_or_invented_metadata():
    html = open(DEMO).read()
    for bad in ("SCHEMATIC", "schematic data", "SER 2.1.4334", "SLICE 112/288"):
        assert bad not in html, f"invented metadata survived: {bad!r}"


def test_page_does_not_present_a_genotype_call():
    """Pooled AUC 0.618 with calibration slope 0.19 does not support a verdict."""
    html = open(DEMO).read()
    for bad in ("EGFR-mutant<", "CASE A EGFR", "KRAS-mutant<"):
        assert bad not in html, f"page asserts a genotype call: {bad!r}"
    assert "P(mutant)" in html, "probabilities must be shown as probabilities"


def test_licence_attribution_present():
    """NSCLC-Radiogenomics is CC BY 3.0: redistributable WITH attribution."""
    html = open(DEMO).read()
    assert "CC BY 3.0" in html and "Bakr" in html
    assert "NSCLC-Radiomics" not in html, (
        "NSCLC-Radiomics is CC BY-NC 3.0 and its pixels must not appear")


def test_assets_exist(cases):
    for c in cases["cases"]:
        for k in ("ct", "attn"):
            assert os.path.exists(os.path.join("demo", c[k])), c[k]
