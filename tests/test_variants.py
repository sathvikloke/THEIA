"""Tests for EGFR variant classification and external label harmonisation.

These run offline. The network-facing `fetch_egfr` is not exercised here; the
classification rules it feeds are, because those are what decide whether a
patient lands in the positive class.
"""
from __future__ import annotations

import pytest

from theia.data.variants import classify, patient_labels, summarise


@pytest.mark.parametrize("change", ["L858R", "L861Q", "G719A", "G719C", "G719S", "S768I"])
def test_canonical_point_mutations_are_activating(change):
    assert classify(change) == "ACTIVATING"


@pytest.mark.parametrize("change", [
    "E746_A750del", "E746_S752delinsV", "L747_T751del", "L747_E749del",
    "L747_P753delinsS", "E746_T751delinsIP",
])
def test_exon19_deletions_are_activating(change):
    """The single largest positive group after L858R, and rule-based rather than listed."""
    assert classify(change) == "ACTIVATING"


@pytest.mark.parametrize("change", ["T790M", "C797S"])
def test_resistance_mutations_are_not_counted_as_activating(change):
    assert classify(change) == "RESISTANCE"


@pytest.mark.parametrize("change,", [
    ("L62R",), ("R222L",), ("E545Q",), ("K479I",), ("H358R",), ("E84K",), ("I143L",),
])
def test_extracellular_variants_are_passengers(change):
    """Every one of these is really present in TCGA/CPTAC labelled 'EGFR mutant'.

    They sit below residue 645, outside the kinase domain, and are not what a
    thoracic oncologist means by EGFR-mutant. Counting them corrupts the positive
    class -- which at ~17 external positives is a patient or two labelled
    backwards.
    """
    assert classify(change[0]) == "PASSENGER"


def test_exon20_insertions_are_kept_separate_from_activating():
    """Oncogene-driven but not first/second-generation TKI responsive.

    Folding them into ACTIVATING would silently answer a clinical question the
    study has not asked, so they get their own class and an explicit decision.
    """
    assert classify("D770_N771insGL") == "EXON20INS"
    assert classify("V769_D770insASV") == "EXON20INS"


def test_kinase_domain_non_hotspots_are_uncertain_not_activating():
    """The module is biased toward UNCERTAIN on purpose.

    A wrongly-included passenger corrupts the positive class; a wrongly-excluded
    activating variant only shrinks it. A871G and L858M are real records from
    CPTAC that look superficially like hotspots and are not.
    """
    assert classify("A871G") == "UNCERTAIN"
    assert classify("L858M") == "UNCERTAIN"          # not L858R
    assert classify("V834L") == "UNCERTAIN"


def test_unparseable_and_empty_changes_do_not_crash_or_become_positive():
    for bad in ("", None, "?", "X210_splice"):
        assert classify(bad) in ("UNCERTAIN", "PASSENGER")


def test_a_patient_with_both_a_passenger_and_a_driver_is_positive():
    """max-aggregation over records, not first-wins."""
    recs = [{"patientId": "P1", "proteinChange": "R222L"},
            {"patientId": "P1", "proteinChange": "L858R"}]
    assert patient_labels(recs, "activating") == {"P1": 1}


def test_a_patient_with_only_passengers_is_negative_under_activating():
    recs = [{"patientId": "P2", "proteinChange": "R222L"},
            {"patientId": "P2", "proteinChange": "E545Q"}]
    assert patient_labels(recs, "activating") == {"P2": 0}
    # ...but positive under the definition that matches the internal binary field.
    assert patient_labels(recs, "any") == {"P2": 1}


def test_the_two_definitions_disagree_and_that_is_the_point():
    recs = [{"patientId": "A", "proteinChange": "L858R"},
            {"patientId": "B", "proteinChange": "L62R"},
            {"patientId": "C", "proteinChange": "T790M"}]
    assert sum(patient_labels(recs, "any").values()) == 3
    assert sum(patient_labels(recs, "activating").values()) == 1


def test_absence_from_the_mutation_table_is_not_recorded_as_wild_type():
    """A patient missing here may be wild-type or may be unsequenced.

    Silently emitting 0 would invent negatives, which is the same failure the
    pipeline already guards against with the -1 label.
    """
    recs = [{"patientId": "A", "proteinChange": "L858R"}]
    labels = patient_labels(recs, "any")
    assert "B" not in labels


def test_records_without_a_patient_id_are_skipped_not_keyed_to_none():
    recs = [{"proteinChange": "L858R"}, {"patientId": "A", "proteinChange": "L858R"}]
    labels = patient_labels(recs, "any")
    assert set(labels) == {"A"}


def test_summarise_counts_reconcile():
    recs = [{"patientId": "A", "proteinChange": "L858R"},
            {"patientId": "A", "proteinChange": "T790M"},
            {"patientId": "B", "proteinChange": "R222L"}]
    s = summarise(recs)
    assert s["n_records"] == 3 and s["n_patients"] == 2
    assert s["n_patients_any"] == 2
    assert s["n_patients_activating"] == 1
    assert sum(s["class_counts"].values()) == 3


def test_invalid_definition_is_rejected_loudly():
    with pytest.raises(ValueError):
        patient_labels([], "whatever")
