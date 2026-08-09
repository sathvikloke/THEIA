"""Label mapping and CT/SEG pairing — the two places a silent bug flips the science.

`label_of` turns a spreadsheet cell into the target. A wrong mapping inverts the
outcome for a subset of patients, which does not raise, does not look odd in any
summary, and simply makes every AUC wrong.

`choose_ct` decides which CT series a segmentation belongs to. Pairing a mask
with the wrong series silently trains the model to attend to anatomy that is not
there.
"""
import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# label_of
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    ("Mutant", 1), ("mutant", 1), ("  MUTANT  ", 1), ("Positive", 1), ("yes", 1), ("1", 1),
    ("Wildtype", 0), ("wild type", 0), ("WT", 0), ("Negative", 0), ("no", 0), ("0", 0),
    ("Unknown", -1), ("Not collected", -1), ("", -1), ("n/a", -1), ("pending", -1),
])
def test_label_of_maps_every_spelling_the_sheet_actually_uses(value, expected):
    from theia.data.preprocess import label_of

    row = pd.Series({"EGFR mutation status": value})
    assert label_of(row, "EGFR") == expected, f"{value!r} mapped wrongly"


def test_label_of_never_guesses_from_an_unrecognised_string():
    """An unknown value must become -1, never 0.

    Mapping it to 0 would fold "we did not test this patient" into
    "this patient is wild-type" -- inventing negatives, inflating the negative
    class, and biasing every AUC. -1 is masked out downstream.
    """
    from theia.data.preprocess import label_of

    for junk in ("equivocal", "failed QC", "see note", "MUTANT?", "0.5"):
        assert label_of(pd.Series({"EGFR mutation status": junk}), "EGFR") == -1, junk


def test_label_of_reads_the_real_tcia_header_spelling():
    """The sheet uses space-separated headers, not underscores."""
    from theia.data.preprocess import label_of

    assert label_of(pd.Series({"EGFR mutation status": "Mutant"}), "EGFR") == 1
    assert label_of(pd.Series({"KRAS mutation status": "Wildtype"}), "KRAS") == 0
    # A gene absent from the sheet is unknown, not negative.
    assert label_of(pd.Series({"EGFR mutation status": "Mutant"}), "ALK") == -1


def test_label_of_matches_the_real_cohort_counts():
    """Guard the actual distribution, so a mapping regression shows up as a count.

    The published NSCLC-Radiogenomics clinical sheet has 43 EGFR-mutant and 129
    wild-type among the calls that are neither Unknown nor Not-collected.
    """
    from theia.data.preprocess import label_of

    sheet = (["Mutant"] * 43 + ["Wildtype"] * 129 + ["Unknown"] * 34
             + ["Not collected"] * 5)
    labels = [label_of(pd.Series({"EGFR mutation status": v}), "EGFR") for v in sheet]
    assert labels.count(1) == 43
    assert labels.count(0) == 129
    assert labels.count(-1) == 39


# --------------------------------------------------------------------------
# choose_ct
# --------------------------------------------------------------------------
def _ser(uid, n=100, axial=True, localizer=False, derived=False, desc=""):
    from theia.data.import_nbia import SeriesInfo

    return SeriesInfo(path=f"/{uid}", patient_id="P1", series_uid=uid, modality="CT",
                      description=desc, n_files=n, axial=axial, localizer=localizer,
                      derived=derived)


def test_choose_ct_prefers_the_series_the_seg_references():
    """The SEG names its source series; that reference must win over any heuristic.

    Falling back to "the largest series" would silently pair the mask with a
    different reconstruction of the same study, so the tumor would sit somewhere
    the image does not show it. On this cohort the reference resolved 420 of 421
    patients, so a heuristic would be wrong far more often than it is needed.
    """
    from theia.data.import_nbia import choose_ct

    got, why = choose_ct([_ser("WANTED", n=80), _ser("OTHER", n=400)], referenced="WANTED")
    assert got.series_uid == "WANTED", f"reference ignored: {why}"
    assert "referenced" in why.lower()


def test_choose_ct_rejects_non_axial_and_localizer_series():
    """A coronal reformat or a scout has the wrong geometry to carry a mask."""
    from theia.data.import_nbia import choose_ct

    cands = [_ser("CORONAL", n=500, axial=False),
             _ser("SCOUT", n=300, localizer=True),
             _ser("AXIAL", n=100)]
    got, why = choose_ct(cands, referenced=None)
    assert got.series_uid == "AXIAL", f"picked {got.series_uid} ({why})"


def test_choose_ct_deprioritises_fusion_and_derived_series():
    """Without a reference, a PET/CT fusion must not win on slice count alone."""
    from theia.data.import_nbia import choose_ct

    cands = [_ser("FUSION", n=900, desc="PET CT Fusion"),
             _ser("DERIVED", n=600, derived=True),
             _ser("PLAIN", n=200, desc="Chest CT")]
    got, _ = choose_ct(cands, referenced=None)
    assert got.series_uid == "PLAIN"


def test_choose_ct_reports_why_it_failed_rather_than_returning_nothing_silently():
    """A patient with only reformats must produce an explanation, not None alone."""
    from theia.data.import_nbia import choose_ct

    got, why = choose_ct([_ser("CORONAL", n=500, axial=False)], referenced=None)
    assert got is None
    assert "non-axial" in why, why


def test_choose_ct_falls_back_when_the_reference_names_a_missing_series():
    """A dangling SEG reference must not drop the patient.

    It must fall through to the heuristic AND say the reference was unusable, so
    the import log distinguishes "resolved by reference" from "guessed".
    """
    from theia.data.import_nbia import choose_ct

    got, why = choose_ct([_ser("PRESENT", n=200)], referenced="ABSENT")
    assert got.series_uid == "PRESENT"
    assert "reference unavailable" in why.lower(), why
