"""Tests for the field-of-view, subgroup and external-grounding analyses."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from theia.analysis.external_grounding import centre_prior, summarise
from theia.analysis.semantic_oracle import (IN_CROP_LABELS, WHOLE_LUNG_LABELS,
                                            build_matrix, parse_labelled)
from theia.analysis.subgroups import auc_on, rank_within_fold

AIM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ImageAnnotationCollection xmlns="gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM">
  <imageAnnotations>
    <ImageAnnotation>
      <imagingObservationCharacteristicCollection>
        <ImagingObservationCharacteristic>
          <label value="Lung Parencyma Features"/>
          <typeCode code="RID1"><iso:displayName xmlns:iso="uri:iso" value="airway abnormality"/></typeCode>
        </ImagingObservationCharacteristic>
        <ImagingObservationCharacteristic>
          <label value="Lung Parencyma Features"/>
          <typeCode code="RID2"><iso:displayName xmlns:iso="uri:iso" value="bronchial wall thickening"/></typeCode>
        </ImagingObservationCharacteristic>
        <ImagingObservationCharacteristic>
          <label value="Nodule Associated Findings"/>
          <typeCode code="RID3"><iso:displayName xmlns:iso="uri:iso" value="attachment to pleura"/></typeCode>
        </ImagingObservationCharacteristic>
        <ImagingObservationCharacteristic>
          <label value="Emphysema"/>
          <typeCode code="RID4"><iso:displayName xmlns:iso="uri:iso" value="present"/></typeCode>
        </ImagingObservationCharacteristic>
      </imagingObservationCharacteristicCollection>
    </ImageAnnotation>
  </imageAnnotations>
</ImageAnnotationCollection>
"""


def test_repeatable_labels_keep_their_provenance(tmp_path):
    """The whole point of re-parsing: which label a finding came from.

    `aim.parse_aim` flattens both repeatable labels into one `findings` list, so
    "airway abnormality" (whole-lung) and "attachment to pleura" (in-crop) become
    indistinguishable. The FOV experiment is built entirely on telling them apart,
    so a regression here would silently mis-assign features to the wrong anatomy
    and the ablation would measure nothing.
    """
    p = tmp_path / "PAT-1.xml"
    p.write_text(AIM_XML)
    got = parse_labelled(str(p))

    assert set(got["Lung Parencyma Features"]) == {"airway abnormality",
                                                   "bronchial wall thickening"}
    assert got["Nodule Associated Findings"] == ["attachment to pleura"]
    assert got["Emphysema"] == ["present"]
    # And the two live on opposite sides of the crop boundary.
    assert "Lung Parencyma Features" in WHOLE_LUNG_LABELS
    assert "Nodule Associated Findings" in IN_CROP_LABELS


def test_label_groups_are_disjoint():
    assert not (WHOLE_LUNG_LABELS & IN_CROP_LABELS)


def test_patients_without_an_aim_read_are_kept_not_dropped(tmp_path):
    """Dropping them would change the cohort between arms.

    Every AUC in this project is compared against another AUC on the same
    patients; an arm that silently scored a different 140 of 158 would not be
    comparable to the headline.
    """
    csv = tmp_path / "clinical.csv"
    csv.write_text("Case ID,%GG\nP1,25\nP2,0\nP3,50\n")
    rows = [{"patient_id": "P1"}, {"patient_id": "P2"}, {"patient_id": "P3"}]
    sem = {"P1": {"Emphysema": ["present"]}}          # P2, P3 have no AIM file

    X, names, groups = build_matrix(rows, sem, str(csv))

    assert X.shape[0] == 3, "every row must survive"
    assert not np.isnan(X).any()
    flag = names.index("__no_aim")
    assert X[0, flag] == 0.0 and X[1, flag] == 1.0 and X[2, flag] == 1.0
    assert len(names) == len(groups) == X.shape[1]
    assert set(groups) <= {"in_crop", "whole_lung"}


def test_subgroup_ranks_are_recomputed_not_inherited():
    """A rank is relative to the patients being compared.

    Carrying full-cohort ranks into a subgroup would import information about the
    patients that were just excluded -- which is exactly the leak the
    adenocarcinoma-only analysis exists to avoid.
    """
    recs = [{"patient_id": f"P{i}", "fold": 0, "egfr_prob": p, "egfr_true": t}
            for i, (p, t) in enumerate([(0.1, 0), (0.2, 0), (0.9, 1), (0.95, 1)])]

    full = rank_within_fold(recs, "egfr_prob")
    assert full["P0"] == pytest.approx(0.0)
    assert full["P3"] == pytest.approx(1.0)

    sub = rank_within_fold([r for r in recs if r["patient_id"] in {"P1", "P2"}],
                           "egfr_prob")
    # Within the subgroup these two span the whole range, not 1/3..2/3.
    assert sub["P1"] == pytest.approx(0.0)
    assert sub["P2"] == pytest.approx(1.0)


def test_auc_on_subset_uses_only_that_subset():
    recs = [{"patient_id": f"P{i}", "fold": 0, "egfr_prob": p, "egfr_true": t}
            for i, (p, t) in enumerate([(0.9, 1), (0.8, 0), (0.7, 1), (0.1, 0)])]
    auc, n, npos = auc_on(recs, {"P0", "P1", "P2", "P3"})
    assert n == 4 and npos == 2
    auc2, n2, npos2 = auc_on(recs, {"P2", "P3"})
    assert n2 == 2 and npos2 == 1
    assert auc2 == pytest.approx(1.0)          # 0.7 > 0.1 and P2 is the positive


def test_single_class_subset_returns_nan_rather_than_a_number():
    recs = [{"patient_id": "P0", "fold": 0, "egfr_prob": 0.9, "egfr_true": 1},
            {"patient_id": "P1", "fold": 0, "egfr_prob": 0.1, "egfr_true": 1}]
    auc, n, npos = auc_on(recs, {"P0", "P1"})
    assert np.isnan(auc) and n == 2 and npos == 2


@pytest.mark.parametrize("roi_frac", [0.02, 0.06, 0.20])
def test_centre_prior_pointing_and_iou_are_width_invariant(roi_frac):
    """Checks a claim the module's docstring makes.

    The centre prior is only a fair control if its pointing and area-matched IoU
    do not depend on the sigma chosen -- otherwise the control could be tuned,
    consciously or not, until the trained model beat it. Both metrics rank cells
    and then threshold at top-k, so any strictly decreasing function of distance
    from the centre must give identical answers. Mass is the one column that
    legitimately moves with width.
    """
    h = w = 14
    roi = torch.zeros(3, 2, 1, 32, 32)          # [B,S,1,H,W]
    roi[:, :, :, 12:20, 12:20] = 1.0                       # centred lesion
    shape = (3, 4, h, w)

    outs = []
    for frac in (roi_frac, roi_frac * 4):               # two very different widths
        cp = centre_prior(shape, frac)
        outs.append(summarise([cp], [roi], seed=0))

    assert outs[0]["grounding_pointing"] == pytest.approx(outs[1]["grounding_pointing"])
    assert outs[0]["grounding_iou"] == pytest.approx(outs[1]["grounding_iou"])


def test_centre_prior_actually_points_at_a_centred_lesion():
    """Sanity: if this control cannot find a centred box, it is not a control."""
    roi = torch.zeros(2, 2, 1, 32, 32)          # [B,S,1,H,W]
    roi[:, :, :, 12:20, 12:20] = 1.0
    cp = centre_prior((2, 4, 14, 14), 0.06)
    out = summarise([cp], [roi], seed=0)
    assert out["grounding_pointing"] == pytest.approx(1.0)
    assert out["grounding_mass"] > out["grounding_mass_shuffled"]


def test_offcentre_lesion_defeats_the_centre_prior():
    """And if the lesion is in a corner the prior must fail, or it measures nothing."""
    roi = torch.zeros(2, 2, 1, 32, 32)          # [B,S,1,H,W]
    roi[:, :, :, 0:6, 0:6] = 1.0
    cp = centre_prior((2, 4, 14, 14), 0.06)
    out = summarise([cp], [roi], seed=0)
    assert out["grounding_pointing"] == pytest.approx(0.0)


def test_external_grounding_figure_renders_from_the_real_archive(tmp_path):
    """Smoke test for the figure attached to the collaboration email.

    Not a check on how it looks -- that is not testable -- but on the two things
    that would make it wrong rather than ugly: that it reads the archive this
    project actually produced, and that it renders every control. A figure that
    silently dropped the centre-prior bar would overstate the result to the one
    audience most able to check it.
    """
    import json
    import os

    from theia.analysis.figures import fig_external_grounding

    src = "results/external_grounding_canonical.json"
    if not os.path.exists(src):
        pytest.skip("external grounding has not been run")
    blob = json.load(open(src))

    for key in ("rows", "control_centre_prior", "control_random_init",
                "mean_mass_lift", "folds_beating_shuffle", "n_folds", "n_external"):
        assert key in blob, f"{src} is missing {key}, which the figure needs"

    out = tmp_path / "fig6.png"
    fig_external_grounding(blob, str(out))
    assert out.exists() and out.stat().st_size > 5000


def test_external_grounding_controls_are_far_below_the_trained_model():
    """The claim the figure makes, asserted as numbers.

    If a randomly-initialised head ever produced a non-trivial lift, the metric
    would be measuring architecture rather than learning and the whole primary
    endpoint would be void.
    """
    import json
    import os

    src = "results/external_grounding_canonical.json"
    if not os.path.exists(src):
        pytest.skip("external grounding has not been run")
    b = json.load(open(src))
    trained_pointing = np.mean([r["grounding_pointing"] for r in b["rows"]])

    assert abs(b["control_random_init"]["grounding_mass_lift"]) < 0.10
    assert b["control_random_init"]["grounding_peak_ratio"] < 1.5, \
        "a random head should produce a near-uniform map"
    assert trained_pointing > b["control_centre_prior"]["grounding_pointing"], \
        "trained model must beat 'look at the middle' on pointing"
