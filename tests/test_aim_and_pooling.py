"""Tests for AIM annotation parsing and pooled out-of-fold evaluation.

Both exist because of defects found against the real NSCLC-RADIOGENOMICS data:

* `build_pseudo_report` read six columns that do not exist in the TCIA clinical
  spreadsheet, so every patient received an identical sentence and the generation
  head learned a constant while reporting an excellent loss.
* Averaging per-fold AUCs over ~23 EGFR positives produced an interval that
  includes chance. Pooling out-of-fold predictions is the correct estimator here.
"""
import json
import math

import numpy as np
import pytest
import torch

AIM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ImageAnnotationCollection xmlns="gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM">
 <imagingObservationEntityCollection>
  <ImagingObservationEntity>
   <imagingObservationCharacteristicCollection>
    <ImagingObservationCharacteristic>
     <typeCode code="RID34284" codeSystem="{margin}" codeSystemName="RadLex"/>
     <label value="Nodule Margins-Primary Pattern"/>
    </ImagingObservationCharacteristic>
    <ImagingObservationCharacteristic>
     <typeCode code="RID5799" codeSystem="{shape}" codeSystemName="RadLex"/>
     <label value="Nodule Shape"/>
    </ImagingObservationCharacteristic>
    <ImagingObservationCharacteristic>
     <typeCode code="RID5741" codeSystem="{atten}" codeSystemName="RadLex"/>
     <label value="Nodule Attenuation"/>
    </ImagingObservationCharacteristic>
    <ImagingObservationCharacteristic>
     <typeCode code="RID5828" codeSystem="peripheral" codeSystemName="RadLex"/>
     <label value="Axial Location"/>
    </ImagingObservationCharacteristic>
    <ImagingObservationCharacteristic>
     <typeCode code="RID46018" codeSystem="attachment to pleura" codeSystemName="RadLex"/>
     <label value="Nodule Associated Findings"/>
    </ImagingObservationCharacteristic>
   </imagingObservationCharacteristicCollection>
  </ImagingObservationEntity>
 </imagingObservationEntityCollection>
 <markupEntityCollection>
  <MarkupEntity>
   <imageReferenceUid root="1.2.3.{uid}"/>
   <referencedFrameNumber value="{frame}"/>
   <twoDimensionSpatialCoordinateCollection>
    <TwoDimensionSpatialCoordinate>
     <coordinateIndex value="0"/><x value="{cx}"/><y value="{cy}"/>
    </TwoDimensionSpatialCoordinate>
    <TwoDimensionSpatialCoordinate>
     <coordinateIndex value="1"/><x value="{cx2}"/><y value="{cy}"/>
    </TwoDimensionSpatialCoordinate>
   </twoDimensionSpatialCoordinateCollection>
  </MarkupEntity>
 </markupEntityCollection>
</ImageAnnotationCollection>"""


def _write_aim(d, pid, margin="spiculated", shape="round", atten="solid",
               frame=120, cx=372.9, cy=254.6, uid="9"):
    p = d / f"{pid}.xml"
    p.write_text(AIM_XML.format(margin=margin, shape=shape, atten=atten, frame=frame,
                                cx=cx, cy=cy, cx2=cx + 5, uid=uid))
    return p


def test_parses_semantics_and_markup(tmp_path):
    from theia.data.aim import parse_aim

    a = parse_aim(str(_write_aim(tmp_path, "AMC-001")))
    assert a.patient_id == "AMC-001"
    assert a.semantics["Nodule Margins-Primary Pattern"] == "spiculated"
    assert a.semantics["Nodule Shape"] == "round"
    assert "attachment to pleura" in a.findings
    assert a.has_location
    m = a.markups[0]
    assert m.frame == 120
    assert m.image_uid == "1.2.3.9"
    assert m.cx == pytest.approx(372.9)
    assert m.radius == pytest.approx(5.0)


def test_report_varies_with_the_annotation(tmp_path):
    """The bug this pins: one identical sentence for every patient."""
    from theia.data.aim import build_report, load_all

    _write_aim(tmp_path, "P1", margin="spiculated", shape="round", atten="solid")
    _write_aim(tmp_path, "P2", margin="smooth", shape="oval", atten="ground glass")
    _write_aim(tmp_path, "P3", margin="poorly defined", shape="complex", atten="solid")
    anns = load_all(str(tmp_path))
    reports = {p: build_report(a) for p, a in anns.items()}
    assert len(set(reports.values())) == 3, f"reports collapsed: {reports}"
    assert "spiculated round solid" in reports["P1"]
    assert "ground glass" in reports["P2"]


def test_report_grammar_and_term_normalization():
    from theia.data.aim import AimAnnotation, build_report, normalize_term, split_size

    assert normalize_term("lessOrEqual 5mm") == "less or equal 5mm"
    assert split_size("solid lessOrEqual 5mm".replace("lessOrEqual", "less or equal")) == (
        "solid", "5 mm or smaller")
    a = AimAnnotation("X", semantics={"Nodule Margins-Primary Pattern": "irregular",
                                      "Nodule Attenuation": "Solid lessOrEqual 5mm"})
    r = build_report(a)
    assert r.startswith("An irregular"), r            # article agrees with the vowel
    assert "5 mm or smaller" in r                      # size reads as a clause
    assert "lessOrEqual" not in r                      # no camelCase leaks to the LM
    assert r.endswith(".")


def test_report_omits_missing_fields_rather_than_defaulting():
    from theia.data.aim import AimAnnotation, build_report

    r = build_report(AimAnnotation("X", semantics={"Nodule Shape": "round"}))
    assert "round" in r
    for invented in ("ill-defined", "no pleural attachment", "no vascular convergence"):
        assert invented not in r, "report asserted a finding nobody recorded"


def test_missing_aim_file_is_skipped_not_fatal(tmp_path):
    from theia.data.aim import load_all

    _write_aim(tmp_path, "GOOD")
    (tmp_path / "BAD.xml").write_text("<not-xml")
    anns = load_all(str(tmp_path))
    assert set(anns) == {"GOOD"}


# --------------------------------------------------------------------------
# has_mask: AIM-located patients must not supervise grounding
# --------------------------------------------------------------------------
def test_zero_roi_contributes_nothing_to_grounding():
    from theia.engine.losses import grounding_loss

    attn = torch.rand(3, 8, 14, 14)
    roi = torch.zeros(3, 4, 1, 224, 224)          # AIM-located: no mask exists
    loss = grounding_loss(attn, roi, "cpu")
    assert float(loss) == 0.0, "an invented empty mask leaked into the grounding loss"


def test_mixed_batch_grounds_only_the_masked_patients():
    from theia.engine.losses import grounding_loss

    roi = torch.zeros(2, 4, 1, 224, 224)
    roi[0, :, :, 80:150, 80:150] = 1.0            # patient 0 has a real mask
    attn = torch.rand(2, 8, 14, 14)
    mixed = float(grounding_loss(attn, roi, "cpu"))
    alone = float(grounding_loss(attn[:1], roi[:1], "cpu"))
    assert mixed == pytest.approx(alone, abs=1e-5), (
        "the unmasked patient altered the grounding loss")


def test_collate_carries_has_mask(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset, collate

    rows = []
    for i, hm in enumerate([True, False]):
        npz = tmp_path / f"p{i}.npz"
        np.savez_compressed(npz, images=np.random.rand(4, 8, 8).astype(np.float32),
                            roi=np.zeros((4, 8, 8), dtype=np.float32))
        rows.append(dict(patient_id=f"p{i}", npz=str(npz), report="r",
                         labels={"EGFR": i}, n_slices=4, has_mask=hm))
    idx = tmp_path / "rows.jsonl"
    idx.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = RadiogenomicsDataset(str(idx), ["EGFR"])
    b = collate([ds[0], ds[1]], genes=["EGFR"])
    assert b["has_mask"].tolist() == [True, False]
    assert b["egfr"].tolist() == [0, 1], "has_mask was mistaken for a gene label"


def test_has_mask_defaults_true_for_legacy_rows(tmp_path):
    from theia.data.dataset import RadiogenomicsDataset

    npz = tmp_path / "p.npz"
    np.savez_compressed(npz, images=np.random.rand(2, 8, 8).astype(np.float32),
                        roi=np.zeros((2, 8, 8), dtype=np.float32))
    idx = tmp_path / "rows.jsonl"
    idx.write_text(json.dumps(dict(patient_id="p", npz=str(npz), report="r",
                                   labels={"EGFR": 1}, n_slices=2)) + "\n")
    assert bool(RadiogenomicsDataset(str(idx), ["EGFR"])[0]["has_mask"]) is True


# --------------------------------------------------------------------------
# pooled out-of-fold evaluation
# --------------------------------------------------------------------------
def _oof(n_pos, n_neg, auc_signal=1.2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos):
        rows.append({"patient_id": f"p{i}", "egfr_true": 1,
                     "egfr_prob": float(rng.normal(auc_signal, 1))})
    for i in range(n_neg):
        rows.append({"patient_id": f"n{i}", "egfr_true": 0,
                     "egfr_prob": float(rng.normal(0, 1))})
    return rows


def test_pooled_metrics_reports_one_auc_with_a_ci():
    from theia.engine.evaluate import pooled_metrics

    m = pooled_metrics(_oof(23, 94), ["EGFR"], bootstrap_n=200)
    assert m["egfr_n"] == 117 and m["egfr_n_pos"] == 23
    assert 0.0 <= m["egfr_auc"] <= 1.0
    assert m["egfr_auc_lo"] < m["egfr_auc"] < m["egfr_auc_hi"]


def test_pooling_is_tighter_than_per_fold():
    """The whole reason for the change: 5-positive folds cannot be measured."""
    from sklearn.metrics import roc_auc_score

    from theia.engine.evaluate import pooled_metrics

    rows = _oof(25, 95, seed=3)
    pooled = pooled_metrics(rows, ["EGFR"], bootstrap_n=400)
    pooled_w = pooled["egfr_auc_hi"] - pooled["egfr_auc_lo"]

    # same data, split into 5 folds, AUC computed per fold
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))
    per_fold = []
    for k in range(5):
        part = [rows[i] for i in order[k::5]]
        y = [r["egfr_true"] for r in part]
        if len(set(y)) < 2:
            continue
        per_fold.append(roc_auc_score(y, [r["egfr_prob"] for r in part]))
    assert len(per_fold) >= 3
    spread = max(per_fold) - min(per_fold)
    assert pooled_w < spread, (
        f"pooled CI ({pooled_w:.3f}) should be tighter than the per-fold "
        f"spread ({spread:.3f})")


def test_pooling_is_robust_to_per_fold_score_scales():
    """Each fold is a different model; their probability ranges differ.

    Concatenating raw probabilities lets a fold that outputs 0.45-0.55 be
    interleaved with one that outputs 0.05-0.95, corrupting the global ranking
    and dragging the pooled AUC below every individual fold. Rank-normalising
    within fold removes the artefact without changing any fold's ordering.
    """
    from sklearn.metrics import roc_auc_score

    from theia.engine.evaluate import _rank_normalize_by_fold

    rng = np.random.default_rng(0)
    rows, per_fold = [], []
    for f, (lo, hi) in enumerate([(0.45, 0.55), (0.05, 0.95), (0.30, 0.70)]):
        y = np.r_[np.ones(8), np.zeros(24)]
        raw = np.r_[rng.normal(1.2, 1, 8), rng.normal(0, 1, 24)]
        p = lo + (hi - lo) * (raw - raw.min()) / (np.ptp(raw) + 1e-9)
        per_fold.append(roc_auc_score(y, p))
        rows += [{"patient_id": f"f{f}_{i}", "fold": f, "egfr_true": int(y[i]),
                  "egfr_prob": float(p[i])} for i in range(len(y))]

    y = np.array([r["egfr_true"] for r in rows])
    raw_auc = roc_auc_score(y, [r["egfr_prob"] for r in rows])
    rank_auc = roc_auc_score(y, _rank_normalize_by_fold(rows, "egfr_prob"))
    assert rank_auc > raw_auc, "rank normalisation did not correct the scale artefact"
    assert rank_auc == pytest.approx(float(np.mean(per_fold)), abs=0.02), (
        "pooled estimate should agree with the per-fold mean when folds are equally sized")


def test_rank_normalization_preserves_within_fold_order():
    from theia.engine.evaluate import _rank_normalize_by_fold

    rows = [{"fold": 0, "p": 0.1}, {"fold": 0, "p": 0.9}, {"fold": 0, "p": 0.5},
            {"fold": 1, "p": 100.0}, {"fold": 1, "p": 200.0}]
    r = _rank_normalize_by_fold(rows, "p")
    assert r[0] < r[2] < r[1]          # fold 0 order intact
    assert r[3] < r[4]                 # fold 1 order intact
    assert set(np.round(r[3:], 6)) == {0.0, 1.0}


def test_pooled_metrics_handles_single_class_and_unknowns():
    from theia.engine.evaluate import pooled_metrics

    rows = [{"patient_id": "a", "egfr_true": 1, "egfr_prob": 0.9},
            {"patient_id": "b", "egfr_true": 1, "egfr_prob": 0.8},
            {"patient_id": "c", "egfr_true": -1, "egfr_prob": 0.5}]
    m = pooled_metrics(rows, ["EGFR"], bootstrap_n=50)
    assert m["egfr_n"] == 2, "unknown label was counted"
    assert math.isnan(m["egfr_auc"])


def test_config_accepts_aim_crop_mm():
    from theia.config import load_config

    cfg = load_config("configs/default.yaml")
    assert float(cfg.data.aim_crop_mm) > 0


def test_flat_attention_is_flagged_not_silently_scored():
    """Uniform attention must be visible as 'no localization', not a 0.000 score.

    Measured on the real untrained model: union values spanned 0.00510-0.00598
    (all ~1/196), the argmax landed on cell 0 for every patient, and pointing
    read exactly 0.000 -- which looks like a confident negative result rather
    than an attention map that says nothing.
    """
    from theia.engine.evaluate import grounding_metrics

    roi = _roi_flat_helper()
    flat = torch.full((4, 8, 14, 14), 1 / 196.0) + torch.rand(4, 8, 14, 14) * 1e-5
    peaked = torch.zeros(4, 8, 14, 14)
    peaked[:, :, 7, 7] = 1.0

    m_flat = grounding_metrics(flat, roi)
    m_peak = grounding_metrics(peaked, roi)
    assert np.mean(m_flat["grounding_peak_ratio"]) < 1.5, "uniform map not flagged as flat"
    assert np.mean(m_peak["grounding_peak_ratio"]) > 10, "peaked map not flagged as concentrated"


def _roi_flat_helper():
    roi = torch.zeros(4, 4, 1, 224, 224)
    roi[:, :, :, 90:150, 90:150] = 1.0
    return roi


def test_radlex_display_name_is_read_when_codesystem_is_absent(tmp_path):
    """Both AIM dialects in this collection must yield readable terms.

    NSCLC-Radiogenomics ships two shapes for the same fact:

        AMC-*: <typeCode code="RID5828" codeSystem="peripheral"/>
        R01-*: <typeCode code="RID5828"><iso:displayName value="peripheral"/></typeCode>

    Reading only codeSystem left 117 of 158 reports as raw ids -- "A rid5801
    rid5757 rid5741 lesion, rid5828" -- so 74% of the rationale supervision was
    unreadable codes and the generation head learned to emit them. Nothing
    raised; the text was simply wrong, and it only surfaced when a reader-study
    case set was inspected by eye.
    """
    from theia.data.aim import parse_aim

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<ImageAnnotationCollection xmlns="gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"
                           aimVersion="4.0">
 <imageAnnotations><ImageAnnotation>
  <imagingObservationEntityCollection><ImagingObservationEntity>
   <imagingObservationCharacteristicCollection>
    <ImagingObservationCharacteristic>
     <typeCode code="RID5828" codeSystemName="RadLex_3.9.1">
       <iso:displayName xmlns:iso="uri:iso.org:21090" value="peripheral"/>
     </typeCode>
     <label value="Axial Location"/>
    </ImagingObservationCharacteristic>
    <ImagingObservationCharacteristic>
     <typeCode code="RID46011" codeSystem="partially solid" codeSystemName="RadLex.3.10"/>
     <label value="Texture"/>
    </ImagingObservationCharacteristic>
   </imagingObservationCharacteristicCollection>
  </ImagingObservationEntity></imagingObservationEntityCollection>
 </ImageAnnotation></imageAnnotations>
</ImageAnnotationCollection>"""
    p = tmp_path / "a.xml"
    p.write_text(xml)

    ann = parse_aim(str(p))
    blob = repr(ann).lower()   # ann holds non-JSON types; repr is enough here
    assert "peripheral" in blob, "displayName term was not read"
    assert "partially solid" in blob, "codeSystem term was not read"
    assert "rid5828" not in blob, "raw RadLex id leaked into the annotation"
