"""Tiny geometry and refusal checks; no source data or model downloads."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from theia.data import lidc_rows as rows


@pytest.fixture
def fixture(tmp_path):
    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[2, :2, 6:] = 1  # deliberately off-centre, at an image boundary
    metadata = dict(
        schema=rows.SCHEMA, consensus_method=rows.CONSENSUS_METHOD,
        mask_axes="zyx", patient_id="LIDC-IDRI-0001",
        study_instance_uid="1.2.1", series_instance_uid="1.2.2",
        frame_of_reference_uid="1.2.3", sop_instance_uids=["1.2.4", "1.2.5", "1.2.6"],
        shape_zyx=[3, 8, 8], spacing_xyz=[0.7, 0.7, 2.0],
        origin_xyz=[10.0, 20.0, 30.0],
        direction=[1., 0., 0., 0., np.sqrt(3) / 2, -0.5, 0., 0.5, np.sqrt(3) / 2],
        source_ct_dir=str(tmp_path / "ct"), source_xml=str(tmp_path / "ann.xml"),
        source_xml_sha256="a" * 64, min_agreement=3,
        annotated_sop_instance_uids=["1.2.6"], reader_indices=[0, 1, 2],
    )
    source = rows.SourceSeries(mask.astype(np.float32), copy.deepcopy(metadata))
    cfg = SimpleNamespace(
        data=SimpleNamespace(n_slices=2, image_size=4, hu_window=[0.5, 1.0],
                             roi_margin_mm=0.7, context_factor=1.0,
                             crop_jitter_frac=0.0, slice_strategy="tumor_bbox",
                             min_tumor_voxels=1000000),
        paths=SimpleNamespace(processed_dir=str(tmp_path / "processed")),
    )
    path = tmp_path / "masks" / "scan.npz"
    rows.write_consensus_npz(path, mask, metadata)
    return mask, metadata, source, cfg, path


def test_round_trip_and_shared_actual_crop_keep_source_pixels(fixture, monkeypatch):
    mask, metadata, source, cfg, path = fixture
    requests = []

    def read(ct_dir, **kwargs):
        requests.append((ct_dir, kwargs))
        return source

    monkeypatch.setattr(rows, "read_source_series", read)
    # If the adapter accidentally goes through SEG handling, fail immediately.
    from theia.data import preprocess
    monkeypatch.setattr(preprocess, "segment_mask",
                        lambda *a, **kw: pytest.fail("NPZ was passed to SEG parser"))
    record = rows.process_archive(path, cfg)
    assert requests == [(metadata["source_ct_dir"], dict(
        series_uid="1.2.2", required_sops=metadata["sop_instance_uids"]))]
    assert record["patient_id"] == "LIDC-IDRI-0001"
    assert record["report"] == "" and record["labels"] == {}
    assert record["n_slices"] == 2 and record["has_mask"] is True
    assert record["transform"]["source_slice_indices"] == [2, 2]
    assert record["transform"]["source_sop_instance_uids"] == ["1.2.6", "1.2.6"]
    assert record["transform"]["crop_boxes_yxyx"] == [[-1, 3, 5, 9]] * 2
    assert record["source"]["direction"] == metadata["direction"]
    assert record["source"]["consensus_npz_sha256"] == rows._sha256(path)
    with np.load(record["npz"], allow_pickle=False) as output:
        assert output["images"].shape == output["roi"].shape == (2, 4, 4)
        assert output["images"].dtype == output["roi"].dtype == np.float32
        np.testing.assert_array_equal(output["images"], output["roi"])
        np.testing.assert_array_equal(output["roi"][0, 1:3, 1:3], np.ones((2, 2)))
        assert output["roi"].sum() == 8
    # Four native voxels survive even though the unrelated clinical threshold is
    # one million. This adapter does not invent an additional size eligibility gate.
    assert mask.sum() == 4


def test_emitted_rows_load_through_dataset_when_torch_available(fixture, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from theia.data.dataset import RadiogenomicsDataset

    _, _, source, cfg, path = fixture
    monkeypatch.setattr(rows, "read_source_series", lambda *a, **kw: source)
    rows.run(cfg, str(path.parent))
    dataset = RadiogenomicsDataset(str(cfg.paths.processed_dir) + "/rows.jsonl",
                                  ["EGFR", "KRAS"])
    item = dataset[0]
    assert item["images"].shape == item["roi"].shape == (2, 1, 4, 4)
    assert item["egfr"].item() == item["kras"].item() == -1
    assert item["report"] == ""


@pytest.mark.parametrize("key,change", [
    ("patient_id", "OTHER-PATIENT"),
    ("study_instance_uid", "9.1"),
    ("series_instance_uid", "9.2"),
    ("frame_of_reference_uid", "9.3"),
    ("sop_instance_uids", ["1.2.6", "1.2.5", "1.2.4"]),
    ("spacing_xyz", [0.8, 0.7, 2.0]),
    ("origin_xyz", [11., 20., 30.]),
    ("direction", [1., 0., 0., 0., 1., 0., 0., 0., 1.]),
])
def test_refuses_identity_order_or_geometry_mismatch(fixture, monkeypatch, key, change):
    _, _, source, cfg, path = fixture
    source.metadata[key] = change
    monkeypatch.setattr(rows, "read_source_series", lambda *a, **kw: source)
    with pytest.raises(ValueError, match=key):
        rows.process_archive(path, cfg)
    assert not Path(cfg.paths.processed_dir).exists()


def test_refuses_legacy_mask_without_identity(tmp_path):
    path = tmp_path / "old.npz"
    np.savez_compressed(path, mask=np.ones((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="re-export"):
        rows.load_consensus_npz(path)


@pytest.mark.parametrize("defect", ["empty", "nonbinary", "shape", "unmatched",
                                   "duplicate_sop", "unannotated", "readers"])
def test_refuses_invalid_mask_or_membership(fixture, tmp_path, defect):
    mask, metadata, _, _, _ = fixture
    if defect == "empty":
        mask[:] = 0
    elif defect == "nonbinary":
        mask[0, 0, 0] = 2
    elif defect == "shape":
        mask = mask[:2]
    elif defect == "unmatched":
        metadata["annotated_sop_instance_uids"] = ["9.9"]
    elif defect == "duplicate_sop":
        metadata["sop_instance_uids"][1] = "1.2.4"
    elif defect == "unannotated":
        mask[0, 1, 1] = 1
    else:
        metadata["reader_indices"] = [0, 1]
    with pytest.raises(ValueError):
        rows.write_consensus_npz(tmp_path / "bad.npz", mask, metadata)


def test_refuses_method_change_before_loading_ct(fixture, monkeypatch):
    _, _, _, cfg, path = fixture
    monkeypatch.setattr(rows, "read_source_series",
                        lambda *a, **kw: pytest.fail("should fail before CT loading"))
    with pytest.raises(ValueError, match="min_agreement"):
        rows.process_archive(path, cfg, expected_min_agreement=2)


def test_empty_source_directory_fails_before_dependency_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        rows.read_source_series(str(tmp_path / "missing"))


def test_empty_input_directory_never_writes_rows(fixture, tmp_path):
    _, _, _, cfg, _ = fixture
    with pytest.raises(ValueError, match="no consensus"):
        rows.run(cfg, str(tmp_path / "nothing"))
    assert not Path(cfg.paths.processed_dir).exists()


def test_run_publishes_existing_row_schema_and_source_map(fixture, monkeypatch):
    _, metadata, source, cfg, path = fixture
    received = []

    def read(ct_dir, **kwargs):
        received.append(ct_dir)
        return source

    monkeypatch.setattr(rows, "read_source_series", read)
    records = rows.run(cfg, str(path.parent), source_map={"1.2.2": "/moved/ct"})
    assert received == ["/moved/ct"]
    index = Path(cfg.paths.processed_dir) / "rows.jsonl"
    assert [json.loads(line) for line in index.read_text().splitlines()] == records
    with pytest.raises(FileExistsError, match="must be empty"):
        rows.run(cfg, str(path.parent))


def test_duplicate_patient_and_missing_mapping_refused_before_ct(fixture, monkeypatch):
    mask, metadata, _, cfg, path = fixture
    monkeypatch.setattr(rows, "read_source_series",
                        lambda *a, **kw: pytest.fail("must validate membership first"))
    with pytest.raises(ValueError, match="no CT directory"):
        rows.run(cfg, str(path.parent), source_map={})
    metadata["series_instance_uid"] = "1.2.99"
    rows.write_consensus_npz(path.parent / "second.npz", mask, metadata)
    with pytest.raises(ValueError, match="multiple LIDC series"):
        rows.run(cfg, str(path.parent))


def test_run_does_not_publish_partial_manifest(fixture, monkeypatch):
    mask, metadata, source, cfg, path = fixture
    second = copy.deepcopy(metadata)
    second["patient_id"], second["series_instance_uid"] = "LIDC-IDRI-0002", "9.9"
    rows.write_consensus_npz(path.parent / "second.npz", mask, second)

    def read(*args, **kwargs):
        if kwargs["series_uid"] == "9.9":
            raise ValueError("unmatched")
        return source

    monkeypatch.setattr(rows, "read_source_series", read)
    with pytest.raises(ValueError, match="unmatched"):
        rows.run(cfg, str(path.parent))
    assert not (Path(cfg.paths.processed_dir) / "rows.jsonl").exists()
    assert len(list(Path(cfg.paths.processed_dir).glob("*.npz"))) == 1


def test_empty_resized_target_is_refused(fixture, monkeypatch):
    _, _, source, cfg, path = fixture
    monkeypatch.setattr(rows, "read_source_series", lambda *a, **kw: source)
    cfg.data.image_size = 1  # resampling selects the padded zero, loses tiny ROI
    with pytest.raises(ValueError, match="vanished during crop/resize"):
        rows.process_archive(path, cfg)
    assert not Path(cfg.paths.processed_dir).exists()


def _fake_dicom_stack(fixture, monkeypatch, tmp_path):
    """Exercise the real loader boundary with synthetic header/reader objects."""
    _, metadata, source, _, _ = fixture
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir(exist_ok=True)
    direction = np.asarray(metadata["direction"]).reshape(3, 3)
    spacing = np.asarray(metadata["spacing_xyz"])
    origin = np.asarray(metadata["origin_xyz"])
    physical = lambda index: origin + direction @ (np.asarray(index) * spacing)
    ct = SimpleNamespace(
        GetDirection=lambda: metadata["direction"],
        GetSpacing=lambda: metadata["spacing_xyz"],
        GetOrigin=lambda: metadata["origin_xyz"],
        TransformIndexToPhysicalPoint=physical,
    )
    headers = {}
    # Deliberately lexically unsorted paths: GDCM's exact ordering must win.
    filenames = ["z.dcm", "a.dcm", "m.dcm"]
    for k, name in enumerate(filenames):
        headers[name] = SimpleNamespace(
            SOPInstanceUID=metadata["sop_instance_uids"][k], PatientID=metadata["patient_id"],
            StudyInstanceUID="1.2.1", SeriesInstanceUID="1.2.2", FrameOfReferenceUID="1.2.3",
            Modality="CT", Rows=8, Columns=8, ImagePositionPatient=physical((0, 0, k)),
            ImageOrientationPatient=np.r_[direction[:, 0], direction[:, 1]],
            PixelSpacing=[0.7, 0.7],
        )
    received = []
    reader = SimpleNamespace(
        GetGDCMSeriesIDs=lambda path: ["1.2.2"],
        GetGDCMSeriesFileNames=lambda path, uid: filenames,
        SetFileNames=lambda names: received.extend(names),
        Execute=lambda: ct,
    )
    monkeypatch.setitem(sys.modules, "SimpleITK", SimpleNamespace(
        ImageSeriesReader=lambda: reader, GetArrayFromImage=lambda image: source.volume_hu))
    monkeypatch.setitem(sys.modules, "pydicom", SimpleNamespace(
        dcmread=lambda name, **kwargs: headers[name]))
    return ct_dir, headers, received


def test_reader_preserves_nonlexical_sop_order_and_oblique_grid(fixture, monkeypatch, tmp_path):
    ct_dir, _, received = _fake_dicom_stack(fixture, monkeypatch, tmp_path)
    source = rows.read_source_series(str(ct_dir), required_sops=["1.2.6"])
    assert received == ["z.dcm", "a.dcm", "m.dcm"]
    assert source.metadata["sop_instance_uids"] == ["1.2.4", "1.2.5", "1.2.6"]
    assert source.metadata["origin_xyz"] == [10., 20., 30.]
    assert source.metadata["direction"] == fixture[1]["direction"]


def test_reader_refuses_ambiguous_series(fixture, monkeypatch, tmp_path):
    ct_dir, _, received = _fake_dicom_stack(fixture, monkeypatch, tmp_path)
    reader = sys.modules["SimpleITK"].ImageSeriesReader()
    reader.GetGDCMSeriesIDs = lambda path: ["1.2.2", "1.2.99"]
    with pytest.raises(ValueError, match="ambiguous source"):
        rows.read_source_series(str(ct_dir), required_sops=["1.2.6"])
    assert received == []


@pytest.mark.parametrize("defect", ["unmatched", "partial", "duplicate", "patient",
                                   "missing_identity", "position", "orientation",
                                   "spacing", "modality"])
def test_reader_refuses_wrong_source_or_irregular_geometry(fixture, monkeypatch, tmp_path, defect):
    ct_dir, headers, _ = _fake_dicom_stack(fixture, monkeypatch, tmp_path)
    required = ["1.2.6"]
    if defect == "unmatched":
        required = ["9.9"]
    elif defect == "partial":
        required = ["1.2.6", "9.9"]
    elif defect == "duplicate":
        headers["a.dcm"].SOPInstanceUID = "1.2.4"
    elif defect == "patient":
        headers["a.dcm"].PatientID = "OTHER"
    elif defect == "missing_identity":
        headers["a.dcm"].PatientID = None
    elif defect == "position":
        headers["a.dcm"].ImagePositionPatient = [10., 20., 32.1]
    elif defect == "orientation":
        headers["a.dcm"].ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
    elif defect == "spacing":
        headers["a.dcm"].PixelSpacing = [0.8, 0.7]
    else:
        headers["a.dcm"].Modality = "SEG"
    with pytest.raises(ValueError):
        rows.read_source_series(str(ct_dir), required_sops=required)


def test_export_adds_truthful_provenance_and_threshold(fixture, monkeypatch, tmp_path):
    from theia.data import lidc

    _, metadata, source, _, _ = fixture
    xml = tmp_path / "original.xml"
    # Three readers on SOP at index two; a fourth reader outlines another slice.
    square = [(4, 0), (7, 0), (7, 3), (4, 3)]
    sessions = []
    for uid in ["1.2.6"] * 3 + ["1.2.4"]:
        edges = "".join(f"<edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>"
                        for x, y in square)
        sessions.append("<readingSession><unblindedReadNodule><roi>"
                        f"<imageSOP_UID>{uid}</imageSOP_UID><inclusion>TRUE</inclusion>"
                        f"{edges}</roi></unblindedReadNodule></readingSession>")
    xml.write_text("<LidcReadMessage>" + "".join(sessions) + "</LidcReadMessage>")
    monkeypatch.setattr(rows, "read_source_series", lambda *a, **kw: source)
    path = lidc.export_mask_for_scan(str(xml), metadata["source_ct_dir"],
                                     str(tmp_path / "export"))
    mask, saved = rows.load_consensus_npz(path)
    assert mask[2].sum() > 0 and mask[:2].sum() == 0
    assert saved["min_agreement"] == 3
    assert saved["consensus_method"] == rows.CONSENSUS_METHOD
    assert saved["reader_indices"] == [0, 1, 2, 3]
    assert saved["annotated_sop_instance_uids"] == ["1.2.4", "1.2.6"]
    assert saved["source_xml_sha256"] == rows._sha256(xml)
    assert saved["sop_instance_uids"] == metadata["sop_instance_uids"]


def test_export_empty_annotations_fail_before_ct(tmp_path, monkeypatch):
    from theia.data import lidc

    xml = tmp_path / "empty.xml"
    xml.write_text("<LidcReadMessage/>")
    monkeypatch.setattr(rows, "read_source_series",
                        lambda *a, **kw: pytest.fail("must reject empty XML first"))
    with pytest.raises(ValueError, match="empty annotation source"):
        lidc.export_mask_for_scan(str(xml), "/missing", str(tmp_path / "export"))


@pytest.mark.parametrize("method", [None, "per-voxel reader union (>=1)"])
def test_missing_or_conflicting_method_is_refused(fixture, tmp_path, method):
    mask, metadata, _, _, _ = fixture
    if method is None:
        metadata.pop("consensus_method", None)
    else:
        metadata["consensus_method"] = method
    path = tmp_path / "wrong-method.npz"
    np.savez_compressed(path, mask=mask, metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="consensus_method"):
        rows.load_consensus_npz(path)


def _hole_xml(tmp_path, *, hole_readers=3, extra_nodule=False, reverse=False):
    outer = [(0, 0), (7, 0), (7, 7), (0, 7)]
    hole = [(2, 2), (5, 2), (5, 5), (2, 5)]

    def roi(points, include):
        if reverse:
            points = points[::-1]
        edges = "".join(f"<edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>"
                        for x, y in points)
        return ("<roi><imageSOP_UID>1.2.6</imageSOP_UID>"
                f"<inclusion>{include}</inclusion>{edges}</roi>")

    sessions = []
    for reader in range(3):
        nodule = roi(outer, "TRUE")
        if reader < hole_readers:
            nodule += roi(hole, "FALSE")
        nodules = f"<unblindedReadNodule>{nodule}</unblindedReadNodule>"
        if extra_nodule:
            nodules += ("<unblindedReadNodule>" +
                        roi([(2, 2), (4, 2), (4, 4), (2, 4)], "TRUE") +
                        "</unblindedReadNodule>")
        sessions.append(f"<readingSession>{nodules}</readingSession>")
    xml = tmp_path / "with-hole.xml"
    xml.write_text("<LidcReadMessage>" + "".join(sessions) + "</LidcReadMessage>")
    return xml


def test_polygon_boundary_pixels_are_not_nodule_pixels():
    from theia.data.lidc import _rasterize

    square = np.array([(0, 0), (7, 0), (7, 7), (0, 7)])
    mask = _rasterize(square, (8, 8))
    expected = np.zeros((8, 8), dtype=bool)
    expected[1:7, 1:7] = True
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_xml_hole_interior_and_border_are_subtracted_before_votes(tmp_path, reverse):
    from theia.data.lidc import consensus_mask, parse_xml

    contours = parse_xml(str(_hole_xml(tmp_path, reverse=reverse)))
    mask = consensus_mask(contours, {"1.2.6": 0}, (8, 8), 1)
    expected = np.zeros((1, 8, 8), dtype=np.uint8)
    expected[0, 1:7, 1:7] = 1
    expected[0, 2:6, 2:6] = 0
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("reverse,closed", [(False, False), (True, False), (False, True),
                                           (True, True)])
def test_diagonal_polygon_border_is_orientation_independent(reverse, closed):
    from theia.data.lidc import _rasterize

    polygon = [(0, 0), (6, 0), (0, 6)]
    if reverse:
        polygon.reverse()
    if closed:
        polygon.append(polygon[0])
    yy, xx = np.mgrid[:7, :7]
    expected = (xx > 0) & (yy > 0) & (xx + yy < 6)
    np.testing.assert_array_equal(_rasterize(np.array(polygon), (7, 7)), expected)


def test_concave_polygon_is_filled_without_adding_its_notch():
    from theia.data.lidc import _rasterize

    polygon = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])
    expected = np.zeros((7, 7), dtype=bool)
    expected[1, 1:6] = True
    expected[1:6, 1] = True
    np.testing.assert_array_equal(_rasterize(polygon, (7, 7)), expected)


def test_exclusion_does_not_erase_another_nodule_from_same_reader(tmp_path):
    from theia.data.lidc import consensus_mask, parse_xml

    contours = parse_xml(str(_hole_xml(tmp_path, extra_nodule=True)))
    assert len({c.nodule_idx for c in contours["1.2.6"]}) == 2
    mask = consensus_mask(contours, {"1.2.6": 0}, (8, 8), 1)
    assert mask[0, 3, 3] == 1  # second nodule fills this pixel of first nodule's hole
    assert mask[0, 4, 4] == 0  # still in first hole, on second nodule's outer border


def test_hole_is_removed_from_only_its_readers_vote(tmp_path):
    from theia.data.lidc import consensus_mask, parse_xml

    contours = parse_xml(str(_hole_xml(tmp_path, hole_readers=1)))
    strict = consensus_mask(contours, {"1.2.6": 0}, (8, 8), 1, min_agreement=3)
    loose = consensus_mask(contours, {"1.2.6": 0}, (8, 8), 1, min_agreement=2)
    assert strict[0, 3, 3] == 0
    assert loose[0, 3, 3] == 1
    assert strict[0, 1, 1] == loose[0, 1, 1] == 1


def test_v1_archive_requires_reexport(fixture, tmp_path):
    mask, metadata, _, _, _ = fixture
    metadata["schema"] = "theia.lidc.consensus.v1"
    path = tmp_path / "v1.npz"
    np.savez_compressed(path, mask=mask, metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="re-export"):
        rows.load_consensus_npz(path)


def test_export_exclusion_only_is_empty_before_ct(tmp_path, monkeypatch):
    from theia.data import lidc

    xml = _hole_xml(tmp_path)
    xml.write_text(xml.read_text().replace("<inclusion>TRUE", "<inclusion>FALSE"))
    monkeypatch.setattr(rows, "read_source_series",
                        lambda *a, **kw: pytest.fail("no included source to load CT for"))
    with pytest.raises(ValueError, match="empty annotation source"):
        lidc.export_mask_for_scan(str(xml), "/missing", str(tmp_path / "export"))
