"""Convert source-bound LIDC consensus NPZs into label-free training rows.

The archive is a native CT-grid mask, never a DICOM SEG. Legacy mask-only NPZs
cannot establish source identity and must be re-exported with ``theia.data.lidc``.
See docs/LIDC_ROWS.md for the schema and a bounded pilot command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "theia.lidc.consensus.v2"
CONSENSUS_METHOD = "distinct-reader-vote/nodule-inclusion-minus-exclusion/strict-polygon-interior/v2"
IDENTITY_KEYS = ("patient_id", "study_instance_uid", "series_instance_uid",
                 "frame_of_reference_uid")
GEOMETRY_SHAPES = {"spacing_xyz": (3,), "origin_xyz": (3,), "direction": (9,)}


@dataclass
class SourceSeries:
    volume_hu: np.ndarray
    metadata: dict


def _text(ds, field: str) -> str:
    raw = getattr(ds, field, None)
    value = "" if raw is None else str(raw).strip()
    if not value:
        raise ValueError(f"CT header missing {field}")
    return value


def read_source_series(ct_dir: str, *, series_uid: str | None = None,
                       required_sops=()) -> SourceSeries:
    """Read exactly one matching CT series, using the same ordered file list.

    Check every DICOM slice against the loaded physical grid. Nonuniform spacing,
    mixed identities, duplicate SOPs, and multi-frame inputs are refused rather
    than represented as an apparently regular mask volume.
    """
    if not Path(ct_dir).is_dir():
        raise FileNotFoundError(f"CT source directory does not exist: {ct_dir}")
    import pydicom
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    ids = list(reader.GetGDCMSeriesIDs(str(ct_dir)) or ())
    if series_uid is not None:
        ids = [uid for uid in ids if str(uid) == series_uid]
    required = set(required_sops)
    candidates = []
    for uid in ids:
        files = list(reader.GetGDCMSeriesFileNames(str(ct_dir), uid))
        headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
        sops = [_text(h, "SOPInstanceUID") for h in headers]
        if headers and required.issubset(sops):
            candidates.append((str(uid), files, headers, sops))
    if len(candidates) != 1:
        raise ValueError(
            f"{ct_dir}: expected exactly one CT series matching all annotation "
            f"SOP UIDs; found {len(candidates)} (unmatched or ambiguous source)")
    uid, files, headers, sops = candidates[0]
    if len(sops) != len(set(sops)):
        raise ValueError("duplicate SOPInstanceUID in CT source")
    fields = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID",
              "FrameOfReferenceUID")
    identity = tuple(_text(headers[0], field) for field in fields)
    if identity[2] != uid:
        raise ValueError("DICOM series UID disagrees with the selected series")
    for h in headers:
        if _text(h, "Modality") != "CT":
            raise ValueError("source contains a non-CT image")
        if tuple(_text(h, field) for field in fields) != identity:
            raise ValueError("mixed patient/study/series/frame identity in CT source")
        if int(getattr(h, "NumberOfFrames", 1)) != 1:
            raise ValueError("multi-frame CT is not supported by this slice adapter")
    reader.SetFileNames(files)
    ct = reader.Execute()
    vol = sitk.GetArrayFromImage(ct).astype(np.float32)
    if vol.ndim != 3 or vol.shape[0] != len(sops):
        raise ValueError("CT array depth does not match the ordered SOP list")
    direction = np.asarray(ct.GetDirection(), dtype=float).reshape(3, 3)
    orientation = np.concatenate((direction[:, 0], direction[:, 1]))
    spacing = ct.GetSpacing()
    for k, h in enumerate(headers):
        if (int(h.Rows), int(h.Columns)) != vol.shape[1:]:
            raise ValueError("CT slice dimensions do not match the loaded grid")
        for name, actual, expected in (
            ("position", getattr(h, "ImagePositionPatient", ()),
             ct.TransformIndexToPhysicalPoint((0, 0, k))),
            ("orientation", getattr(h, "ImageOrientationPatient", ()), orientation),
            ("pixel spacing", getattr(h, "PixelSpacing", ()), spacing[1::-1]),
        ):
            actual = np.asarray(actual, dtype=float)
            expected = np.asarray(expected, dtype=float)
            if actual.shape != expected.shape or not np.allclose(
                    actual, expected, rtol=0, atol=1e-4):
                raise ValueError(f"CT slice {k} {name} differs from the loaded grid")
    metadata = dict(zip(IDENTITY_KEYS, identity))
    metadata.update(
        source_ct_dir=str(Path(ct_dir).resolve()),
        sop_instance_uids=sops,
        shape_zyx=list(vol.shape),
        spacing_xyz=list(spacing),
        origin_xyz=list(ct.GetOrigin()),
        direction=list(ct.GetDirection()),
    )
    _validate_geometry(metadata)
    if not np.isfinite(vol).all():
        raise ValueError("CT contains non-finite intensities")
    return SourceSeries(vol, metadata)


def _validate_geometry(metadata: dict) -> None:
    for key in IDENTITY_KEYS:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"missing source identity: {key}")
    shape = metadata.get("shape_zyx")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape)):
        raise ValueError("invalid shape_zyx")
    sops = metadata.get("sop_instance_uids")
    if (not isinstance(sops, list) or len(sops) != shape[0]
            or any(not isinstance(uid, str) or not uid.strip() for uid in sops)
            or len(set(sops)) != len(sops)):
        raise ValueError("ordered SOP list must be unique and match the CT depth")
    for key, shape in GEOMETRY_SHAPES.items():
        values = np.asarray(metadata.get(key), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid geometry: {key}")
    if np.any(np.asarray(metadata["spacing_xyz"]) <= 0):
        raise ValueError("CT spacing must be positive")
    direction = np.asarray(metadata["direction"]).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("CT direction must be orthonormal")


def write_consensus_npz(path: str | Path, mask: np.ndarray, metadata: dict) -> None:
    """Write a versioned, pickle-free archive, refusing ambiguous metadata."""
    _validate_archive(mask, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting a prior export or basename collision.
    with path.open("xb") as stream:
        np.savez_compressed(stream, mask=mask.astype(np.uint8),
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def _validate_archive(mask: np.ndarray, metadata: dict) -> None:
    if metadata.get("schema") != SCHEMA or metadata.get("mask_axes") != "zyx":
        raise ValueError("unsupported consensus schema or mask axes; re-export the XML")
    if metadata.get("consensus_method") != CONSENSUS_METHOD:
        raise ValueError("missing or unsupported consensus_method; re-export the XML")
    _validate_geometry(metadata)
    if list(mask.shape) != metadata["shape_zyx"]:
        raise ValueError("mask shape does not match the declared source CT grid")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("consensus mask must be binary")
    if not mask.any():
        raise ValueError("empty consensus mask; no usable grounding source")
    agreement = metadata.get("min_agreement")
    if type(agreement) is not int or not 1 <= agreement <= 4:
        raise ValueError("min_agreement must be an integer from 1 to 4")
    annotated = metadata.get("annotated_sop_instance_uids")
    if (not isinstance(annotated, list) or not annotated
            or any(not isinstance(uid, str) for uid in annotated)
            or len(annotated) != len(set(annotated))
            or not set(annotated).issubset(metadata["sop_instance_uids"])):
        raise ValueError("annotation SOP membership is empty or unmatched")
    occupied = {metadata["sop_instance_uids"][i]
                for i in np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))}
    if not occupied.issubset(annotated):
        raise ValueError("mask occupies a slice without annotation provenance")
    readers = metadata.get("reader_indices")
    if (not isinstance(readers, list) or not readers
            or any(type(r) is not int or not 0 <= r <= 3 for r in readers)
            or len(set(readers)) != len(readers)
            or len(readers) < agreement):
        raise ValueError("invalid reader provenance for consensus threshold")
    digest = metadata.get("source_xml_sha256", "")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("missing source XML SHA-256 provenance")
    for key in ("source_ct_dir", "source_xml"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"missing source path: {key}")


def load_consensus_npz(path: str | Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive or "mask" not in archive:
            raise ValueError(
                f"{path}: mask-only/legacy NPZ has no source identity; "
                "re-export from XML with theia.data.lidc")
        raw = archive["metadata_json"]
        if raw.ndim != 0 or raw.dtype.kind != "U":
            raise ValueError("metadata_json must be a scalar Unicode JSON string")
        metadata = json.loads(raw.item())
        mask = archive["mask"]
    if not isinstance(metadata, dict):
        raise ValueError("consensus metadata must be a JSON object")
    _validate_archive(mask, metadata)
    return mask, metadata


def _match_source(mask: np.ndarray, metadata: dict, source: SourceSeries) -> None:
    actual = source.metadata
    _validate_geometry(actual)
    for key in (*IDENTITY_KEYS, "sop_instance_uids", "shape_zyx"):
        if metadata[key] != actual[key]:
            raise ValueError(f"consensus/source mismatch: {key}")
    if list(source.volume_hu.shape) != list(mask.shape):
        raise ValueError("consensus/source mismatch: CT array shape")
    for key in GEOMETRY_SHAPES:
        if not np.allclose(metadata[key], actual[key], rtol=0, atol=1e-4):
            raise ValueError(f"consensus/source mismatch: {key}")
    if not np.isfinite(source.volume_hu).all():
        raise ValueError("CT contains non-finite intensities")


def process_archive(path: str | Path, cfg, *, ct_dir: str | None = None,
                    expected_min_agreement: int = 3) -> dict:
    """Create one existing-schema row, retaining exact source/crop provenance."""
    mask, metadata = load_consensus_npz(path)
    if metadata["min_agreement"] != expected_min_agreement:
        raise ValueError("consensus min_agreement differs from the requested method")
    source = read_source_series(
        ct_dir or metadata["source_ct_dir"],
        series_uid=metadata["series_instance_uid"],
        required_sops=metadata["sop_instance_uids"])
    _match_source(mask, metadata, source)
    # These are the actual training transforms. Do not call the SEG parser or
    # apply the separate clinical cohort's min_tumor_voxels eligibility rule.
    from theia.data.preprocess import (_resize, crop_box, roi_box, tumor_slices,
                                       window_hu)

    data = cfg.data
    n, size = int(data.n_slices), int(data.image_size)
    if n < 1 or size < 1:
        raise ValueError("n_slices and image_size must be positive")
    center, width = map(float, data.hu_window)
    margin_mm = float(data.roi_margin_mm)
    context = float(getattr(data, "context_factor", 1.0))
    jitter = float(getattr(data, "crop_jitter_frac", 0.0))
    if (not np.isfinite([center, width, margin_mm, context, jitter]).all()
            or width <= 0 or margin_mm < 0 or context < 1 or not 0 <= jitter <= 1):
        raise ValueError("invalid HU window or crop parameters")
    vol = window_hu(source.volume_hu, center, width)
    margin_px = int(round(margin_mm / metadata["spacing_xyz"][0]))
    # Stable across Python processes; builtin hash(patient_id) is randomized.
    seed = int.from_bytes(hashlib.sha256(
        metadata["patient_id"].encode()).digest()[:4], "big")
    slices = [int(i) for i in tumor_slices(mask, n, data.slice_strategy)]
    if len(slices) != n:
        raise ValueError("slice strategy did not return n_slices; max_area requires an even count")
    images, rois, boxes = [], [], []
    for index in slices:
        m2d = mask[index].astype(np.float32)
        box = roi_box(m2d, vol[index].shape, margin_px, context, jitter,
                      np.random.default_rng(seed))
        images.append(_resize(crop_box(vol[index], box), size))
        rois.append(_resize(crop_box(m2d, box), size))
        boxes.append([int(v) for v in box])
    images = np.stack(images).astype(np.float32)
    roi = (np.stack(rois) > 0.5).astype(np.float32)
    if not roi.any():
        raise ValueError("consensus mask vanished during crop/resize; no usable grounding target")
    name = "lidc_" + hashlib.sha256(
        metadata["series_instance_uid"].encode()).hexdigest()[:20] + ".npz"
    out = Path(cfg.paths.processed_dir).resolve() / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, images=images, roi=roi)
    return dict(
        patient_id=metadata["patient_id"], npz=str(out), report="", labels={},
        n_slices=len(slices), has_mask=True, cohort="LIDC-IDRI",
        source=dict(
            consensus_npz=str(Path(path).resolve()),
            consensus_npz_sha256=_sha256(path),
            **metadata,
            resolved_ct_dir=source.metadata["source_ct_dir"],
        ),
        transform=dict(
            hu_window=[center, width], roi_margin_mm=margin_mm,
            margin_px=margin_px, context_factor=context, crop_jitter_frac=jitter,
            crop_seed=seed, image_size=size, slice_strategy=data.slice_strategy,
            source_slice_indices=slices,
            source_sop_instance_uids=[metadata["sop_instance_uids"][i] for i in slices],
            crop_boxes_yxyx=boxes, image_resize="scipy.ndimage.zoom(order=1)",
            roi_resize="scipy.ndimage.zoom(order=1), threshold > 0.5",
        ),
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cfg, masks_dir: str, *, source_map: dict | None = None,
        expected_min_agreement: int = 3, limit: int | None = None) -> list[dict]:
    """Fail on any invalid input; publish rows.jsonl only after every row passes."""
    paths = sorted(Path(masks_dir).glob("*.npz"))
    if not paths:
        raise ValueError(f"no consensus NPZ sources in {masks_dir}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        paths = paths[:limit]
    if source_map is not None and (
            not isinstance(source_map, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) or not v
                   for k, v in source_map.items())):
        raise ValueError("source_map must map SeriesInstanceUID strings to CT directories")
    # Check patient uniqueness before creating outputs: the existing dataset's
    # split helper assumes one row per person. Do not silently select a series.
    seen_patients = set()
    sources = []
    for path in paths:
        _, metadata = load_consensus_npz(path)
        pid = metadata["patient_id"]
        if pid in seen_patients:
            raise ValueError(f"multiple LIDC series/archives for patient {pid}; "
                             "resolve cohort membership before row generation")
        seen_patients.add(pid)
        uid = metadata["series_instance_uid"]
        if source_map is not None and uid not in source_map:
            raise ValueError(f"source map has no CT directory for series {uid}")
        sources.append(None if source_map is None else source_map[uid])
    out = Path(cfg.paths.processed_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")
    rows = [process_archive(path, cfg, ct_dir=ct_dir,
                            expected_min_agreement=expected_min_agreement)
            for path, ct_dir in zip(paths, sources)]
    with (out / "rows.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[lidc-rows] wrote {len(rows)} source-bound label-free rows -> {out}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--source_map", help="JSON {SeriesInstanceUID: CT directory}; "
                    "required for moved sources, otherwise use recorded paths")
    ap.add_argument("--expected_min_agreement", type=int, choices=range(1, 5), default=3)
    ap.add_argument("--limit", type=int, help="bounded pilot: first N sorted NPZs only")
    args = ap.parse_args()
    from theia.config import load_config

    cfg = load_config(args.config)
    cfg["paths"]["processed_dir"] = args.out_dir
    source_map = None
    if args.source_map:
        with open(args.source_map) as stream:
            source_map = json.load(stream)
    run(cfg, args.masks_dir, source_map=source_map,
        expected_min_agreement=args.expected_min_agreement, limit=args.limit)


if __name__ == "__main__":
    main()
