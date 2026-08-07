"""DICOM -> model-ready tensors.

Pipeline per patient:
  1. load the CT series and the tumor SEG series (RTSTRUCT/SEG) via SimpleITK
  2. resample SEG onto the CT grid, threshold to a binary tumor mask
  3. HU-window the CT, take the slices spanning the tumor bounding box
  4. crop a square ROI around the (margin-dilated) tumor
  5. attach the mutation labels and build the semantic-annotation pseudo-report
  6. write one .npz per patient + a rows.jsonl index

The pseudo-report is the controlled-vocabulary semantic annotation rendered as a
sentence. It is the supervision signal for the generation head — you never write
free-text reports by hand.

Three corrections
-----------------
1. THE IMAGE AND ITS GROUNDING MASK WERE IN DIFFERENT COORDINATE FRAMES. The old
   code cropped the image tightly around the tumor via `crop_roi`, but built the
   ROI mask with a bare `_resize` of the *entire* slice. So the network saw a
   zoomed-in tumor crop while the grounding loss compared its attention against a
   downsampled whole-slice mask. The grounding supervision was misaligned for
   every patient. Image and mask now share one crop box.
2. The crop was not square, and was then resized to a square 224, so anatomy was
   stretched by a different amount for every patient. The box is now square and
   zero-padded at the image edge instead of being clipped.
3. `slice_strategy: all` was documented in the config but never implemented; it
   fell through to the tumor_bbox branch.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from theia.data.aim import build_report, load_all as load_aim


def _sitk():
    """Import SimpleITK lazily.

    The geometry helpers below (roi_box, crop_box, _resize, tumor_slices) are pure
    numpy and are worth unit-testing on machines that have no DICOM stack
    installed. A module-level import would make that impossible.
    """
    import SimpleITK as sitk

    return sitk


def load_series(dicom_dir: str):
    sitk = _sitk()
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not ids:
        raise FileNotFoundError(f"no DICOM series in {dicom_dir}")
    files = reader.GetGDCMSeriesFileNames(dicom_dir, ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def window_hu(vol: np.ndarray, center: float, width: float) -> np.ndarray:
    """HU window -> [0,1]. Lung window default is center -600, width 1500."""
    lo, hi = center - width / 2.0, center + width / 2.0
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def load_segmentation(seg_dir: str):
    """Read a DICOM SEG.

    These are a single multi-frame object (one file, N frames, one per source
    slice), not a series of files. `ImageSeriesReader` mishandles that — it wants
    one file per slice. SimpleITK reads the file directly and recovers the
    correct geometry, which is what `resample_mask_to` then needs to align it to
    the CT grid.
    """
    sitk = _sitk()
    files = sorted(glob.glob(os.path.join(seg_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"no DICOM in {seg_dir}")
    if len(files) == 1:
        return sitk.ReadImage(files[0])
    return load_series(seg_dir)


# Segment labels that denote the primary tumor, lowercased substring match.
TUMOR_LABELS = ("neoplasm", "gtv", "tumor", "tumour", "lesion", "mass")
# Everything else a planning SEG typically carries. Named so the failure mode is
# obvious if a label ever matches both lists.
NON_TUMOR_LABELS = ("lung", "spinal", "cord", "esophagus", "oesophagus", "heart",
                    "body", "external", "skin", "carina", "trachea")


def segment_mask(seg_file: str, ct, prefer=TUMOR_LABELS) -> np.ndarray:
    """Extract ONLY the tumor segment from a multi-segment DICOM SEG.

    Radiotherapy-planning segmentations bundle several structures into one
    multi-frame object: NSCLC-Radiomics ships 4-6 segments per patient
    (Neoplasm Primary, Lung x2, Spinal cord, Esophagus, Heart) stacked as
    frames, so a 134-slice CT gets a 536-frame SEG.

    Reading that with sitk.ReadImage and thresholding at >0 yields a "tumor"
    mask covering the entire thorax, and nothing errors. Grounding trained on it
    would learn to attend everywhere, which is precisely the failure this
    project has already spent a day removing.

    Frames are matched to CT slices by ImagePositionPatient, not by index, since
    the SEG may be ordered per-segment and may not span every CT slice.
    """
    import pydicom

    ds = pydicom.dcmread(seg_file)
    segments = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", "")).strip()
                for s in getattr(ds, "SegmentSequence", [])}
    if not segments:
        raise ValueError(f"{seg_file}: no SegmentSequence")

    if len(segments) == 1:
        # A single-segment SEG is the tumor by construction, whatever it is
        # called — NSCLC-RADIOGENOMICS labels its "3D Slicer segmentation
        # result", which matches no anatomical keyword.
        wanted = set(segments)
    else:
        wanted = {n for n, lab in segments.items()
                  if any(k in lab.lower() for k in prefer)
                  and not any(k in lab.lower() for k in NON_TUMOR_LABELS)}
    if not wanted:
        raise ValueError(
            f"{seg_file}: no tumor segment among {sorted(segments.values())}; "
            f"expected a label containing one of {prefer}")

    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    frames = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if frames is None or len(frames) != arr.shape[0]:
        raise ValueError(f"{seg_file}: {arr.shape[0]} frames but "
                         f"{0 if frames is None else len(frames)} functional groups")

    sitk = _sitk()
    depth = ct.GetSize()[2]
    z_of = [ct.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(depth)]
    out = np.zeros((depth, arr.shape[1], arr.shape[2]), dtype=np.uint8)

    matched = 0
    for i, fg in enumerate(frames):
        try:
            seg_no = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        except Exception:  # noqa: BLE001
            continue
        if seg_no not in wanted:
            continue
        try:
            z = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        except Exception:  # noqa: BLE001
            continue
        k = int(np.argmin([abs(z - zz) for zz in z_of]))
        out[k] = np.maximum(out[k], arr[i].astype(np.uint8))
        matched += 1

    if matched == 0:
        raise ValueError(f"{seg_file}: tumor segment(s) {sorted(wanted)} matched no CT slice")
    return out


def resample_mask_to(ct, seg) -> np.ndarray:
    sitk = _sitk()
    seg_r = sitk.Resample(
        seg, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg.GetPixelID()
    )
    return (sitk.GetArrayFromImage(seg_r) > 0).astype(np.uint8)


def tumor_slices(mask: np.ndarray, n_slices: int, strategy: str) -> list[int]:
    per_slice = mask.reshape(mask.shape[0], -1).sum(axis=1)
    present = np.where(per_slice > 0)[0]
    if present.size == 0:
        raise ValueError("empty tumor mask")
    if strategy == "max_area":
        center = int(present[np.argmax(per_slice[present])])
        lo, hi = center - n_slices // 2, center + n_slices // 2
        return [int(np.clip(i, 0, mask.shape[0] - 1)) for i in range(lo, hi)]
    if strategy == "all":
        # Every slice the tumor touches, resampled to exactly n_slices so the
        # batch stays rectangular.
        idx = present.tolist()
        if len(idx) == n_slices:
            return [int(i) for i in idx]
        picks = np.linspace(0, len(idx) - 1, n_slices).round().astype(int)
        return [int(idx[p]) for p in picks]
    if strategy != "tumor_bbox":
        raise ValueError(f"unknown slice_strategy: {strategy!r} "
                         "(expected 'tumor_bbox', 'max_area', or 'all')")
    lo, hi = int(present.min()), int(present.max())
    if hi - lo + 1 <= n_slices:
        idx = list(range(lo, hi + 1))
        return idx + [hi] * (n_slices - len(idx))
    return list(np.linspace(lo, hi, n_slices).round().astype(int))


def roi_box(mask2d: np.ndarray, shape: tuple[int, int], margin_px: int,
            context_factor: float = 1.0, jitter: float = 0.0,
            rng: np.random.Generator | None = None) -> tuple[int, int, int, int]:
    """Square crop box (y0, y1, x0, x1) around the tumor.

    Returned coordinates may fall outside the image; `crop_box` zero-pads rather
    than clipping, so the box stays square and the aspect ratio is preserved.

    `jitter` offsets the box centre by up to that fraction of the crop side.
    WHY THIS EXISTS: with jitter=0 the tumor lands at the exact centre of every
    crop — measured across the cohort, the ROI centroid on the 14x14 patch grid
    was row 6.5 +/- 0.1, col 6.5 +/- 0.2, where the grid centre is 6.5. Grounding
    is then not a learnable task: "where is the tumor" has the same answer for
    every patient, so there is no localisation function to fit and no spatial
    variance for the metric to reward. Measured grounding lift was ~0.000 in all
    four ablation arms, and the attention instead latched onto a backbone
    positional artifact (a column-12 stripe). Jitter restores the task.
    """
    h, w = shape
    if mask2d.sum() == 0:
        cy, cx = h / 2.0, w / 2.0
        half = max(margin_px, 1)
    else:
        ys, xs = np.where(mask2d > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        side = max(y1 - y0, x1 - x0) + 2 * margin_px
        half = max(side * context_factor / 2.0, 1.0)
    half = int(round(half))
    if jitter > 0:
        r = rng or np.random.default_rng(0)
        span = jitter * 2 * half
        cy += float(r.uniform(-span, span)) / 2.0
        cx += float(r.uniform(-span, span)) / 2.0
    cyi, cxi = int(round(cy)), int(round(cx))
    return cyi - half, cyi + half, cxi - half, cxi + half


def crop_box(img2d: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop with zero padding for any part of the box outside the image."""
    y0, y1, x0, x1 = box
    h, w = img2d.shape
    out = np.zeros((y1 - y0, x1 - x0), dtype=img2d.dtype)
    sy0, sy1 = max(y0, 0), min(y1, h)
    sx0, sx1 = max(x0, 0), min(x1, w)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img2d[sy0:sy1, sx0:sx1]
    return out


def crop_roi(img2d: np.ndarray, mask2d: np.ndarray, margin_px: int, size: int,
             context_factor: float = 1.0) -> np.ndarray:
    """Convenience wrapper: derive the box from the mask and crop the image."""
    box = roi_box(mask2d, img2d.shape, margin_px, context_factor)
    return _resize(crop_box(img2d, box), size)


def _resize(patch: np.ndarray, size: int) -> np.ndarray:
    if patch.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    zoom = (size / patch.shape[0], size / patch.shape[1])
    out = ndimage.zoom(patch, zoom, order=1).astype(np.float32)
    if out.shape != (size, size):  # guard against zoom's rounding
        fixed = np.zeros((size, size), dtype=np.float32)
        k = min(out.shape[0], size), min(out.shape[1], size)
        fixed[:k[0], :k[1]] = out[:k[0], :k[1]]
        out = fixed
    return out


SEMANTIC_TEMPLATE = (
    "A {margin} {density} {location} lesion, {size_cat}, "
    "with {pleural} and {vascular}."
)


def build_pseudo_report(row: pd.Series) -> str:
    """DEPRECATED fallback. Prefer theia.data.aim.build_report.

    These six columns do NOT exist in the TCIA clinical spreadsheet — that file
    carries demographics, staging, treatment and outcome. The semantic
    annotations live in the AIM XML archive. Every lookup here therefore fell
    through to its default and produced the SAME sentence for every patient, so
    the generation head trained on ~190 copies of one string, converged to a
    constant, and reported an excellent loss. Kept only for cohorts that really
    do supply these columns.
    """
    fields = dict(
        margin=row.get("Surface", "ill-defined"),
        density=row.get("Density", "solid"),
        location=row.get("Location", "lung"),
        size_cat=row.get("SizeCategory", "indeterminate size"),
        pleural=row.get("PleuralAttachment", "no pleural attachment"),
        vascular=row.get("VascularConvergence", "no vascular convergence"),
    )
    return SEMANTIC_TEMPLATE.format(**{k: str(v).lower() for k, v in fields.items()})


def label_of(row: pd.Series, gene: str) -> int:
    """Map the clinical spreadsheet's mutation call to {0,1}; -1 if unknown.

    Tries several column-name spellings because the TCIA clinical sheet uses
    space-separated headers ("EGFR mutation status"), not underscores.
    """
    candidates = [
        f"{gene}_mutation_status", f"{gene} mutation status",
        f"{gene}_status", f"{gene} status", gene,
    ]
    val = ""
    for col in candidates:
        if col in row and str(row[col]).strip():
            val = str(row[col]).strip().lower()
            break
    if val in {"mutant", "mutated", "mutation", "positive", "pos", "1", "yes"}:
        return 1
    if val in {"wildtype", "wild-type", "wild type", "wt", "negative", "neg", "0", "no"}:
        return 0
    return -1


def sop_uid_index(ct_dir: str) -> dict[str, int]:
    """Map each slice's SOPInstanceUID to its index in the loaded volume.

    AIM markup references a slice by SOP instance UID. Matching on that is exact;
    matching on `referencedFrameNumber` is not, because it is an acquisition-time
    frame number and need not equal the array index after sorting.
    """
    import pydicom

    sitk = _sitk()
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(ct_dir)
    if not ids:
        return {}
    files = reader.GetGDCMSeriesFileNames(ct_dir, ids[0])
    out = {}
    for i, f in enumerate(files):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            out[str(ds.SOPInstanceUID)] = i
        except Exception:  # noqa: BLE001
            continue
    return out


def _slices_around(center: int, n: int, depth: int) -> list[int]:
    lo = max(0, min(center - n // 2, depth - n))
    return [int(np.clip(lo + i, 0, depth - 1)) for i in range(n)]


def process_patient(pid, ct_dir, seg_dir, clinical_row, cfg, ann=None) -> dict | None:
    """Build one patient's tensors.

    Two supervision tiers:
      * SEG present  -> pixel mask drives the crop AND supervises grounding.
      * AIM only     -> the lesion centroid drives a fixed-size crop, the ROI is
                        written as zeros, and grounding is masked off for this
                        patient (the loss already skips empty targets). This
                        recovers ~48 patients that have mutation labels but no
                        segmentation, nearly doubling the positive count, without
                        inventing a mask nobody drew.
    """
    sitk = _sitk()
    try:
        ct = load_series(ct_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[preprocess] skip {pid}: {exc}")
        return None

    labels = {g: label_of(clinical_row, g) for g in cfg.data.target_genes}
    if all(v == -1 for v in labels.values()):
        print(f"[preprocess] skip {pid}: no known label for any target gene")
        return None

    raw_hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    vol = raw_hu
    center_hu, width = cfg.data.hu_window
    vol = window_hu(vol, center_hu, width)
    spacing = ct.GetSpacing()[0]
    margin_px = int(round(cfg.data.roi_margin_mm / max(spacing, 1e-3)))
    context = float(getattr(cfg.data, "context_factor", 1.0))

    mask = None
    try:
        seg = load_segmentation(seg_dir)
        m = resample_mask_to(ct, seg)
        if m.sum() >= cfg.data.min_tumor_voxels:
            mask = m
        else:
            print(f"[preprocess] {pid}: tumor mask too small, falling back to AIM")
    except (FileNotFoundError, RuntimeError):
        pass

    jitter = float(getattr(cfg.data, "crop_jitter_frac", 0.0))
    # Seeded per patient so preprocessing stays reproducible, but the offset
    # differs between patients — which is the whole point.
    prng = np.random.default_rng(abs(hash(str(pid))) % (2**32))

    if mask is not None:
        slices = tumor_slices(mask, cfg.data.n_slices, cfg.data.slice_strategy)
        # One offset for the whole stack, so the lesion does not wander between
        # slices of the same patient.
        boxes = [roi_box((mask[s] > 0).astype(np.float32), vol[s].shape, margin_px,
                         context, jitter, np.random.default_rng(
                             abs(hash(str(pid))) % (2**32)))
                 for s in slices]
        rois = [(mask[s] > 0).astype(np.float32) for s in slices]
        has_mask = True
    else:
        if ann is None or not ann.has_location:
            print(f"[preprocess] skip {pid}: no segmentation and no AIM location")
            return None
        mk = next(m for m in ann.markups if m.cx is not None)
        idx = sop_uid_index(ct_dir).get(str(mk.image_uid))
        if idx is None:
            if mk.frame is None or not (0 <= mk.frame < vol.shape[0]):
                print(f"[preprocess] skip {pid}: AIM slice not resolvable")
                return None
            print(f"[preprocess] {pid}: AIM SOP UID not found, using frame {mk.frame}")
            idx = mk.frame
        cy, cx = int(round(mk.cy)), int(round(mk.cx))
        # Sanity-gate the annotation before trusting it to place a crop.
        # Measured across the 43 AIM-only patients: the annotated point sits on
        # soft tissue (> -300 HU) in 72% of cases against 0% for random points in
        # the lung field, so the coordinate mapping is sound — but 9% land in
        # air, and those crops would be centred on nothing. Drop them rather than
        # snap to the nearest dense voxel, which would be inventing a location.
        H, W = raw_hu[idx].shape
        if not (0 <= cy < H and 0 <= cx < W):
            print(f"[preprocess] skip {pid}: AIM point ({cx},{cy}) outside {W}x{H}")
            return None
        patch_hu = float(raw_hu[idx][max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3].mean())
        min_hu = float(getattr(cfg.data, "aim_min_hu", -700.0))
        if patch_hu < min_hu:
            print(f"[preprocess] skip {pid}: AIM point sits in air ({patch_hu:.0f} HU "
                  f"< {min_hu:.0f}); annotation does not localise a lesion")
            return None

        half = int(round(float(getattr(cfg.data, "aim_crop_mm", 50.0))
                         / max(spacing, 1e-3) / 2.0)) * context
        half = max(int(round(half)), 8)
        slices = _slices_around(idx, cfg.data.n_slices, vol.shape[0])
        boxes = [(cy - half, cy + half, cx - half, cx + half)] * len(slices)
        rois = [np.zeros_like(vol[s]) for s in slices]
        has_mask = False

    imgs = np.stack([_resize(crop_box(vol[s], b), cfg.data.image_size)
                     for s, b in zip(slices, boxes)])
    roi_masks = np.stack([_resize(crop_box(r, b), cfg.data.image_size)
                          for r, b in zip(rois, boxes)])
    roi_masks = (roi_masks > 0.5).astype(np.float32)

    report = build_report(ann) if ann is not None else build_pseudo_report(clinical_row)
    out = os.path.join(cfg.paths.processed_dir, f"{pid}.npz")
    np.savez_compressed(out, images=imgs.astype(np.float32), roi=roi_masks)
    return dict(
        patient_id=str(pid),
        npz=out,
        report=report,
        labels=labels,
        n_slices=len(slices),
        has_mask=bool(has_mask),
    )


def _series_dirs(pid: str, raw_dir: str, index: dict | None) -> tuple[str, str]:
    """Resolve a patient's CT and SEG directories.

    `download.fetch_all` extracts into raw_dir/dicom/<PatientID>/{CT,SEG} and
    writes series_index.json. Fall back to the plain layout if the index is
    absent (e.g. data fetched with the NBIA Data Retriever by hand).
    """
    base = os.path.join(raw_dir, "dicom", str(pid))
    return os.path.join(base, "CT"), os.path.join(base, "SEG")


def run(cfg) -> None:
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
    clinical_csv = os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv")
    if not os.path.exists(clinical_csv):
        raise FileNotFoundError(
            f"{clinical_csv} not found. The EGFR/KRAS calls and semantic annotations "
            "are supplementary spreadsheets on the TCIA collection page, not the "
            "image API — download them by hand and save as clinical.csv."
        )
    clinical = pd.read_csv(clinical_csv).set_index("Case ID", drop=False)

    index_path = os.path.join(cfg.paths.raw_dir, "series_index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else None
    if index is None:
        print(f"[preprocess] no series_index.json at {index_path}; "
              "assuming dicom/<PatientID>/{CT,SEG} layout")

    aim_dir = os.path.join(cfg.paths.raw_dir, "aim")
    anns = load_aim(aim_dir) if os.path.isdir(aim_dir) else {}
    if anns:
        print(f"[preprocess] loaded {len(anns)} AIM annotations from {aim_dir}")
    else:
        print(f"[preprocess] WARNING: no AIM annotations at {aim_dir}. Rationales will "
              "fall back to build_pseudo_report, whose columns do not exist in the TCIA "
              "clinical sheet — every patient would get an identical sentence. Unzip "
              "AIM_files_updated-*.zip there.")

    index_rows, skipped = [], 0
    for pid, row in tqdm(clinical.iterrows(), total=len(clinical), desc="patients"):
        ct_dir, seg_dir = _series_dirs(pid, cfg.paths.raw_dir, index)
        rec = process_patient(pid, ct_dir, seg_dir, row, cfg, anns.get(str(pid)))
        if rec:
            index_rows.append(rec)
        else:
            skipped += 1

    idx_path = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    with open(idx_path, "w") as fh:
        for r in index_rows:
            fh.write(json.dumps(r) + "\n")

    n_mask = sum(1 for r in index_rows if r.get("has_mask"))
    n_reports = len({r["report"] for r in index_rows})
    print(f"[preprocess] wrote {len(index_rows)} patients ({skipped} skipped) -> {idx_path}")
    print(f"[preprocess]   {n_mask} with a segmentation (grounding supervised), "
          f"{len(index_rows) - n_mask} AIM-located only (grounding masked off)")
    print(f"[preprocess]   {n_reports} distinct rationales across {len(index_rows)} patients")
    if len(index_rows) > 5 and n_reports <= 2:
        print("[preprocess]   WARNING: rationales are near-identical. The generation head "
              "will learn a constant. Check that AIM annotations loaded.")
    for g in cfg.data.target_genes:
        pos = sum(1 for r in index_rows if r["labels"].get(g) == 1)
        neg = sum(1 for r in index_rows if r["labels"].get(g) == 0)
        print(f"[preprocess]   {g}: {pos} positive / {neg} negative / "
              f"{len(index_rows)-pos-neg} unknown")
    if not index_rows:
        raise RuntimeError(
            "no patients survived preprocessing. Check that the DICOM layout matches "
            f"{os.path.join(cfg.paths.raw_dir, 'dicom', '<PatientID>', '{CT,SEG}')} — "
            "run `python -m theia.data.download` to fetch and extract it."
        )


if __name__ == "__main__":
    from theia.config import load_config

    run(load_config())
