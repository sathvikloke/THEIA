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

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


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
            context_factor: float = 1.0) -> tuple[int, int, int, int]:
    """Square crop box (y0, y1, x0, x1) centred on the tumor.

    Returned coordinates may fall outside the image; `crop_box` zero-pads rather
    than clipping, so the box stays square and the aspect ratio is preserved.
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
    """Render the TCIA controlled-vocabulary semantic annotation as a sentence."""
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


def process_patient(pid, ct_dir, seg_dir, clinical_row, cfg) -> dict | None:
    sitk = _sitk()
    try:
        ct = load_series(ct_dir)
        seg = load_series(seg_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[preprocess] skip {pid}: {exc}")
        return None

    vol = sitk.GetArrayFromImage(ct).astype(np.float32)
    mask = resample_mask_to(ct, seg)
    if mask.sum() < cfg.data.min_tumor_voxels:
        print(f"[preprocess] skip {pid}: tumor too small")
        return None

    center, width = cfg.data.hu_window
    vol = window_hu(vol, center, width)
    spacing = ct.GetSpacing()[0]
    margin_px = int(round(cfg.data.roi_margin_mm / max(spacing, 1e-3)))
    context = float(getattr(cfg.data, "context_factor", 1.0))

    slices = tumor_slices(mask, cfg.data.n_slices, cfg.data.slice_strategy)
    imgs, roi_masks = [], []
    for s in slices:
        m2d = (mask[s] > 0).astype(np.float32)
        # One box per slice, applied to BOTH the image and the mask, so the
        # grounding target lives in the same frame as the pixels.
        box = roi_box(m2d, vol[s].shape, margin_px, context)
        imgs.append(_resize(crop_box(vol[s], box), cfg.data.image_size))
        roi_masks.append(_resize(crop_box(m2d, box), cfg.data.image_size))
    imgs = np.stack(imgs)
    roi_masks = (np.stack(roi_masks) > 0.5).astype(np.float32)

    labels = {g: label_of(clinical_row, g) for g in cfg.data.target_genes}
    if all(v == -1 for v in labels.values()):
        print(f"[preprocess] skip {pid}: no known label for any target gene")
        return None

    out = os.path.join(cfg.paths.processed_dir, f"{pid}.npz")
    np.savez_compressed(out, images=imgs.astype(np.float32), roi=roi_masks)
    return dict(
        patient_id=str(pid),
        npz=out,
        report=build_pseudo_report(clinical_row),
        labels=labels,
        n_slices=len(slices),
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

    index_rows, skipped = [], 0
    for pid, row in tqdm(clinical.iterrows(), total=len(clinical), desc="patients"):
        ct_dir, seg_dir = _series_dirs(pid, cfg.paths.raw_dir, index)
        rec = process_patient(pid, ct_dir, seg_dir, row, cfg)
        if rec:
            index_rows.append(rec)
        else:
            skipped += 1

    idx_path = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    with open(idx_path, "w") as fh:
        for r in index_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[preprocess] wrote {len(index_rows)} patients ({skipped} skipped) -> {idx_path}")
    if not index_rows:
        raise RuntimeError(
            "no patients survived preprocessing. Check that the DICOM layout matches "
            f"{os.path.join(cfg.paths.raw_dir, 'dicom', '<PatientID>', '{CT,SEG}')} — "
            "run `python -m theia.data.download` to fetch and extract it."
        )


if __name__ == "__main__":
    from theia.config import load_config

    run(load_config())
