"""DICOM -> model-ready tensors.

Pipeline per patient:
  1. load the CT series and the tumor SEG series (RTSTRUCT/SEG) via SimpleITK
  2. resample SEG onto the CT grid, threshold to a binary tumor mask
  3. HU-window the CT, take the slices spanning the tumor bounding box
  4. crop a square ROI around the (margin-dilated) tumor
  5. attach the EGFR/KRAS labels and build the semantic-annotation pseudo-report
  6. write one .npz per patient + a rows.jsonl index

The pseudo-report is the controlled-vocabulary semantic annotation rendered as a
sentence. It is the supervision signal for the generation head — you never write
free-text reports by hand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm


def load_series(dicom_dir: str) -> sitk.Image:
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


def resample_mask_to(ct: sitk.Image, seg: sitk.Image) -> np.ndarray:
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
    lo, hi = int(present.min()), int(present.max())
    if hi - lo + 1 <= n_slices:
        idx = list(range(lo, hi + 1))
        return idx + [hi] * (n_slices - len(idx))
    return list(np.linspace(lo, hi, n_slices).round().astype(int))


def crop_roi(img2d: np.ndarray, mask2d: np.ndarray, margin_px: int, size: int) -> np.ndarray:
    if mask2d.sum() == 0:
        ys, xs = np.array([img2d.shape[0] // 2]), np.array([img2d.shape[1] // 2])
    else:
        ys, xs = np.where(mask2d > 0)
    y0, y1 = max(ys.min() - margin_px, 0), min(ys.max() + margin_px, img2d.shape[0])
    x0, x1 = max(xs.min() - margin_px, 0), min(xs.max() + margin_px, img2d.shape[1])
    patch = img2d[y0:y1, x0:x1]
    return _resize(patch, size)


def _resize(patch: np.ndarray, size: int) -> np.ndarray:
    if patch.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    zoom = (size / patch.shape[0], size / patch.shape[1])
    return ndimage.zoom(patch, zoom, order=1).astype(np.float32)


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

    slices = tumor_slices(mask, cfg.data.n_slices, cfg.data.slice_strategy)
    imgs = np.stack([crop_roi(vol[s], mask[s], margin_px, cfg.data.image_size) for s in slices])
    roi_masks = np.stack(
        [_resize((mask[s] > 0).astype(np.float32), cfg.data.image_size) for s in slices]
    )

    labels = {g: label_of(clinical_row, g) for g in cfg.data.target_genes}
    out = os.path.join(cfg.paths.processed_dir, f"{pid}.npz")
    np.savez_compressed(out, images=imgs.astype(np.float32), roi=roi_masks.astype(np.float32))
    return dict(
        patient_id=str(pid),
        npz=out,
        report=build_pseudo_report(clinical_row),
        labels=labels,
        n_slices=len(slices),
    )


def run(cfg) -> None:
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
    clinical_csv = os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv")
    clinical = pd.read_csv(clinical_csv).set_index("Case ID", drop=False)
    index_rows = []
    for pid, row in tqdm(clinical.iterrows(), total=len(clinical), desc="patients"):
        # TODO(theia): map pid -> its CT dir and SEG dir using series_manifest.csv
        ct_dir = os.path.join(cfg.paths.raw_dir, "dicom", str(pid), "CT")
        seg_dir = os.path.join(cfg.paths.raw_dir, "dicom", str(pid), "SEG")
        rec = process_patient(pid, ct_dir, seg_dir, row, cfg)
        if rec:
            index_rows.append(rec)
    idx_path = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    with open(idx_path, "w") as fh:
        for r in index_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[preprocess] wrote {len(index_rows)} patients -> {idx_path}")


if __name__ == "__main__":
    from theia.config import load_config

    run(load_config())
