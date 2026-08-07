"""Label-free preprocessing for grounding-pretraining cohorts.

NSCLC-Radiomics (421 patients) and LIDC-IDRI (875) have CT plus a tumor
segmentation but no mutation calls. They are useless for the classifier and
ideal for the grounding head, which only needs an image and a mask.

`theia.data.preprocess.run` is driven by the clinical spreadsheet and drops any
patient with no known label, so it cannot be used here. This walks the imported
DICOM tree instead and emits the same `rows.jsonl` + `.npz` contract with an
empty `labels` dict. `RadiogenomicsDataset` already returns -1 for any gene it
cannot find, and `classification_loss` masks -1, so these rows are inert for
every head except grounding.

Run:
  python -m theia.data.import_nbia --nbia_root <retriever>/nsclc_radiomics \
      --out data/raw/nsclc_radiomics
  python -m theia.data.preprocess_masks --raw_dir data/raw/nsclc_radiomics \
      --out_dir data/processed_pretrain
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import glob

from theia.data.preprocess import (_resize, crop_box, load_series, roi_box,
                                   segment_mask, tumor_slices, window_hu)


def process_patient(pid: str, ct_dir: str, seg_dir: str, cfg) -> dict | None:
    import SimpleITK as sitk

    try:
        ct = load_series(ct_dir)
        segf = sorted(glob.glob(os.path.join(seg_dir, "*.dcm")))
        if not segf:
            raise FileNotFoundError(f"no SEG in {seg_dir}")
        # segment_mask, not resample_mask_to: planning SEGs bundle lungs, cord,
        # heart and esophagus alongside the tumor, and thresholding the whole
        # object at >0 would make "tumor" mean the entire thorax.
        mask = segment_mask(segf[0], ct)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[pretrain-prep] skip {pid}: {exc}")
        return None
    if mask.sum() < cfg.data.min_tumor_voxels:
        print(f"[pretrain-prep] skip {pid}: tumor too small ({int(mask.sum())} voxels)")
        return None

    vol = sitk.GetArrayFromImage(ct).astype(np.float32)
    center, width = cfg.data.hu_window
    vol = window_hu(vol, center, width)
    spacing = ct.GetSpacing()[0]
    margin_px = int(round(cfg.data.roi_margin_mm / max(spacing, 1e-3)))
    context = float(getattr(cfg.data, "context_factor", 1.0))
    jitter = float(getattr(cfg.data, "crop_jitter_frac", 0.0))
    seed = abs(hash(str(pid))) % (2 ** 32)

    slices = tumor_slices(mask, cfg.data.n_slices, cfg.data.slice_strategy)
    imgs, rois = [], []
    for s in slices:
        m2d = (mask[s] > 0).astype(np.float32)
        # One offset per patient, same box for image and mask — the same
        # invariant the main preprocessor enforces.
        box = roi_box(m2d, vol[s].shape, margin_px, context, jitter,
                      np.random.default_rng(seed))
        imgs.append(_resize(crop_box(vol[s], box), cfg.data.image_size))
        rois.append(_resize(crop_box(m2d, box), cfg.data.image_size))

    out = os.path.join(cfg.paths.processed_dir, f"{pid}.npz")
    np.savez_compressed(out, images=np.stack(imgs).astype(np.float32),
                        roi=(np.stack(rois) > 0.5).astype(np.float32))
    return dict(patient_id=str(pid), npz=out, report="", labels={},
                n_slices=len(slices), has_mask=True)


def run(cfg, raw_dir: str) -> None:
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
    dicom_root = Path(raw_dir) / "dicom"
    if not dicom_root.is_dir():
        raise FileNotFoundError(
            f"{dicom_root} not found — run theia.data.import_nbia first to resolve "
            "each patient's CT against its segmentation.")

    patients = sorted(p.name for p in dicom_root.iterdir() if p.is_dir())
    rows, skipped = [], 0
    for pid in tqdm(patients, desc="pretrain patients"):
        rec = process_patient(pid, str(dicom_root / pid / "CT"),
                              str(dicom_root / pid / "SEG"), cfg)
        if rec:
            rows.append(rec)
        else:
            skipped += 1

    idx = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    with open(idx, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[pretrain-prep] wrote {len(rows)} patients ({skipped} skipped) -> {idx}")
    if not rows:
        raise RuntimeError("no patients survived; check the import step")


if __name__ == "__main__":
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/processed_pretrain")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["paths"]["processed_dir"] = a.out_dir
    run(cfg, a.raw_dir)
