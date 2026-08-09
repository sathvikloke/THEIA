"""Read LIDC-IDRI radiologist annotations into tumour masks.

LIDC-IDRI does NOT ship DICOM SEG. It ships one XML per scan containing up to
four independent `readingSession` blocks — four radiologists who annotated the
same scan without seeing each other's work — and each nodule is a stack of
per-slice polygon contours keyed by SOP instance UID. `import_nbia` and
`preprocess_masks` handle SEG objects and would read none of this, so this
module exists to bridge the two.

The consensus decision is the substantive one, and it is a scientific choice
rather than a formatting detail:

    Four readers rarely agree exactly. A union mask is the most inclusive and
    the least reliable — it accepts anything any single reader circled, which
    for grounding supervision means training attention onto whatever one reader
    over-called. Requiring all four is the opposite failure: it discards the
    ambiguous periphery that makes a lesion a lesion, and drops nodules entirely
    when one reader missed them.

    Default here is >= 3 of 4, the LUNA16 convention, which is also what most
    published LIDC work uses. It is exposed as `min_agreement` because the right
    answer depends on what the mask is FOR, and because a hidden default that
    changes the supervision signal is precisely the kind of thing this project
    has been bitten by.

LIDC also marks lesions < 3 mm as `nonNodule` or as nodules without contours.
Those carry no usable extent and are excluded rather than given an invented one.

Run (after the DICOM tree is in place):
  python -m theia.data.lidc --lidc_root <retriever>/LIDC-IDRI \
      --out data/raw/lidc --min_agreement 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NS = re.compile(r"\{.*\}")


def _tag(el) -> str:
    return NS.sub("", el.tag)


def _find_all(root, name: str):
    return [el for el in root.iter() if _tag(el) == name]


def parse_xml(path: str) -> dict:
    """Return {sop_uid: [contour, ...]} where a contour is (reader_idx, Nx2 array).

    Reader index is the position of the readingSession, which is how the four
    annotators are distinguished; LIDC does not name them.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    per_slice: dict[str, list] = defaultdict(list)
    for r_idx, session in enumerate(_find_all(root, "readingSession")):
        for nodule in _find_all(session, "unblindedReadNodule"):
            for roi in _find_all(nodule, "roi"):
                uid_el = [e for e in roi.iter() if _tag(e) == "imageSOP_UID"]
                if not uid_el or not (uid_el[0].text or "").strip():
                    continue
                uid = uid_el[0].text.strip()
                # inclusion=FALSE marks an excluded region (e.g. a vessel the
                # reader carved out); treating it as tumour would grow the mask.
                inc = [e for e in roi.iter() if _tag(e) == "inclusion"]
                if inc and (inc[0].text or "").strip().upper() == "FALSE":
                    continue
                pts = []
                for em in _find_all(roi, "edgeMap"):
                    x = [e for e in em.iter() if _tag(e) == "xCoord"]
                    y = [e for e in em.iter() if _tag(e) == "yCoord"]
                    if x and y:
                        pts.append((int(float(x[0].text)), int(float(y[0].text))))
                if len(pts) >= 3:            # a polygon needs three points
                    per_slice[uid].append((r_idx, np.array(pts, dtype=np.int32)))
    return dict(per_slice)


def _rasterize(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill one polygon onto a boolean grid. (rows, cols) = (y, x)."""
    from matplotlib.path import Path as MplPath

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    # radius=0.5 includes boundary pixels, matching how a reader's traced
    # outline is normally interpreted.
    inside = MplPath(points).contains_points(grid, radius=0.5)
    return inside.reshape(h, w)


def consensus_mask(per_slice: dict, uid_to_index: dict, shape: tuple[int, int],
                   depth: int, min_agreement: int = 3) -> np.ndarray:
    """Build a [depth, H, W] mask keeping voxels >= min_agreement readers marked.

    Each reader contributes at most once per voxel however many contours they
    drew there, so a reader who outlined two adjacent nodules on one slice does
    not count twice toward agreement.
    """
    out = np.zeros((depth, *shape), dtype=np.uint8)
    for uid, contours in per_slice.items():
        k = uid_to_index.get(uid)
        if k is None or not (0 <= k < depth):
            continue
        by_reader: dict[int, np.ndarray] = {}
        for r_idx, pts in contours:
            filled = _rasterize(pts, shape)
            by_reader[r_idx] = filled | by_reader.get(r_idx, np.zeros(shape, bool))
        if not by_reader:
            continue
        votes = np.sum(np.stack(list(by_reader.values())), axis=0)
        out[k] = (votes >= min_agreement).astype(np.uint8)
    return out


def mask_for_scan(xml_path: str, ct_dir: str, min_agreement: int = 3) -> np.ndarray:
    """[depth, H, W] consensus mask aligned to the CT series in `ct_dir`.

    Slices are matched by SOP instance UID, never by index. LIDC contours carry
    the UID of the image they were drawn on, and the DICOM files in a directory
    do not arrive in acquisition order, so index matching would silently place
    every contour on the wrong slice -- the same class of error as pairing a
    mask with the wrong series.
    """
    from theia.data.preprocess import load_series, sop_uid_index

    ct = load_series(ct_dir)
    uid_to_index = sop_uid_index(ct_dir)
    depth, h, w = _volume_shape(ct)
    per_slice = parse_xml(xml_path)
    matched = sum(1 for uid in per_slice if uid in uid_to_index)
    if per_slice and matched == 0:
        raise ValueError(
            f"{xml_path}: none of its {len(per_slice)} annotated slices match a "
            f"SOP UID in {ct_dir}. The XML and the series belong to different scans.")
    return consensus_mask(per_slice, uid_to_index, (h, w), depth, min_agreement)


def _volume_shape(ct) -> tuple[int, int, int]:
    w, h, depth = ct.GetSize()          # SimpleITK reports (x, y, z)
    return depth, h, w


def summarise(xml_paths: list[str], min_agreement: int = 3) -> dict:
    """Reader-agreement statistics over a set of XMLs, without loading any CT.

    Worth running before committing to a consensus rule: it reports how many
    annotated slices survive each threshold, which is the difference between a
    large noisy pretraining cohort and a small clean one.
    """
    stats = {"scans": 0, "annotated_slices": 0, "by_readers": defaultdict(int)}
    for p in xml_paths:
        try:
            per_slice = parse_xml(p)
        except Exception:                                  # noqa: BLE001
            continue
        stats["scans"] += 1
        for _uid, contours in per_slice.items():
            stats["annotated_slices"] += 1
            stats["by_readers"][len({r for r, _ in contours})] += 1
    stats["by_readers"] = dict(sorted(stats["by_readers"].items()))
    kept = sum(v for k, v in stats["by_readers"].items() if k >= min_agreement)
    stats["kept_at_threshold"] = kept
    stats["kept_frac"] = kept / max(stats["annotated_slices"], 1)
    stats["min_agreement"] = min_agreement
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidc_root", required=True,
                    help="directory holding LIDC-IDRI, as the TCIA Data Retriever left it")
    ap.add_argument("--min_agreement", type=int, default=3,
                    help="readers who must agree on a voxel (LUNA16 uses 3 of 4)")
    ap.add_argument("--summarise_only", action="store_true",
                    help="report reader agreement and exit, without touching CT")
    ap.add_argument("--masks_out", default=None,
                    help="write per-scan consensus masks here (.npz per scan)")
    ap.add_argument("--out", default="results/lidc_agreement.json")
    a = ap.parse_args()

    xmls = sorted(glob.glob(os.path.join(a.lidc_root, "**", "*.xml"), recursive=True))
    if not xmls:
        raise SystemExit(
            f"no XML annotations under {a.lidc_root}. LIDC-IDRI ships one XML per "
            "scan alongside the DICOM series; check the Data Retriever output path.")
    print(f"[lidc] {len(xmls)} annotation files")
    stats = summarise(xmls, a.min_agreement)
    print(f"[lidc] {stats['scans']} scans, {stats['annotated_slices']} annotated slices")
    for k, v in stats["by_readers"].items():
        print(f"[lidc]   {k} reader(s) agreed on {v} slice(s)")
    print(f"[lidc] at >= {a.min_agreement}: {stats['kept_at_threshold']} slices kept "
          f"({100*stats['kept_frac']:.0f}%)")
    Path(os.path.dirname(a.out) or ".").mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(a.out, "w"), indent=2)
    print(f"[lidc] wrote {a.out}")

    if a.summarise_only:
        print("[lidc] --summarise_only: stopping before any CT is read")
        return
    if not a.masks_out:
        print("[lidc] no --masks_out given; nothing further to do. Pass it to write "
              "per-scan consensus masks for grounding pretraining.")
        return

    out_dir = Path(a.masks_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    for xml in xmls:
        scan_dir = os.path.dirname(xml)
        try:
            m = mask_for_scan(xml, scan_dir, a.min_agreement)
        except Exception as exc:                              # noqa: BLE001
            print(f"[lidc] skip {os.path.basename(scan_dir)}: "
                  f"{type(exc).__name__}: {exc}")
            skipped += 1
            continue
        if int(m.sum()) == 0:
            skipped += 1
            continue
        np.savez_compressed(out_dir / f"{Path(scan_dir).name}.npz", mask=m)
        written += 1
    print(f"[lidc] wrote {written} consensus mask(s) to {out_dir} ({skipped} skipped)")
    print("[lidc] next: python -m theia.data.preprocess_masks --raw_dir <tree> "
          f"--out_dir data/processed_lidc  (masks from {out_dir})")


if __name__ == "__main__":
    main()
