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

    The project's existing default is >= 3 distinct readers per voxel. This is
    a voxel mask rule, not a clinical nodule eligibility rule. It is exposed as
    `min_agreement` because changing it changes the supervision signal.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NS = re.compile(r"\{.*\}")


@dataclass(frozen=True)
class Contour:
    """One XML ROI; exclusions belong to one reader's particular nodule."""

    reader_idx: int
    points: np.ndarray
    inclusion: bool
    nodule_idx: int

    def __iter__(self):
        # Preserve the reader/points unpacking used by earlier parser callers.
        yield self.reader_idx
        yield self.points


def _tag(el) -> str:
    return NS.sub("", el.tag)


def _find_all(root, name: str):
    return [el for el in root.iter() if _tag(el) == name]


def parse_xml(path: str) -> dict:
    """Return {sop_uid: [Contour, ...]}, retaining inclusion and nodule identity.

    Reader index is the position of the readingSession, which is how the four
    annotators are distinguished; LIDC does not name them.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    per_slice: dict[str, list] = defaultdict(list)
    for r_idx, session in enumerate(_find_all(root, "readingSession")):
        for n_idx, nodule in enumerate(_find_all(session, "unblindedReadNodule")):
            for roi in _find_all(nodule, "roi"):
                uid_el = [e for e in roi.iter() if _tag(e) == "imageSOP_UID"]
                if not uid_el or not (uid_el[0].text or "").strip():
                    continue
                uid = uid_el[0].text.strip()
                # Keep exclusions: dropping them fills holes inside an included
                # outline. They must be subtracted from this nodule before votes.
                inc = [e for e in roi.iter() if _tag(e) == "inclusion"]
                inclusion = not (inc and (inc[0].text or "").strip().upper() == "FALSE")
                pts = []
                for em in _find_all(roi, "edgeMap"):
                    x = [e for e in em.iter() if _tag(e) == "xCoord"]
                    y = [e for e in em.iter() if _tag(e) == "yCoord"]
                    if x and y:
                        pts.append((int(float(x[0].text)), int(float(y[0].text))))
                if len(pts) >= 3:            # a polygon needs three points
                    per_slice[uid].append(Contour(
                        r_idx, np.array(pts, dtype=np.int32), inclusion, n_idx))
    return dict(per_slice)


def _rasterize(points: np.ndarray, shape: tuple[int, int],
               include_boundary: bool = False) -> np.ndarray:
    """Fill strict polygon interior, or interior plus border for exclusions.

    LIDC contour pixels lie outside the nodule. Use even-odd ray crossing and
    explicitly detect every grid point on an edge, avoiding orientation-dependent
    boundary behavior from polygon libraries. Coordinates are integer pixels;
    cross products are exact. Work only in the polygon's clipped bounding box.
    """
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or not np.equal(points, np.floor(points)).all()):
        raise ValueError("contour must contain at least three finite integer pixel points")
    points = points.astype(np.int64)
    h, w = shape
    result = np.zeros(shape, dtype=bool)
    x0, y0 = np.maximum(points.min(axis=0), (0, 0))
    x1, y1 = np.minimum(points.max(axis=0), (w - 1, h - 1))
    if x0 > x1 or y0 > y1:
        return result
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.zeros(xx.shape, dtype=bool)
    boundary = np.zeros(xx.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(points, np.roll(points, -1, axis=0)):
        dx, dy = bx - ax, by - ay
        boundary |= (((xx - ax) * dy == (yy - ay) * dx)
                     & (xx >= min(ax, bx)) & (xx <= max(ax, bx))
                     & (yy >= min(ay, by)) & (yy <= max(ay, by)))
        if dy:
            crosses = (ay > yy) != (by > yy)
            intersection_x = ax + (yy - ay) * dx / dy
            inside ^= crosses & (xx < intersection_x)
    result[y0:y1 + 1, x0:x1 + 1] = (
        inside | boundary if include_boundary else inside & ~boundary)
    return result


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
        by_nodule: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for contour in contours:
            r_idx, pts = contour
            include = getattr(contour, "inclusion", True)
            key = r_idx, getattr(contour, "nodule_idx", 0)
            if key not in by_nodule:
                by_nodule[key] = np.zeros(shape, bool), np.zeros(shape, bool)
            filled = _rasterize(pts, shape, include_boundary=not include)
            by_nodule[key][0 if include else 1][:] |= filled
        by_reader: dict[int, np.ndarray] = {}
        for (r_idx, _), (included, excluded) in by_nodule.items():
            nodule = included & ~excluded
            by_reader[r_idx] = nodule | by_reader.get(r_idx, np.zeros(shape, bool))
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


def export_mask_for_scan(xml_path: str, ct_dir: str, out_dir: str,
                         min_agreement: int = 3) -> Path:
    """Export a mask with the exact CT slice order and geometry that produced it.

    The archive can subsequently be verified by ``theia.data.lidc_rows``.
    Mask-only legacy exports cannot establish that correspondence.
    """
    import hashlib
    from theia.data.lidc_rows import (CONSENSUS_METHOD, SCHEMA, _sha256, read_source_series,
                                     write_consensus_npz)

    if type(min_agreement) is not int or not 1 <= min_agreement <= 4:
        raise ValueError("min_agreement must be an integer from 1 to 4")
    per_slice = parse_xml(xml_path)
    if not any(contour.inclusion for contours in per_slice.values() for contour in contours):
        raise ValueError(f"{xml_path}: empty annotation source (no included polygons)")
    source = read_source_series(ct_dir, required_sops=per_slice)
    metadata = dict(source.metadata)
    sops = metadata["sop_instance_uids"]
    depth, h, w = source.volume_hu.shape
    mask = consensus_mask(per_slice, {uid: i for i, uid in enumerate(sops)},
                          (h, w), depth, min_agreement)
    metadata.update(
        schema=SCHEMA, mask_axes="zyx", min_agreement=min_agreement,
        annotated_sop_instance_uids=sorted(per_slice),
        reader_indices=sorted({reader for contours in per_slice.values()
                               for reader, _ in contours}),
        source_xml=str(Path(xml_path).resolve()),
        source_xml_sha256=_sha256(xml_path),
        consensus_method=CONSENSUS_METHOD,
    )
    name = "lidc_" + hashlib.sha256(
        metadata["series_instance_uid"].encode()).hexdigest()[:20] + ".npz"
    path = Path(out_dir) / name
    write_consensus_npz(path, mask, metadata)
    return path


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
            readers = {contour.reader_idx for contour in contours if contour.inclusion}
            if not readers:
                continue
            stats["annotated_slices"] += 1
            stats["by_readers"][len(readers)] += 1
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
    ap.add_argument("--min_agreement", type=int, choices=range(1, 5), default=3,
                    help="distinct readers who must agree on a voxel (default: 3)")
    ap.add_argument("--summarise_only", action="store_true",
                    help="report reader agreement and exit, without touching CT")
    ap.add_argument("--masks_out", default=None,
                    help="write per-scan consensus masks here (.npz per scan)")
    ap.add_argument("--limit", type=int, help="bounded pilot: first N sorted XMLs only")
    ap.add_argument("--out", default="results/lidc_agreement.json")
    a = ap.parse_args()

    xmls = sorted(glob.glob(os.path.join(a.lidc_root, "**", "*.xml"), recursive=True))
    if not xmls:
        raise SystemExit(
            f"no XML annotations under {a.lidc_root}. LIDC-IDRI ships one XML per "
            "scan alongside the DICOM series; check the Data Retriever output path.")
    if a.limit is not None:
        if a.limit < 1:
            ap.error("--limit must be positive")
        xmls = xmls[:a.limit]
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
            export_mask_for_scan(xml, scan_dir, str(out_dir), a.min_agreement)
        except Exception as exc:                              # noqa: BLE001
            print(f"[lidc] skip {os.path.basename(scan_dir)}: "
                  f"{type(exc).__name__}: {exc}")
            skipped += 1
            continue
        written += 1
    print(f"[lidc] wrote {written} consensus mask(s) to {out_dir} ({skipped} skipped)")
    if not written:
        raise RuntimeError("no usable matched consensus sources were exported")
    print("[lidc] next: python -m theia.data.lidc_rows "
          f"--masks_dir {out_dir} --out_dir data/processed_lidc "
          f"--expected_min_agreement {a.min_agreement}")
    if skipped:
        raise RuntimeError(f"{skipped} consensus source(s) failed export; inspect the "
                           "messages above before treating this as a complete cohort")


if __name__ == "__main__":
    main()
