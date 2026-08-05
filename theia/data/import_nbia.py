"""Normalise an NBIA Data Retriever download into the layout preprocess expects.

The Retriever writes `<root>/<PatientID>/<study-hash>/<series-hash>/*.dcm`, with
no modality or description in the path and a `metadata.csv` that carries only
UIDs. `preprocess` wants `dicom/<PatientID>/{CT,SEG}`. This bridges the two by
reading one header per series, choosing the right CT, and symlinking — never
copying, because the download is ~61 GB.

Choosing the CT is the part that matters
----------------------------------------
A patient has several CT series and only one is correct. Measured on the real
download:

  R01-018   .625 mm Chest      580 slices   axial      <- SEG references this
            CT_SLICES          257 slices   axial
  R01-011   Recon 2: CHEST     241 slices   axial      <- SEG references this
            NONCONTRAST ACCT   223 slices   axial
            PET_CT_BODY          1 slice    LOCALIZER

Series descriptions are inconsistent across patients ('.625 mm Chest',
'CHEST 1.25 MM SHARP', 'THORAX LUNG 2MM'), so no string heuristic is safe, and
'most slices' is a guess. The DICOM SEG object states which series it was drawn
on in `ReferencedSeriesSequence`; that is authoritative and is used first.
Everything else is a fallback, and every decision is logged with its reason so
the selection can be audited rather than trusted.

Coronal reformats are the dangerous case: same anatomy, plausible slice count,
wrong plane. `tumor_slices` walks axis 0 assuming axial, so a reformat produces
silently wrong crops. They are rejected on `ImageOrientationPatient`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Axial direction cosines are (1,0,0, 0,1,0). Allow a little obliquity.
AXIAL_TOL = 0.10


@dataclass
class SeriesInfo:
    path: str
    patient_id: str
    series_uid: str
    modality: str
    description: str
    n_files: int
    axial: bool = True
    localizer: bool = False
    derived: bool = False
    referenced_series: str | None = None


def _is_axial(iop) -> bool:
    if iop is None:
        return True  # SEG objects carry no IOP at the top level
    try:
        v = [float(x) for x in iop]
    except (TypeError, ValueError):
        return True
    return (abs(v[0] - 1) < AXIAL_TOL and abs(v[1]) < AXIAL_TOL and abs(v[2]) < AXIAL_TOL
            and abs(v[3]) < AXIAL_TOL and abs(v[4] - 1) < AXIAL_TOL and abs(v[5]) < AXIAL_TOL)


def scan(root: str) -> dict[str, list[SeriesInfo]]:
    """Read one header per series directory. Returns {patient_id: [SeriesInfo]}."""
    import pydicom

    out: dict[str, list[SeriesInfo]] = defaultdict(list)
    series_dirs = sorted({os.path.dirname(f) for f in
                          glob.glob(os.path.join(root, "*", "*", "*", "*.dcm"))})
    for d in series_dirs:
        files = sorted(glob.glob(os.path.join(d, "*.dcm")))
        if not files:
            continue
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[import] unreadable {d}: {exc}")
            continue
        image_type = [str(x).upper() for x in (getattr(ds, "ImageType", None) or [])]
        info = SeriesInfo(
            path=d,
            patient_id=str(getattr(ds, "PatientID", Path(d).parts[-3])),
            series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
            modality=str(getattr(ds, "Modality", "")).upper(),
            description=str(getattr(ds, "SeriesDescription", "")),
            n_files=len(files),
            axial=_is_axial(getattr(ds, "ImageOrientationPatient", None)),
            localizer="LOCALIZER" in image_type,
            derived="DERIVED" in image_type,
        )
        if info.modality == "SEG":
            try:
                info.referenced_series = str(
                    ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
            except Exception:  # noqa: BLE001
                info.referenced_series = None
        out[info.patient_id].append(info)
    return dict(out)


def choose_ct(cands: list[SeriesInfo], referenced: str | None) -> tuple[SeriesInfo | None, str]:
    """Pick the diagnostic axial CT. Returns (series, reason)."""
    usable = [c for c in cands if c.modality == "CT" and not c.localizer and c.n_files > 1]
    rejected_plane = [c for c in usable if not c.axial]
    usable = [c for c in usable if c.axial]
    if not usable:
        return None, ("all CT series rejected"
                      + (f" ({len(rejected_plane)} non-axial)" if rejected_plane else ""))

    if referenced:
        for c in usable:
            if c.series_uid == referenced:
                return c, "referenced by SEG"

    # No usable reference: prefer a non-fusion, non-derived series, then slice count.
    def rank(c: SeriesInfo):
        fusion = "fusion" in c.description.lower() or "pet" in c.description.lower()
        return (fusion, c.derived, -c.n_files)

    best = sorted(usable, key=rank)[0]
    return best, "most slices, axial, non-localizer (SEG reference unavailable)"


def build(root: str, out_dir: str, link: bool = True) -> dict:
    """Create dicom/<PatientID>/{CT,SEG} pointing at the chosen series."""
    scanned = scan(root)
    dicom_root = Path(out_dir) / "dicom"
    dicom_root.mkdir(parents=True, exist_ok=True)

    log, n_ok, n_ct_only = [], 0, 0
    for pid, series in sorted(scanned.items()):
        segs = [s for s in series if s.modality == "SEG"]
        seg = max(segs, key=lambda s: s.n_files) if segs else None
        ct, reason = choose_ct(series, seg.referenced_series if seg else None)
        entry = dict(patient_id=pid, ct_reason=reason,
                     ct_description=ct.description if ct else None,
                     ct_slices=ct.n_files if ct else 0,
                     has_seg=seg is not None,
                     n_ct_candidates=sum(1 for s in series if s.modality == "CT"))
        if ct is None:
            entry["status"] = "skipped: no usable CT"
            log.append(entry)
            continue

        pdir = dicom_root / pid
        pdir.mkdir(parents=True, exist_ok=True)
        for name, src in (("CT", ct.path), ("SEG", seg.path if seg else None)):
            if src is None:
                continue
            dst = pdir / name
            if dst.is_symlink() or dst.exists():
                dst.unlink() if dst.is_symlink() else None
            if link:
                dst.symlink_to(os.path.abspath(src))
            else:
                os.makedirs(dst, exist_ok=True)
        entry["status"] = "ok" if seg else "ok (CT only, no SEG)"
        n_ok += 1
        n_ct_only += 0 if seg else 1
        log.append(entry)

    log_path = Path(out_dir) / "import_log.json"
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=2)

    by_reason: dict[str, int] = defaultdict(int)
    for e in log:
        by_reason[e["ct_reason"]] += 1
    print(f"[import] {n_ok}/{len(scanned)} patients linked -> {dicom_root}")
    print(f"[import]   {n_ok - n_ct_only} with SEG, {n_ct_only} CT-only")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"[import]   {n:4d}  {reason}")
    print(f"[import] per-patient decisions -> {log_path}")
    return dict(patients=n_ok, with_seg=n_ok - n_ct_only, log=str(log_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbia_root", required=True,
                    help="the folder containing <PatientID>/ dirs from the Retriever")
    ap.add_argument("--out", required=True, help="cfg.paths.raw_dir")
    a = ap.parse_args()
    build(a.nbia_root, a.out)
