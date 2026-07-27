"""TCIA download helper for NSCLC-RADIOGENOMICS.

TCIA requires accepting the Data Usage Policy and (for images) the NBIA Data
Retriever or the REST API with a token. This module wraps the REST API for the
metadata + segmentation manifests and shells out to the retriever for images.
It does NOT bypass any access control — you still need a registered account.

Layout note (this was broken)
-----------------------------
`fetch_all` used to leave zips at `raw_dir/dicom/<SeriesInstanceUID>.zip` while
`preprocess.run` looked for extracted per-patient folders at
`raw_dir/dicom/<PatientID>/CT` and `/SEG`. Nothing unzipped anything and nothing
mapped a UID to a patient, so the documented quickstart downloaded the whole
collection and then skipped every single patient with "no DICOM series",
producing an empty rows.jsonl. Downloads are now extracted into the layout
preprocess expects, and a `series_index.json` records the mapping.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "NSCLC-RADIOGENOMICS"

# Modalities we keep: the CT volume and the tumor segmentation.
CT_MODALITIES = {"CT"}
SEG_MODALITIES = {"SEG", "RTSTRUCT"}


def _session() -> requests.Session:
    load_dotenv()
    s = requests.Session()
    token = os.environ.get("TCIA_API_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def list_series(out_csv: str) -> list[dict]:
    """Fetch the series manifest for the collection and write it to CSV."""
    s = _session()
    resp = s.get(f"{TCIA_BASE}/getSeries", params={"Collection": COLLECTION}, timeout=60)
    resp.raise_for_status()
    series = resp.json()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    if series:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(series[0].keys()))
            writer.writeheader()
            writer.writerows(series)
    print(f"[download] {len(series)} series listed -> {out_csv}")
    return series


def download_series(series_uid: str, dest: str) -> str:
    """Download one DICOM series as a zip via the TCIA REST API."""
    s = _session()
    dest_path = Path(dest) / f"{series_uid}.zip"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(".zip.part")
    with s.get(
        f"{TCIA_BASE}/getImage",
        params={"SeriesInstanceUID": series_uid},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    # Rename only after a complete download so an interrupted run does not leave
    # a truncated zip that the resume check would happily skip.
    tmp.replace(dest_path)
    return str(dest_path)


def _kind(modality: str) -> str | None:
    m = (modality or "").upper()
    if m in CT_MODALITIES:
        return "CT"
    if m in SEG_MODALITIES:
        return "SEG"
    return None


def extract_series(zip_path: str, patient_id: str, kind: str, raw_dir: str) -> str:
    """Unpack one series zip into raw_dir/dicom/<PatientID>/<CT|SEG>/."""
    out = Path(raw_dir) / "dicom" / str(patient_id) / kind
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            # Flatten and sanitise: never trust archive paths.
            target = out / Path(member).name
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return str(out)


def fetch_all(raw_dir: str, extract: bool = True) -> dict:
    """List, download, and extract every CT/SEG series. Resumable."""
    manifest = os.path.join(raw_dir, "series_manifest.csv")
    series = list_series(manifest)
    zip_dir = os.path.join(raw_dir, "zips")
    index: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"CT": [], "SEG": []})

    for row in tqdm(series, desc="TCIA series"):
        uid = row["SeriesInstanceUID"]
        kind = _kind(row.get("Modality", ""))
        if kind is None:
            continue
        pid = str(row.get("PatientID") or row.get("PatientId") or "").strip()
        if not pid:
            print(f"[download] WARN {uid}: no PatientID in manifest, skipping")
            continue

        zip_path = os.path.join(zip_dir, f"{uid}.zip")
        if not os.path.exists(zip_path):
            try:
                download_series(uid, zip_dir)
            except requests.HTTPError as exc:
                print(f"[download] WARN {uid}: {exc}")
                continue
        if extract:
            try:
                extract_series(zip_path, pid, kind, raw_dir)
            except zipfile.BadZipFile:
                print(f"[download] WARN {uid}: corrupt zip, re-download it")
                continue
        index[pid][kind].append(uid)

    index_path = os.path.join(raw_dir, "series_index.json")
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2)

    usable = sum(1 for v in index.values() if v["CT"] and v["SEG"])
    print(f"[download] {len(index)} patients indexed, {usable} with both CT and SEG "
          f"-> {index_path}")
    print("[download] NOTE: clinical labels (EGFR/KRAS) and semantic annotations ship as "
          "supplementary spreadsheets on the collection page, not the image API. "
          f"Download those once by hand into {os.path.join(raw_dir, 'clinical', 'clinical.csv')}.")
    return dict(index)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/nsclc_radiogenomics")
    ap.add_argument("--no_extract", action="store_true")
    a = ap.parse_args()
    fetch_all(a.raw_dir, extract=not a.no_extract)
