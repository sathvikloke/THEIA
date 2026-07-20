"""TCIA download helper for NSCLC-RADIOGENOMICS.

TCIA requires accepting the Data Usage Policy and (for images) the NBIA Data
Retriever or the REST API with a token. This module wraps the REST API for the
metadata + segmentation manifests and shells out to the retriever for images.
It does NOT bypass any access control — you still need a registered account.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "NSCLC-RADIOGENOMICS"


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
    with s.get(
        f"{TCIA_BASE}/getImage",
        params={"SeriesInstanceUID": series_uid},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return str(dest_path)


def fetch_all(raw_dir: str) -> None:
    """List then download every CT/SEG series. Resumable: skips existing zips."""
    manifest = os.path.join(raw_dir, "series_manifest.csv")
    series = list_series(manifest)
    img_dir = os.path.join(raw_dir, "dicom")
    for row in tqdm(series, desc="TCIA series"):
        uid = row["SeriesInstanceUID"]
        if os.path.exists(os.path.join(img_dir, f"{uid}.zip")):
            continue
        try:
            download_series(uid, img_dir)
        except requests.HTTPError as exc:
            print(f"[download] WARN {uid}: {exc}")
    # TODO(theia): clinical labels (EGFR/KRAS) + semantic annotations ship as
    # supplementary spreadsheets on the collection page, not the image API.
    # Download those manually once and drop them in raw_dir/clinical/.


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
    fetch_all(ap.parse_args().raw_dir)
