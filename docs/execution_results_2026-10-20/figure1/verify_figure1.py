#!/usr/bin/env python3
"""Verify aggregate arithmetic, source identity, rendered counts and PNG bytes."""

import argparse
import csv
import datetime
import hashlib
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
from build_figure1 import validate

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    data = json.loads((HERE / "SOURCE_DATA.json").read_text())
    checks = validate(data)
    rows = {row["id"]: row for row in data["rows"]}
    with (HERE / "SOURCE_DATA.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    for row in csv_rows:
        row["count"] = int(row["count"])
    checks["csv_json_identical"] = csv_rows == data["rows"]

    svg = ET.parse(HERE / "FIGURE1.svg").getroot()
    rendered = {}
    for element in svg.iter():
        if "data-row-id" in element.attrib:
            key = element.attrib["data-row-id"]
            count = int(element.attrib["data-count"])
            displayed = int(re.match(r"^(\d+)", element.text).group(1))
            assert count == rows[key]["count"] == displayed, (key, count, displayed)
            rendered[key] = count
    required_rows = {
        "i_source", "i_imported", "i_not_imported", "i_no_download_no_aim",
        "i_unmatched_aim", "i_ct_seg", "i_aim_only", "i_unbuilt",
        "i_unknown_both", "i_air", "i_approx_order", "i_scored",
        "i_extent", "i_point", "i_egfr_unknown", "i_labeled", "i_mutant",
        "e_source", "e_no_location", "e_ineligible_scan", "e_air_adapter",
        "e_development", "e_unknown", "e_evaluated", "e_positive",
        "e_research_negative", "r_paired", "c_chosen", "c_eligible",
    }
    checks["required_rendered_counts_present"] = required_rows <= rendered.keys()
    text = " ".join(svg.itertext())
    required_phrases = [
        "Selection reason unrecorded", "Reconstructed air-rule exclusions",
        "approximate slice order (IPP)", "No original preprocessing log was saved",
        "EGFR and KRAS unknown", "Skipped before localization",
        "Excluded from EGFR metrics; never negative",
        "No clinical-assay negative is represented",
        "do not establish a deduplicated union of persons",
        "Preparation only", "No molecular label",
        "Publication adoption pending student / author review",
        "7 mislocated point inputs (5 EGFR-labelled)",
    ]
    checks["required_limitations_visible"] = all(phrase in text for phrase in required_phrases)
    checks["no_193_union_count"] = "193" not in text
    checks["no_unverified_gdcm_claim"] = "GDCM" not in text
    checks["no_invented_internal_intermediate"] = "162" not in text
    png = (HERE / "FIGURE1.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "Not a PNG"
    width, height = struct.unpack(">II", png[16:24])
    checks["png_3200x3280"] = (width, height) == (3200, 3280)
    checks["exact_source_caption_retained"] = data["excerpts"]["source_caption"]["text"] in (HERE / "CAPTION.md").read_text()
    local_links = []
    for filename in ["CAPTION.md", "README.md"]:
        content = (HERE / filename).read_text()
        for link in re.findall(r"\]\(([^)]+)\)", content):
            if ":" not in link and not link.startswith("#"):
                target = HERE / link.split("#")[0]
                local_links.append(target)
                # The self-linked verification receipt is written below.
                if target.name != "VERIFICATION.json":
                    assert target.is_file(), (filename, link)
    checks["local_caption_readme_links_resolve"] = True
    source_checks = {}
    if args.source_root:
        for source_id, source in data["sources"].items():
            actual = args.source_root / source["path_relative_to_retained_campaign"]
            source_checks[source_id] = digest(actual) == source["sha256"]
        checks["consumed_source_bytes_unchanged"] = all(source_checks.values())
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}

    names = [
        "SOURCE_DATA.json", "SOURCE_DATA.csv", "SOURCE_EXCERPTS.md",
        "FIGURE1.svg", "FIGURE1.png", "CAPTION.md", "README.md", "VISUAL_REVIEW.md",
        "build_figure1.py", "rasterize.mjs", "verify_figure1.py",
    ]
    result = {
        "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS",
        "checks": checks,
        "source_checks": source_checks,
        "rendered_count_bindings": rendered,
        "png": {"width": width, "height": height, "bytes": len(png)},
        "artifacts": {name: {"sha256": digest(HERE / name), "bytes": (HERE / name).stat().st_size} for name in names},
        "scope": "Deterministic aggregate/artifact verification only; no patient-level scientific replay, clinical adjudication or independent audit.",
        "visual_review": "Recorded separately in VISUAL_REVIEW.md against the final raster hash.",
    }
    (HERE / "VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(target.is_file() for target in local_links)
    print(json.dumps({"status": result["status"], "checks_passed": len(checks), "source_files_unchanged": len(source_checks), "png": result["png"]}, indent=2))


if __name__ == "__main__":
    main()
