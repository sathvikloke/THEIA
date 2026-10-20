#!/usr/bin/env python3
"""Render the retained historical cohort flow; stdlib only, no patient data.

Rebuild with: python3 build_figure1.py
Source capture: python3 build_figure1.py --capture-sources /path/to/2026-10-13_nine_hour_210100
The source-capture option creates aggregate data/excerpts from the named closed
review copy. It does not recompute attrition from patient records.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_sources(root):
    paths = {
        "main": "codex_closeout_review/journal_set_v3/manuscript/MAIN_PAPER_reviewed_copy.md",
        "supplement": "codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md",
        "claim_table": "codex_closeout_review/journal_set_v3/manuscript/CLAIM_EVIDENCE_reviewed_copy.md",
        "hourly_review": "HOURLY_REVIEW_02.md",
        "readiness_seal": "REVIEW_02_READINESS_SEAL.json",
        "prior_consumed_sources": "codex_paper_work/CONSUMED_SOURCES.json",
    }
    documents = {key: (root / path).read_text() for key, path in paths.items()}
    sources = {
        key: {"path_relative_to_retained_campaign": path, "sha256": sha256(root / path)}
        for key, path in paths.items()
    }
    selectors = {
        "main_data": ("main", "The internal cohort, NSCLC-Radiogenomics"),
        "main_external": ("main", "TCGA-LUAD (13,15"),
        "source_caption": ("main", "**Figure 1:**"),
        "attrition": ("supplement", "**Cohort attrition (reconstructed"),
        "external_flow": ("supplement", "- **TCGA-LUAD flow.**"),
        "external_endpoint": ("supplement", "- **TCGA-LUAD endpoint.**"),
        "cptac": ("supplement", "- **CPTAC-LUAD.**"),
        "overlap": ("supplement", "- **Overlap between cohorts"),
        "hourly": ("hourly_review", "Readiness sealed tranche02"),
        "claim_attrition": ("claim_table", "| 18 |"),
    }
    excerpts = {}
    for key, (source, prefix) in selectors.items():
        lines = documents[source].splitlines()
        matches = [(i + 1, line) for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one source excerpt for {key}; got {len(matches)}")
        line, text = matches[0]
        excerpts[key] = {"source_id": source, "line_start": line, "line_end": line, "text": text}
    lines = documents["main"].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("**Table 1:"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("**Table 2:"))
    excerpts["source_table1"] = {
        "source_id": "main", "line_start": start + 1, "line_end": end,
        "text": "\n".join(lines[start:end]).rstrip(),
    }
    # Guard the particular closed copy used to transcribe the aggregate rows.
    required = {
        "attrition": [
            "211 → 188", "188 → 158", "158 → 153",
            "144 with a segmentation", "44 point-annotated",
            "26 had no known EGFR or KRAS status",
            "3 had a local AIM annotation that references no CT series",
            "four air-rule exclusions are replays",
            "approximate the historical slice order by image position",
        ],
        "external_flow": ["67 subjects", "11 had no lesion", "2 had an ineligible", "4 were development", "8 had an unknown", "40 were evaluated"],
        "external_endpoint": ["1 assay-positive, 3 research-positive-only and 36 research-negative", "no clinical-assay negative"],
        "cptac": ["29 index CTs", "28 eligible"],
    }
    for key, fragments in required.items():
        for fragment in fragments:
            if fragment not in excerpts[key]["text"]:
                raise ValueError(f"Closed-source guard failed: {key}: {fragment}")
    rows = []

    def row(identifier, cohort, count, kind, parent, label, evidence, certainty="retained_report_count"):
        rows.append({
            "id": identifier, "cohort": cohort, "count": count, "kind": kind,
            "parent": parent, "label": label, "source_excerpt": evidence,
            "certainty": certainty,
        })

    row("i_source", "internal", 211, "stage", "", "Public source subjects with listed CT", "attrition")
    row("i_imported", "internal", 188, "stage", "i_source", "Imported locally", "attrition")
    row("i_not_imported", "internal", 23, "excluded", "i_source", "Not imported locally", "attrition")
    row("i_no_download_no_aim", "internal", 20, "exclusion_component", "i_not_imported", "No local AIM and absent from CT/SEG download", "main_data")
    row("i_unmatched_aim", "internal", 3, "exclusion_component", "i_not_imported", "AIM reference matches no listed CT series; selection reason unrecorded", "attrition", "selection_reason_unknown")
    row("i_ct_seg", "internal", 144, "download_component", "i_imported", "CT + SEG download", "attrition")
    row("i_aim_only", "internal", 44, "download_component", "i_imported", "AIM-only CT download", "attrition")
    row("i_unbuilt", "internal", 30, "excluded", "i_imported", "Imported but no model input built", "attrition")
    row("i_unknown_both", "internal", 26, "exclusion_component", "i_unbuilt", "EGFR and KRAS unknown; skipped before localization", "attrition", "reconstructed_under_frozen_skip_order")
    row("i_air", "internal", 4, "exclusion_component", "i_unbuilt", "Air-rule exclusions; mean HU < -700", "attrition", "reconstructed_rule_replay_not_execution_log")
    row("i_approx_order", "internal", 3, "nested_subset", "i_air", "Historical slice order approximated using image position (IPP)", "hourly", "approximate_reconstruction")
    row("i_scored", "internal", 158, "stage", "i_imported", "Built and scored historical inputs", "attrition")
    row("i_extent", "internal", 119, "annotation_component", "i_scored", "Lesion-extent annotation", "main_data")
    row("i_point", "internal", 39, "annotation_component", "i_scored", "Point annotation", "attrition")
    row("i_egfr_unknown", "internal", 5, "metric_excluded", "i_scored", "EGFR unknown; KRAS known; excluded from EGFR metrics", "attrition")
    row("i_labeled", "internal", 153, "stage", "i_scored", "Clinical EGFR label", "main_data")
    row("i_mutant", "internal", 40, "label_subset", "i_labeled", "EGFR mutant", "main_data")
    row("i_mislocated", "internal", 7, "limitation_subset", "i_scored", "Known mislocated point-annotated inputs in historical results", "attrition")
    row("i_mislocated_labeled", "internal", 5, "nested_subset", "i_mislocated", "Mislocated inputs with EGFR labels", "attrition")
    row("e_source", "external", 67, "stage", "", "TCGA-LUAD subjects with CT", "external_flow")
    row("e_no_location", "external", 11, "exclusion_component", "e_source", "No resolvable lesion location", "external_flow")
    row("e_ineligible_scan", "external", 2, "exclusion_component", "e_source", "Ineligible index scan", "external_flow")
    row("e_air_adapter", "external", 2, "exclusion_component", "e_source", "Air gate or input adapter failed", "external_flow")
    row("e_development", "external", 4, "exclusion_component", "e_source", "Development subjects used to freeze adapter", "external_flow")
    row("e_unknown", "external", 8, "exclusion_component", "e_source", "Unknown endpoint", "external_flow")
    row("e_evaluated", "external", 40, "stage", "e_source", "External molecular evaluation", "external_flow")
    row("e_positive", "external", 4, "label_component", "e_evaluated", "Research-sequencing positive", "main_external")
    row("e_research_negative", "external", 36, "label_component", "e_evaluated", "Research-sequencing negative; not clinical-assay negative", "external_endpoint")
    row("e_clinical_positive", "external", 1, "nested_subset", "e_positive", "Assay-positive and research-positive", "external_endpoint")
    row("e_research_only_positive", "external", 3, "nested_subset", "e_positive", "Research-positive only", "external_endpoint")
    row("r_paired", "rider", 28, "stage", "", "Paired test–retest scans; no molecular label", "main_external")
    row("c_chosen", "cptac", 29, "preparation", "", "Index CTs chosen by a rule that read no label field", "cptac")
    row("c_eligible", "cptac", 28, "preparation", "c_chosen", "Eligible before reading; not evaluated", "cptac")
    data = {
        "schema_version": 1, "artifact": "P01 Figure 1 author-review copy",
        "created_date": "2026-10-20", "evidence_as_of": "2026-10-14",
        "status": "technical_figure_generated_publication_adoption_pending_student_author_review",
        "source_choice": "Latest closed journal_set_v3 retained author-review copy",
        "scientific_replay_performed": False, "new_statistics_performed": False,
        "source_root_locator": "theia_campaigns/2026-10-13_nine_hour_210100",
        "sources": sources, "excerpts": excerpts, "rows": rows,
        "limitations": [
            "Aggregate transcription from retained closed review evidence; original patient-level R01 files were not available during construction.",
            "No preprocessing log was saved; exclusion reasons were reconstructed after outcomes were known.",
            "Selection reason for three non-imported AIM holders is unrecorded.",
            "Four air-rule exclusions are replays, three with approximate image-position/IPP ordering.",
            "The retained evidence does not name GDCM; no exact historical GDCM-order claim is added.",
            "Historical M0 inputs include seven known mislocated inputs, five with EGFR labels.",
            "The internal and external cohorts are not demonstrated to be a person-level deduplicated union.",
            "Research sequencing negative is not clinical assay-negative; unknowns are not negatives.",
            "CPTAC is preparation only; labels were technically accessible before the lock.",
        ],
    }
    (HERE / "SOURCE_DATA.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    with (HERE / "SOURCE_DATA.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Retained source excerpts",
        "",
        "Exact excerpts from the closed 2026-10-14 review evidence, captured 2026-10-20.",
        "These are aggregate sources, not a fresh patient-level replay. Full-file hashes are in SOURCE_DATA.json.",
    ]
    for key, excerpt in excerpts.items():
        source = sources[excerpt["source_id"]]
        text.extend(["", f"## {key}", "", f"Source: `{source['path_relative_to_retained_campaign']}`, lines {excerpt['line_start']}–{excerpt['line_end']}.", "", excerpt["text"]])
    (HERE / "SOURCE_EXCERPTS.md").write_text("\n".join(text) + "\n")


def validate(data):
    n = {r["id"]: r["count"] for r in data["rows"]}
    checks = {
        "internal_import": n["i_source"] == n["i_imported"] + n["i_not_imported"],
        "not_imported_components": n["i_not_imported"] == n["i_no_download_no_aim"] + n["i_unmatched_aim"],
        "download_components": n["i_imported"] == n["i_ct_seg"] + n["i_aim_only"],
        "internal_build": n["i_imported"] == n["i_scored"] + n["i_unbuilt"],
        "unbuilt_components": n["i_unbuilt"] == n["i_unknown_both"] + n["i_air"],
        "annotation_components": n["i_scored"] == n["i_extent"] + n["i_point"],
        "internal_label": n["i_scored"] == n["i_labeled"] + n["i_egfr_unknown"],
        "external_flow": n["e_source"] == n["e_evaluated"] + sum(n[k] for k in ["e_no_location", "e_ineligible_scan", "e_air_adapter", "e_development", "e_unknown"]),
        "external_labels": n["e_evaluated"] == n["e_positive"] + n["e_research_negative"],
        "external_positive_sources": n["e_positive"] == n["e_clinical_positive"] + n["e_research_only_positive"],
        "approximation_subset": n["i_approx_order"] <= n["i_air"],
        "mislocated_labeled_subset": n["i_mislocated_labeled"] <= n["i_mislocated"],
    }
    if not all(checks.values()):
        raise ValueError(checks)
    return checks


def render(data):
    validate(data)
    n = {r["id"]: r["count"] for r in data["rows"]}
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1640" viewBox="0 0 1600 1640" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Figure 1. Historical cohort flow</title>",
        "<desc id=\"desc\">Internal source flow 211 to 188 to 158 to 153. Exclusions distinguish unknown selection reasons and reconstructed air-rule replays. External TCGA-LUAD, RIDER and preparation-only CPTAC remain separate. Author-review copy; publication adoption pending.</desc>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#51616d"/></marker></defs>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172d3b} .title{font-size:36px;font-weight:700} .panel{font-size:27px;font-weight:700} .head{font-size:24px;font-weight:700} .body{font-size:22px} .small{font-size:20px} .muted{fill:#485a66} .warning{fill:#68451b} .count{font-size:36px;font-weight:700} .rule{stroke:#d7dfe4;stroke-width:1.5}</style>',
        '<rect width="1600" height="1640" fill="white"/>',
    ]

    def text(x, y, line, cls="body", anchor="start", row_id=None):
        attrs = f' data-row-id="{row_id}" data-count="{n[row_id]}"' if row_id else ""
        svg.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{attrs}>{html.escape(str(line))}</text>')

    def lines(x, y, content, cls="body", dy=29):
        for i, line in enumerate(content):
            text(x, y + i * dy, line, cls)

    def rect(x, y, w, h, fill="#f5f8fa", stroke="#c1ced6", radius=8):
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def path(d, arrow=True):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        svg.append(f'<path d="{d}" fill="none" stroke="#51616d" stroke-width="2.3"{marker}/>')

    def number(x, y, row_id, suffix="", cls="body"):
        text(x, y, f'{n[row_id]}{suffix}', cls, row_id=row_id)

    def stage(x, y, w, row_id, label, detail, h=120, final=False):
        rect(x, y, w, h, "#e9f3f5" if final else "#f3f7fa", "#6595a2" if final else "#aabdc9")
        number(x + 20, y + 44, row_id, cls="count")
        text(x + 20, y + 76, label, "head")
        if detail:
            text(x + 20, y + 104, detail, "small")

    text(40, 58, "Figure 1. Historical cohort flow", "title")
    text(40, 96, "Retained evidence through 14 October 2026 · Author-review image prepared 20 October 2026", "body muted")
    text(40, 126, "Corrected reporting of historical inputs (M0); this figure does not report corrected-input replication.", "small muted")
    rect(40, 154, 954, 1098, "white", "#d7dfe4", 10)
    rect(1016, 154, 544, 1098, "white", "#d7dfe4", 10)
    text(62, 193, "A   Internal · NSCLC-Radiogenomics", "panel")
    text(1038, 193, "B   External · TCGA-LUAD", "panel")

    stage(64, 222, 356, "i_source", "Public source subjects", "All have a listed CT series")
    stage(64, 476, 356, "i_imported", "Imported locally", "")
    number(84, 578, "i_ct_seg", " CT + SEG", "small")
    number(246, 578, "i_aim_only", " AIM-only CT", "small")
    stage(64, 817, 356, "i_scored", "Built and scored", "Historical input version")
    stage(64, 1065, 356, "i_labeled", "Clinical EGFR label", "", final=True)
    number(84, 1169, "i_mutant", " EGFR-mutant", "small")

    path("M 242 342 V 476")
    path("M 242 596 V 817")
    path("M 242 937 V 1065")
    path("M 242 394 H 470")
    path("M 242 656 H 470")
    path("M 242 992 H 470")

    rect(480, 327, 490, 194, "#fff8ee", "#cbb18b")
    number(500, 363, "i_not_imported", " not imported", "head warning")
    number(500, 400, "i_no_download_no_aim", " · No local AIM; absent from", "body")
    text(542, 428, "CT/SEG download", "body")
    number(500, 463, "i_unmatched_aim", " · AIM reference matches no listed CT", "body")
    text(542, 491, "Selection reason unrecorded", "small warning")

    rect(480, 615, 490, 251, "#fff8ee", "#cbb18b")
    number(500, 653, "i_unbuilt", " imported, no input built", "head warning")
    number(500, 692, "i_unknown_both", " · EGFR and KRAS unknown", "body")
    text(542, 722, "Skipped before localization", "small")
    number(500, 759, "i_air", " · Reconstructed air-rule exclusions", "body")
    text(542, 789, "Mean HU < −700; frozen-rule replays", "small")
    number(542, 821, "i_approx_order", " use approximate slice order (IPP)", "small warning")
    text(500, 850, "No original preprocessing log was saved.", "small warning")

    number(480, 919, "i_extent", " lesion-extent annotations", "body")
    number(480, 948, "i_point", " point annotations", "body")

    rect(480, 975, 490, 128, "#fff8ee", "#cbb18b")
    number(500, 1014, "i_egfr_unknown", " with unknown EGFR", "head warning")
    text(500, 1046, "KRAS known; retained among scored inputs", "small")
    text(500, 1076, "Excluded from EGFR metrics; never negative", "small")

    text(64, 1220, f'Known historical input defect: {n["i_mislocated"]} mislocated point inputs ({n["i_mislocated_labeled"]} EGFR-labelled).', "small muted")

    stage(1040, 222, 496, "e_source", "Subjects with CT", "")
    path("M 1070 342 V 850 H 1115")
    path("M 1070 426 H 1100")
    rect(1110, 383, 426, 363, "#fff8ee", "#cbb18b")
    text(1130, 421, "Excluded from evaluation", "head warning")
    number(1130, 466, "e_no_location", " · No lesion location", "body")
    number(1130, 509, "e_ineligible_scan", " · Ineligible index scan", "body")
    number(1130, 552, "e_air_adapter", " · Air gate or adapter failed", "body")
    number(1130, 595, "e_development", " · Development subjects", "body")
    text(1170, 624, "Used to freeze the adapter", "small")
    number(1130, 668, "e_unknown", " · Unknown endpoint", "body")
    text(1130, 714, "Unknown is never negative.", "small warning")
    stage(1125, 795, 411, "e_evaluated", "Molecular evaluation", "", h=177, final=True)
    number(1145, 903, "e_positive", " research-sequencing positive", "small")
    number(1145, 942, "e_research_negative", " research-sequencing negative", "small")
    lines(1040, 1024, [
        "Research negative: no qualifying variant",
        "in the open masked sequencing file.",
        "No clinical-assay negative is represented.",
        "Internal and external endpoint definitions",
        "have not been shown to be equivalent.",
    ], "small", 31)
    number(1040, 1201, "e_clinical_positive", " of the 4 positives is also assay-positive.", "small muted")

    rect(40, 1276, 744, 244, "white", "#d7dfe4", 10)
    rect(808, 1276, 752, 244, "white", "#d7dfe4", 10)
    text(62, 1317, "C   Test–retest · RIDER Lung CT", "panel")
    number(62, 1374, "r_paired", " patients with paired repeat scans", "head")
    lines(62, 1418, ["Manual contours; repeatability / annotation alignment", "No molecular label"], "body", 34)
    text(830, 1317, "D   Preparation only · CPTAC-LUAD", "panel")
    number(830, 1374, "c_chosen", " index CTs chosen", "head")
    number(1180, 1374, "c_eligible", " eligible before reading", "small")
    lines(830, 1417, [
        "Selection rule read no label field; not evaluated.",
        "Labels technically accessible before the lock;",
        "analyst blindness is not demonstrated.",
    ], "small", 29)
    text(40, 1560, "Separate cohorts: these counts do not establish a deduplicated union of persons.", "head")
    text(40, 1594, "Publication adoption pending student / author review. Sources, retained caption and provenance accompany this figure.", "small muted")
    svg.append("</svg>")
    (HERE / "FIGURE1.svg").write_text("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-sources", type=Path)
    args = parser.parse_args()
    if args.capture_sources:
        capture_sources(args.capture_sources)
    data = json.loads((HERE / "SOURCE_DATA.json").read_text())
    render(data)
    print(json.dumps({"svg": str(HERE / "FIGURE1.svg"), "arithmetic_checks": validate(data)}, indent=2))


if __name__ == "__main__":
    main()
