"""Participant flow and cohort characteristics — the CLAIM tables.

CLAIM asks for a flow of participants through the study with reasons for every
exclusion, and a table of cohort characteristics. Both are derived here from the
data rather than typed into a document, so they cannot drift away from what the
pipeline actually did.

One thing this module deliberately does NOT produce: an accrual window. The
collection's `CT Date` column spans 1989-11-07 to 1996-09-11 across all 211 rows,
which is impossible for a cohort whose sequencing and surgery are documented in
the 2000s — those values are de-identification offsets, not dates. Publishing a
range computed from them would put a fabricated accrual window in Methods. The
index-test-to-reference-standard interval is reported instead, which *is*
meaningful because it is a difference of two offset dates and the offset cancels.

Acquisition characteristics (slice thickness, kernel, vendor) are read from DICOM
headers, and `data/raw/nsclc_radiogenomics/dicom/` currently holds 188 patient
directories containing zero files — the imaging was deleted to reclaim disk. What
survives is recorded in `import_log.json` for the subset it covers, and the gap
is reported as a gap rather than left blank.

Run: python -m theia.analysis.cohort
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

UNKNOWN = {"unknown", "not collected", "nan", ""}


def _clean(s) -> str:
    return str(s).strip()


def flow(cfg) -> list[dict]:
    """Counts at each stage, with the reason patients are lost."""
    raw = cfg.paths.raw_dir
    clin = pd.read_csv(os.path.join(raw, "clinical", "clinical.csv"))
    dicom_dir = os.path.join(raw, "dicom")
    with_dicom = (len([d for d in os.listdir(dicom_dir)
                       if os.path.isdir(os.path.join(dicom_dir, d))])
                  if os.path.isdir(dicom_dir) else 0)

    rows = [json.loads(l) for l in
            open(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))]
    labelled = [r for r in rows if int(r.get("labels", {}).get("EGFR", -1)) in (0, 1)]
    hist = dict(zip(clin["Case ID"], clin["Histology "].map(_clean)))
    adeno = [r for r in labelled if hist.get(r["patient_id"]) == "Adenocarcinoma"]
    seg_adeno = [r for r in adeno if r.get("has_mask")]

    return [
        {"stage": "patients in the clinical sheet", "n": len(clin), "lost": None,
         "reason": None},
        {"stage": "with an imaging series present", "n": with_dicom,
         "lost": len(clin) - with_dicom,
         "reason": "no CT series in the collection for this patient"},
        {"stage": "successfully preprocessed", "n": len(rows),
         "lost": with_dicom - len(rows),
         "reason": "no usable tumour localisation: no segmentation and no AIM "
                   "point, AIM point unresolvable to a slice, or AIM point "
                   "landing in air (< -700 HU)"},
        {"stage": "with a known EGFR label", "n": len(labelled),
         "lost": len(rows) - len(labelled),
         "reason": "EGFR status recorded as Unknown or Not collected"},
        {"stage": "adenocarcinoma (primary analysis set, before tier)",
         "n": len(adeno), "lost": len(labelled) - len(adeno),
         "reason": "squamous cell or NSCLC-NOS; all are EGFR wild-type, so "
                   "including them makes 13% of the cohort classifiable from "
                   "histology alone"},
        {"stage": "PRIMARY ANALYSIS SET: segmented adenocarcinoma",
         "n": len(seg_adeno), "lost": len(adeno) - len(seg_adeno),
         "reason": "AIM-only supervision tier: an annotated point but no pixel "
                   "mask, so the crop geometry differs and the ROI is all zeros"},
    ]


def table_one(cfg) -> pd.DataFrame:
    """Cohort characteristics split by EGFR status, restricted to labelled patients."""
    raw = cfg.paths.raw_dir
    clin = pd.read_csv(os.path.join(raw, "clinical", "clinical.csv")).set_index("Case ID",
                                                                                drop=False)
    rows = [json.loads(l) for l in
            open(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))]
    ids = [r["patient_id"] for r in rows
           if int(r.get("labels", {}).get("EGFR", -1)) in (0, 1)]
    sub = clin.loc[[i for i in ids if i in clin.index]].copy()
    sub["_egfr"] = sub["EGFR mutation status"].map(_clean)

    groups = [("EGFR mutant", sub[sub["_egfr"] == "Mutant"]),
              ("EGFR wild-type", sub[sub["_egfr"] == "Wildtype"]),
              ("all", sub)]
    out = []

    def add(label, fn):
        out.append({"characteristic": label, **{g: fn(d) for g, d in groups}})

    add("n", lambda d: f"{len(d)}")
    for col, name in (("Age at Histological Diagnosis", "age, median [IQR]"),
                      ("Pack Years", "pack-years, median [IQR]")):
        def num(d, col=col):
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            if v.empty:
                return "-"
            return f"{v.median():.0f} [{v.quantile(.25):.0f}-{v.quantile(.75):.0f}]"
        add(name, num)

    # %GG is recorded as ordered bands ("0%", ">0 - 25%", ...), not a number, so it
    # is tabulated as a category. Reading it with to_numeric silently yields all-NaN
    # and prints an empty row -- which is how it first appeared here.
    GG_ORDER = ["0%", ">0 - 25%", "25 - 50%", "50 - 75%", "75 - < 100%", "100%"]
    for col in ("Gender", "Ethnicity", "Smoking status", "Histology ",
                "Pathological T stage", "Pathological N stage", "%GG"):
        levels = sorted({_clean(x) for x in sub[col].dropna()})
        if col == "%GG":
            levels = [x for x in GG_ORDER if x in levels]
        for lv in levels:
            if lv.lower() in UNKNOWN:
                continue
            add(f"{col.strip()}: {lv}",
                lambda d, c=col, l=lv: (
                    f"{int((d[c].map(_clean) == l).sum())} "
                    f"({100*(d[c].map(_clean) == l).mean():.0f}%)"))

    return pd.DataFrame(out)


def demographics(cfg) -> dict:
    """The abstract's Materials and Methods sentence, in RSNA house style.

    Radiology journals require the abstract to name the cohort as "mean age,
    X years +/- SD; N male, M female", separately by sex where relevant. Table 1
    reports median [IQR] because age here is mildly skewed and the median is the
    honest summary; both are produced from the same column so the manuscript
    cannot quote a mean that does not match the table's median.
    """
    raw = cfg.paths.raw_dir
    clin = pd.read_csv(os.path.join(raw, "clinical", "clinical.csv")).set_index(
        "Case ID", drop=False)
    rows = [json.loads(l) for l in
            open(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))]
    ids = [r["patient_id"] for r in rows
           if int(r.get("labels", {}).get("EGFR", -1)) in (0, 1)]
    sub = clin.loc[[i for i in ids if i in clin.index]].copy()
    age = pd.to_numeric(sub["Age at Histological Diagnosis"], errors="coerce")
    sex = sub["Gender"].map(_clean)

    def block(mask) -> dict:
        a = age[mask].dropna()
        return {"n": int(mask.sum()), "n_age_known": int(a.size),
                "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max())}

    out = {"all": block(pd.Series(True, index=sub.index)),
           "male": block(sex == "Male"), "female": block(sex == "Female")}
    out["sentence"] = (
        f"{out['all']['n']} patients (mean age, {out['all']['mean']:.0f} years "
        f"+/- {out['all']['sd']:.0f} [SD]; {out['male']['n']} male, "
        f"{out['female']['n']} female)")
    return out


def acquisition(cfg) -> dict:
    """What is recoverable about how the scans were made, and what is not."""
    raw = cfg.paths.raw_dir
    log_path = os.path.join(raw, "import_log.json")
    dicom_dir = os.path.join(raw, "dicom")
    n_dirs = (len([d for d in os.listdir(dicom_dir)
                   if os.path.isdir(os.path.join(dicom_dir, d))])
              if os.path.isdir(dicom_dir) else 0)
    n_files = sum(len(fs) for _, _, fs in os.walk(dicom_dir)) if os.path.isdir(dicom_dir) else 0

    out = {"dicom_patient_dirs": n_dirs, "dicom_files_on_disk": n_files,
           "headers_readable": n_files > 0}
    if os.path.exists(log_path):
        log = json.load(open(log_path))
        out["import_log_entries"] = len(log)
        out["series_descriptions"] = dict(
            Counter(e.get("ct_description", "?") for e in log).most_common(15))
        slices = [e.get("ct_slices") for e in log if isinstance(e.get("ct_slices"), int)]
        if slices:
            out["ct_slices"] = {"n": len(slices), "median": int(np.median(slices)),
                                "min": int(min(slices)), "max": int(max(slices))}
    if n_files == 0:
        out["note"] = (
            "Slice thickness, convolution kernel and scanner vendor/model come from "
            "DICOM headers and cannot be recomputed: the imaging was deleted from "
            "local disk to reclaim space. Re-download from TCIA is required to "
            "populate the acquisition table. Series descriptions above are the "
            "residue preserved in import_log.json and cover only the subset it "
            "records.")
    return out


def interval(cfg) -> dict:
    """Index test to reference standard, in days.

    Meaningful even though the absolute dates are de-identification offsets,
    because this is a difference of two offset dates and the offset cancels.
    """
    clin = pd.read_csv(os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv"))
    v = pd.to_numeric(clin["Days between CT and surgery"], errors="coerce").dropna()
    return {"n": int(v.size), "median": float(v.median()),
            "q1": float(v.quantile(.25)), "q3": float(v.quantile(.75)),
            "min": float(v.min()), "max": float(v.max())}


def main() -> None:
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/cohort.json")
    ap.add_argument("--table1", default="docs/table1.md")
    a = ap.parse_args()
    cfg = load_config(a.config)

    print("=== PARTICIPANT FLOW ===")
    f = flow(cfg)
    for s in f:
        lost = "" if s["lost"] in (None, 0) else f"  (-{s['lost']}: {s['reason']})"
        print(f"  {s['n']:>4}  {s['stage']}{lost}")

    t1 = table_one(cfg)
    print("\n=== TABLE 1 (first rows) ===")
    print(t1.head(8).to_string(index=False))

    acq = acquisition(cfg)
    print("\n=== ACQUISITION ===")
    print(f"  dicom dirs {acq['dicom_patient_dirs']}, files on disk "
          f"{acq['dicom_files_on_disk']}, headers readable: {acq['headers_readable']}")
    if "note" in acq:
        print(f"  NOTE: {acq['note'][:150]}...")

    iv = interval(cfg)
    print(f"\n=== INDEX TEST -> REFERENCE STANDARD ===")
    print(f"  n={iv['n']}, median {iv['median']:.0f} days "
          f"[IQR {iv['q1']:.1f}-{iv['q3']:.1f}], range {iv['min']:.0f}-{iv['max']:.0f}")

    dem = demographics(cfg)
    print(f"\n=== ABSTRACT DEMOGRAPHICS (RSNA house style) ===")
    print(f"  {dem['sentence']}")

    json.dump({"flow": f, "acquisition": acq, "interval_days": iv,
               "demographics": dem, "table_one": t1.to_dict(orient="records")},
              open(a.out, "w"), indent=2)
    with open(a.table1, "w") as fh:
        fh.write("# Table 1 — cohort characteristics by EGFR status\n\n")
        fh.write("Generated by `python -m theia.analysis.cohort`. Do not edit by hand.\n\n")
        fh.write(t1.to_markdown(index=False))
        fh.write("\n\n## Participant flow\n\n| stage | n | excluded | reason |\n")
        fh.write("|---|---|---|---|\n")
        for s in f:
            fh.write(f"| {s['stage']} | {s['n']} | {s['lost'] or ''} | "
                     f"{s['reason'] or ''} |\n")
        fh.write(f"\n## Index test to reference standard\n\n"
                 f"Median {iv['median']:.0f} days (IQR {iv['q1']:.1f}–{iv['q3']:.1f}, "
                 f"range {iv['min']:.0f}–{iv['max']:.0f}, n={iv['n']}).\n\n"
                 f"No accrual window is reported: the collection's `CT Date` values "
                 f"span 1989–1996 and are de-identification offsets, not dates.\n")
    print(f"\n[cohort] wrote {a.out} and {a.table1}")


if __name__ == "__main__":
    main()
