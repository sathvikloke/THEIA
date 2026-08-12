"""What counts as "EGFR-positive"? Harmonising the label across cohorts.

The internal cohort (NSCLC-Radiogenomics) records EGFR status as a bare
`Mutant / Wildtype / Unknown`. Any external cohort assembled from TCGA or CPTAC
comes through cBioPortal with the actual protein change, and those two things are
not the same label.

Measured against the public API, `luad_tcga_gdc` reports 91 EGFR mutation records
over 75 patients and `luad_cptac_gdc` 85 over 70. Alongside the expected L858R
(22 and 33) and exon-19 deletions (18 and 17) sit L62R, R222L, E545Q, K479I,
H358R, E84K and I143L -- all in the extracellular domain, none of them
TKI-relevant, none of them what a thoracic oncologist means by "EGFR-mutant".
About 10% of the "mutant" calls are passengers. On an external test set with ~17
positives, that is two patients labelled backwards.

This matters more than its size suggests, because it is the same lever the
Netherlands Cancer Institute group pulled: their result was that biopsy-anchored
labels beat all-lesion labels *despite ten times less training data*
(doi 10.1007/s00330-026-12601-9). Label fidelity, not sample size.

Scope and honesty about it. The authoritative classification is OncoKB's, which
is curated by humans and versioned. This module is a conservative offline
approximation of it, built from the variants actually present in the two
collections plus the standard exon-19/exon-21 rules. It is deliberately biased
toward `UNCERTAIN` rather than toward `ACTIVATING`: a wrongly-included passenger
corrupts the positive class, whereas a wrongly-excluded activating variant only
shrinks it. Anything it cannot place is reported, never silently dropped.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter

CBIOPORTAL = "https://www.cbioportal.org/api"
EGFR_ENTREZ = 1956

# EGFR domain boundaries (UniProt P00533 numbering, mature-protein convention as
# used in the clinical literature). The kinase domain is ~712-979; everything
# below ~645 is extracellular.
EXTRACELLULAR_MAX = 645
KINASE_LO, KINASE_HI = 688, 979

# Exon 19 in-frame deletions cluster over the LREA motif at 746-750; the accepted
# clinical window for a sensitising exon-19 deletion is roughly 729-761.
EXON19_LO, EXON19_HI = 729, 761

# Explicitly curated, because rules alone get these wrong.
ACTIVATING_POINT = {"L858R", "L861Q", "G719A", "G719C", "G719S", "G719D",
                    "S768I", "E709K", "E709A"}
RESISTANCE = {"T790M", "C797S", "L718Q", "G724S"}

_POS = re.compile(r"[A-Z](\d+)")


def _positions(change: str) -> list[int]:
    return [int(m) for m in _POS.findall(change or "")]


def classify(change: str) -> str:
    """One protein change -> ACTIVATING | RESISTANCE | EXON20INS | PASSENGER | UNCERTAIN.

    EXON20INS is kept separate rather than folded into ACTIVATING: those tumours
    are oncogene-driven but do not respond to first- or second-generation TKIs, so
    a study whose clinical premise is "who should get a TKI" may want them out.
    Whichever way that is decided, it should be decided explicitly.
    """
    if not change:
        return "UNCERTAIN"
    c = change.strip().lstrip("p.")
    if c in RESISTANCE:
        return "RESISTANCE"
    if c in ACTIVATING_POINT:
        return "ACTIVATING"

    pos = _positions(c)
    lo = min(pos) if pos else None

    if "ins" in c and "del" not in c:
        if lo is not None and 762 <= lo <= 775:
            return "EXON20INS"
        return "UNCERTAIN"

    if "del" in c:                       # covers del and delins
        if lo is not None and EXON19_LO <= lo <= EXON19_HI:
            return "ACTIVATING"
        return "UNCERTAIN"

    if lo is None:
        return "UNCERTAIN"
    if lo <= EXTRACELLULAR_MAX:
        return "PASSENGER"               # extracellular; not TKI-relevant
    if KINASE_LO <= lo <= KINASE_HI:
        return "UNCERTAIN"               # in the kinase domain but not a known hotspot
    return "UNCERTAIN"


def fetch_egfr(study: str, timeout: int = 60) -> list[dict]:
    """Every EGFR mutation record in a cBioPortal study, with protein changes."""
    url = f"{CBIOPORTAL}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED"
    body = json.dumps({"sampleListId": f"{study}_all",
                       "entrezGeneIds": [EGFR_ENTREZ]}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def patient_labels(records: list[dict], definition: str = "activating") -> dict[str, int]:
    """Patient -> 1/0 under one of two pre-specifiable definitions.

    `any`        : any EGFR mutation counts. Matches the internal cohort's binary
                   field, and is therefore the definition that keeps internal and
                   external comparable -- at the cost of including passengers.
    `activating` : only canonical sensitising variants count. Matches what the
                   label means clinically, at the cost of an internal/external
                   asymmetry that must be disclosed.

    Only patients appearing in `records` can be labelled 1; a patient absent from
    the mutation table is *not* labelled 0 here, because absence from this call
    can mean "wild-type" or "not sequenced" and the caller is the only one who
    knows which. Resolve that against the study's sequenced-sample list.
    """
    if definition not in ("any", "activating"):
        raise ValueError(f"definition must be 'any' or 'activating', got {definition!r}")
    out: dict[str, int] = {}
    for r in records:
        pid = r.get("patientId")
        if not pid:
            continue
        cls = classify(r.get("proteinChange", ""))
        pos = 1 if definition == "any" else int(cls == "ACTIVATING")
        out[pid] = max(out.get(pid, 0), pos)
    return out


def summarise(records: list[dict]) -> dict:
    counts = Counter(classify(r.get("proteinChange", "")) for r in records)
    by_class: dict[str, Counter] = {}
    for r in records:
        by_class.setdefault(classify(r.get("proteinChange", "")), Counter())[
            r.get("proteinChange", "?")] += 1
    return {
        "n_records": len(records),
        "n_patients": len({r.get("patientId") for r in records if r.get("patientId")}),
        "class_counts": dict(counts),
        "n_patients_any": len(patient_labels(records, "any")),
        "n_patients_activating": sum(patient_labels(records, "activating").values()),
        "variants_by_class": {k: dict(v) for k, v in by_class.items()},
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", default="luad_tcga_gdc,luad_cptac_gdc,lusc_tcga_gdc")
    ap.add_argument("--out", default="results/external_labels.json")
    a = ap.parse_args()

    out = {}
    for study in a.studies.split(","):
        try:
            recs = fetch_egfr(study)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[variants] {study}: fetch failed ({exc})")
            continue
        s = summarise(recs)
        out[study] = s
        drop = s["n_patients_any"] - s["n_patients_activating"]
        print(f"\n=== {study} ===")
        print(f"  {s['n_records']} records over {s['n_patients']} patients")
        for cls, n in sorted(s["class_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {cls:<11} {n:>3}  "
                  f"{', '.join(list(s['variants_by_class'][cls])[:6])}")
        print(f"  positives under 'any'        : {s['n_patients_any']}")
        print(f"  positives under 'activating' : {s['n_patients_activating']}"
              f"   ({drop} patients change label, "
              f"{100*drop/max(s['n_patients_any'],1):.0f}%)")

    if out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\n[variants] wrote {a.out}")
        print("[variants] the definition used for the external test set is "
              "pre-specified in docs/ANALYSIS_PLAN.md section 3.")


if __name__ == "__main__":
    main()
