# Retained source excerpts

Exact excerpts from the closed 2026-10-14 review evidence, captured 2026-10-20.
These are aggregate sources, not a fresh patient-level replay. Full-file hashes are in SOURCE_DATA.json.

## main_data

Source: `codex_closeout_review/journal_set_v3/manuscript/MAIN_PAPER_reviewed_copy.md`, lines 52–52.

The internal cohort, NSCLC-Radiogenomics (12,23,24), provided 158 scored patients from 211 source subjects (Fig 1). Public CT metadata lists all 211; 188 were imported locally. Of the 23 not imported, 20 lacked a local AIM annotation and were absent from the CT/SEG download, while three had AIM annotations but were not imported for an unrecorded selection reason. Among the 188 imported, 26 were skipped because neither EGFR nor KRAS was known, and four further exclusions were reconstructed under the preprocessing air rule; three of those four reconstructions used approximate historical slice ordering and are not original execution records. Of the 158 scored patients, 153 had a clinical EGFR label (40 mutant) and 119 had a lesion extent, from semiautomatic segmentations edited by one thoracic radiologist and approved by a second (23); the rest had a point annotation. The reference was the clinical-record result of a targeted SNaPshot assay at tested positions in exons 18–21, so "wild type" means that no mutation was detected at those positions. Institution, subcohort, annotation tier, sex, smoking and scanner vendor were potential confounders, associated with one another and the label: 32 of 81 labeled patients from one institution were mutant, against 8 of 72 from the other (Table 1). Subgroup performance was therefore not estimated.

## main_external

Source: `codex_closeout_review/journal_set_v3/manuscript/MAIN_PAPER_reviewed_copy.md`, lines 54–54.

TCGA-LUAD (13,15–19) contributed 40 evaluable patients: four with an activating EGFR kinase-domain variant in research sequencing and 36 research-sequencing negatives; lesion locations were resolved by a prespecified rule from five to seven crowdsourced line measurements per patient (14). Unknown labels were excluded, never counted as negative, and this endpoint has not been shown to identify the same condition as the internal label. RIDER Lung CT (20,25) provided 28 patients with paired repeat scans, described as same-day, and manual tumor contours drawn by a single radiation oncologist blinded to other delineations (21), with no molecular label (acquisition: Supplemental Material S7).

## source_caption

Source: `codex_closeout_review/journal_set_v3/manuscript/MAIN_PAPER_reviewed_copy.md`, lines 192–192.

**Figure 1:** Cohort flow for the internal, external molecular, test–retest and preparation-only cohorts. The corrected internal flow is 211 public source subjects → 188 locally imported → 158 scored → 153 EGFR-labelled; the 23 not imported reflect local selection, not absence of public CT. The four reconstructed preprocessing exclusions include three with approximate historical slice ordering. External and internal counts are separate and do not establish a deduplicated union. Preparation-only CPTAC cohort (CPTAC-LUAD [22]: index CT scans chosen by a label-free rule and prepared for reading; not analyzed; per-subject labels were technically accessible before the lock).

## attrition

Source: `codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md`, lines 50–50.

**Cohort attrition (reconstructed 2026-10-14).** Public NBIA metadata lists at least one CT series for each of the 211 source subjects, so the losses below are study import and processing losses, not absent collection imaging. 211 → 188: 188 subjects were imported from two disjoint downloads, 144 with a segmentation (CT + SEG download) and 44 point-annotated subjects without one (AIM-only CT download). Of the 23 not imported, 20 had neither a segmentation download nor a local AIM annotation, and 3 had a local AIM annotation that references no CT series in the collection listing; the selection reason for these 3 was not recorded. Sixteen of the 23 have a two-class EGFR label (3 mutant). 188 → 158: 30 imported subjects produced no model input; 26 had no known EGFR or KRAS status and were skipped before localization, and 4 point-annotated subjects were excluded because the annotated point fell in air (mean HU < −700), 3 of them after the annotated image was not found in the imported series and the frame-number fallback applied. 158 → 153: 5 built subjects have an unknown EGFR label (KRAS known). In total, 19 subjects with a two-class EGFR label were never built (16 not imported, 3 air-rule exclusions); they were not analyzed individually, come from the development institutions and are not external capacity. These reasons were reconstructed after outcomes were known by joining the clinical sheet with the saved download records, import links, AIM annotation files and built rows under the frozen skip order. No preprocessing log was saved, the importer's log covers only the AIM-only download, and the four air-rule exclusions are replays of the frozen rule, three of which approximate the historical slice order by image position. They are reconstructed causes, not contemporaneous execution records. Among the 39 scored point-annotated subjects, 7 were cropped from an imported series other than the one their annotation references (5 of them among the 153 labeled); these are the mislocated inputs described next.

## external_flow

Source: `codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md`, lines 323–323.

- **TCGA-LUAD flow.** 67 subjects have CT; 11 had no lesion location resolvable by the rule below; 2 had an ineligible index scan; 2 failed the air gate or the input adapter at the resolved location; 4 were development subjects used to freeze the adapter; 8 had an unknown endpoint; 40 were evaluated. The evaluation specification was frozen by hash before any model was run on TCGA-LUAD images.

## external_endpoint

Source: `codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md`, lines 324–324.

- **TCGA-LUAD endpoint.** Primary: an activating EGFR kinase-domain variant in open research sequencing (GDC open masked mutation files, with the cBioPortal PanCanAtlas study as a concordance check). Exclusive label partition of the 67: 2 clinical-assay positive (both also research positive), 6 research-sequencing positive only, 51 research negative, 8 unknown; the evaluated 40 comprise 1 assay-positive, 3 research-positive-only and 36 research-negative patients. "Research-negative" means profiled primary-tumor whole-exome sequencing with no qualifying variant retained in the open, masked file; per-subject callability of exons 18–21 cannot be verified from open data, 29 of the 51 negatives used whole-genome-amplified DNA, and no clinical-assay negative is represented. Unknown results were never treated as negative. The internal label (clinical SNaPshot at tested positions in exons 18–21) and this endpoint differ in assay, callable region, variant qualification and kind of negative and have not been shown to identify the same condition. A secondary "sensitizing" definition gave 3 positives among the 40. Lesion localization for this cohort used Crowds Cure 2017 (14): for the 40 evaluated patients, 264 straight-line measurements (five to seven per patient, median seven), each by a different annotator, with no contours. They were collected with the Crowds Cure Cancer web application, built from Open Health Imaging Foundation viewer and Cornerstone components (Ziegler E, et al. JCO Clin Cancer Inform 2020;4:336–345; source code github.com/crowds-cure/cancer); no version is published for the 2017 deployment. Collection began at the RSNA 2017 meeting; the released timestamps run from November 26 to December 27, 2017 (43 of the 264 after December 1). Annotator status is self-reported and unverified: 164 of the 264 measurements were by self-reported radiologists and 100 by non-radiologists, and every patient had at least one radiologist measurement (across the whole release, 240 of the 368 lung measurements were by self-reported radiologists). No annotator measured a patient twice, so intra-rater variability cannot be assessed. Each patient's measurements are by different annotators, so inter-rater agreement for a matched finding could be described; the source reports no agreement statistic, and none is reported here. The location used was resolved by this study's own prespecified rule (single-linkage clustering of measurement midpoints; the finding measured by the most annotators, at least two; its medoid measurement), not supplied as a consensus by the source, which makes no consensus or blinding statement.

## cptac

Source: `codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md`, lines 328–328.

- **CPTAC-LUAD.** 29 index CTs chosen by a rule that read no label field, 28 eligible before reading. Labels are not joined. Per-subject label-bearing source files have been on local disk since 2026-09-21, before the selection lock, and a published aggregate (11 / 25 of 36 imaged) is known, so the analysts' blindness is not demonstrated.

## overlap

Source: `codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md`, lines 329–329.

- **Overlap between cohorts (bounded check).** The internal 153 and the external 40 are separate cohorts, and 193 = 153 + 40 is a nominal sum, not a deduplicated union of persons. A sampled stored-pixel identity check on 2026-10-14 hashed three fixed, filename-ordered instances from each of the 188 imported internal CT series (564 instances; 0 read errors) and every external instance (16,856 files: TCGA-LUAD 5,398, its second batch 2,567, CPTAC-LUAD 8,485 and 406 CPTAC survey headers; 46 constant CPTAC images excluded; 16,764 distinct non-constant digests; 0 read errors), using the existing exact-pixel digest. No sampled internal instance matched any external one. This detects duplication of the sampled stored images between collections; it cannot exclude overlap confined to unsampled slices or a different scan of the same person, nor a re-exported or re-encoded copy with different stored pixels, and it is not person-level deduplication. Only the imported series of each internal patient was checked.

## hourly

Source: `HOURLY_REVIEW_02.md`, lines 13–13.

Readiness sealed tranche02 corrects both attribution and availability:211 listed CT subjects ->188 imported ->158 built ->153 EGFR two-class. The23 non-imported consist of20 absent from CT+SEG downloads with no local AIM and3 AIM holders whose reference matches no listed CT series; the latter actual selection reason is unrecorded. The30 imported/unbuilt consist of26 skipped for unknown EGFR and KRAS before localization, plus4 reconstructed air-rule exclusions (3 rely on IPP-order approximation). These are reconstructed causes under frozen skip order, not recovered historical stdout. Five evaluated AIM-only cases used a series different from their annotation reference, confirming an existing M0 limitation. State this in the paper rather than implying clean corrected inputs.

## claim_attrition

Source: `codex_closeout_review/journal_set_v3/manuscript/CLAIM_EVIDENCE_reviewed_copy.md`, lines 31–31.

| 18 | Internal flow 211 → 188 → 158 → 153; the 23 not imported reflect local selection, not absent public CT | Exclusive per-subject reason table | readiness R01 (2026-10-14 tranche 02: RESULT, ATTRITION_COUNTS) | 211 source subjects | Completed with stated limitations (reconstructed 2026-10-14) | Exact enumeration; 4 air-rule exclusions are rule replays (3 approximate slice order) | Reconstructed after outcomes were known; no preprocessing log; selection reason for 3 AIM holders unrecorded | "not imported locally"; never "no CT series in the collection" or "contemporaneous log" | Data and Reference Standards; Fig 1 legend / S2 |

## source_table1

Source: `codex_closeout_review/journal_set_v3/manuscript/MAIN_PAPER_reviewed_copy.md`, lines 147–156.

**Table 1: Internal Cohort by Institution, Source Subcohort and Annotation Tier**

| Institution, subcohort (annotation tier) | EGFR mutant/ labeled | n | Age (y), median (IQR) | Women | Never-smokers | Scanner vendor, analyzed CT |
|-------------------|-----------|-----|--------------|----------|------------|-----------------|
| Stanford, AMC (point-annotated) | 17/36 | 39 | 68 (64–76) | 28 | 18 | Siemens 30, GE 8, Philips 1 |
| Stanford, R01 (lesion extent) | 15/45 | 45 | 73 (64–78) | 25 | 12 | GE 27, Siemens 11, other or blank 7 |
| VA Palo Alto, R01 (lesion extent) | 8/72 | 74 | 68 (64–71) | 3 | 5 | GE 74 |

Note.—n = scored patients. AMC is an identifier prefix of a Stanford subcohort, not a hospital. Five scored patients had an unknown label. Age is at histological diagnosis; age, sex and smoking status were recorded for all 158; ethnicity was not recorded for AMC. Vendor is from public series metadata of the analyzed series (Supplemental Material S2). Subgroup performance was not estimated.
