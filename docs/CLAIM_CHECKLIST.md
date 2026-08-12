# CLAIM checklist

Checklist for Artificial Intelligence in Medical Imaging, as required by
*Radiology: Artificial Intelligence* at submission.

Each item is **Yes**, **No** or **NA**. Every **No** carries a written
justification — an unexplained No is what a reviewer treats as an omission rather
than a decision. Page:line references are filled in against the manuscript at
submission; until then each item points at the repository artifact that satisfies
it, so nothing has to be reconstructed from memory later.

**Status:** pre-submission. Items marked *(manuscript)* have no artifact yet
because the manuscript does not exist; they are listed so they cannot be
forgotten, not claimed as done.

---

## Title and abstract

| # | Item | Y/N/NA | Where |
|---|---|---|---|
| 1 | Identification as a study of AI methodology, specifying the category of technology used | — | *(manuscript)* |
| 2 | Structured summary of study design, methods, results, and conclusions | — | *(manuscript)* |

## Introduction

| # | Item | Y/N/NA | Where |
|---|---|---|---|
| 3 | Scientific and clinical background, including the intended use and role of the AI approach | — | *(manuscript)*; scope and intended use in [MODEL_CARD.md](MODEL_CARD.md) |
| 4 | Study objectives and hypotheses | **Yes** | [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) §1 — H₁ grounding generalisation (primary), H₂ incremental value (co-primary, pre-declared null) |

## Methods

| # | Item | Y/N/NA | Where |
|---|---|---|---|
| 5 | Prospective or retrospective study | **Yes** | Retrospective secondary analysis of a public archive. ANALYSIS_PLAN §0 |
| 6 | Study goal, such as model creation, exploratory study, feasibility study, non-inferiority trial | **Yes** | Exploratory feasibility. Article type Technical Developments. ANALYSIS_PLAN header |
| 7 | Data sources | **Yes** | NSCLC-Radiogenomics (TCIA) internal; NSCLC-Radiomics/Lung1 (Maastro) external for grounding. RESULTS §4a |
| 8 | Eligibility criteria: how, where, and when potentially eligible participants or studies were identified | **Yes** | [table1.md](table1.md) participant flow; `theia.analysis.cohort` |
| 9 | Data pre-processing steps | **Yes** | `theia/data/preprocess.py`; crop `(bbox_mm + 24) × 2.5`, `crop_jitter_frac 0.30`, HU window centre −600 width 1500, 16 slices |
| 10 | Selection of data subsets, if applicable | **Yes** | Primary analysis set = segmented adenocarcinoma (n=97). ANALYSIS_PLAN §3; rationale in RESULTS §1 |
| 11 | Definitions of data elements, with references to Common Data Elements | **Yes** | EGFR status: canonical activating variants, `theia/data/variants.py`. Internal cohort is binary-only and the asymmetry is disclosed |
| 12 | De-identification methods | **Yes** | Data arrive de-identified from TCIA. Note that `CT Date` values (1989–1996) are de-identification offsets, not dates, and no accrual window is derived from them |
| 13 | How missing data were handled | **Yes** | Labels: −1, masked from loss and every metric — never mapped to wild-type. Clinical: median-fill numeric with an indicator, explicit level for categorical. `theia/analysis/baselines.py:clinical_features` |
| 14 | Flow of participants or cases, using a diagram to indicate inclusion and exclusion | **Yes** | [table1.md](table1.md) — 211 → 188 → 158 → 153 → 133 → 97 with reasons |
| 15 | Demographic and clinical characteristics of cases in each partition | **Yes** | [table1.md](table1.md), split by EGFR status |
| 16 | Definition of ground truth reference standard, in sufficient detail to allow replication | **Yes** | Mutation status by sequencing (Gevaert et al.; Bakr et al. *Sci Data* 2018;5:180202). Grounding reference: expert tumour segmentations distributed with the collection |
| 17 | Rationale for choosing the reference standard | **Yes** | *(manuscript)* — sequencing is the clinical standard for genotype; masks are the only pixel-level reference available |
| 18 | Source of ground-truth annotations; qualifications and preparation of annotators | **Partial** | Segmentations are the collection's, drawn by a thoracic radiologist (Bakr et al.). **Inter-rater ICC is not measured** — see limitations below |
| 19 | Annotation tools | **Yes** | Collection-provided DICOM SEG and AIM v4; parsers at `theia/data/preprocess.py`, `theia/data/aim.py` |
| 20 | Measurement of inter- and intrarater variability; how discrepancies were resolved | **No** | *Justification:* the segmentations are redistributed from a public collection with one contour per lesion; no second reader exists to compute ICC against, and re-segmentation by an independent radiologist is scheduled but not complete. A mask perturbation (dilate/erode/translate) sensitivity sweep is reported instead and is labelled robustness, **not** measured rater variability |
| 21 | Sample size justification / power calculation | **Yes** | ANALYSIS_PLAN §4; `theia.analysis.power` — Hanley–McNeil `n_for_auc_ci` and Riley minimum-n, with the deliberate omission of Riley for the deep arm justified in place |
| 22 | Data partitions and how they were determined | **Yes** | Nested 5-fold stratified CV, `theia/data/dataset.py:nested_kfold_indices`. Flat CV is worth +0.066 on identical features and is not used |
| 23 | Level at which partitions were disjoint | **Yes** | Patient level |
| 24 | Model description, including inputs, outputs, and all intermediate layers | **Yes** | [MODEL_CARD.md](MODEL_CARD.md); `theia/models/theia_model.py` |
| 25 | Software libraries, frameworks, and packages | **Yes** | [requirements-lock.txt](../requirements-lock.txt) |
| 26 | Initialization of model parameters | **Yes** | BiomedCLIP ViT-B/16 pretrained, last 2 blocks unfrozen; BioGPT + LoRA. MODEL_CARD |
| 27 | Details of training approach | **Yes** | `configs/default.yaml`; OneCycleLR, gradient clipping at 1.0 with non-finite-step skipping, early stopping with `early_stop_min_epochs` |
| 28 | Method of selecting the final model | **Yes** | Composite monitor on inner validation: `[[egfr_auc, 1.0], [grounding_mass_lift, 0.5], [gen_loss, −0.01]]` |
| 29 | Ensembling techniques, if applicable | **NA** | No ensembling in the reported models. Seed-averaged ranks appear only in a clearly-labelled secondary analysis |
| 30 | Metrics of model performance | **Yes** | AUC with bootstrap CI; calibration slope/intercept, Brier with Murphy decomposition, O:E; grounding mass/pointing/area-matched IoU each against a shuffled baseline, a centre prior and a random-init head |
| 31 | Statistical measures of significance and uncertainty | **Yes** | Paired bootstrap over patients, permutation tests, sign tests; Holm correction across the 10 pooling-sweep arms |
| 32 | Robustness or sensitivity analysis | **Yes** | RESULTS §3 — radiomics moves 0.136 AUC across six defensible analytic choices; subgroup re-analysis; acquisition sensitivity |
| 33 | Methods for explainability or interpretability | **Yes** | Supervised grounding, evaluated quantitatively against three controls rather than shown as example heatmaps. RESULTS §4, §4a |
| 34 | Validation or testing on external data | **Partial** | Grounding **is** externally validated (420 NSCLC-Radiomics patients, RESULTS §4a). Classification is **not**: see limitations |

## Results

| # | Item | Y/N/NA | Where |
|---|---|---|---|
| 35 | Performance metrics for optimal model(s) on all data partitions | **Yes** | RESULTS §1, §4, §4a |
| 36 | Estimates of diagnostic accuracy and their precision | **Yes** | Bootstrap CIs throughout; multi-seed mean ± sd is the headline, never the pooled CI alone |
| 37 | Failure analysis of incorrectly classified cases | **Partial** | Five of 30 external grounding evaluations fail with pointing exactly 0.000 at peak ratios 9–24, named individually in RESULTS §4a. Per-case classification failure analysis is *(manuscript)* |

## Discussion

| # | Item | Y/N/NA | Where |
|---|---|---|---|
| 38 | Study limitations, including potential bias, statistical uncertainty, and generalizability | **Yes** | See below and RESULTS §1, §5 |
| 39 | Implications for practice, including the intended use and/or clinical role | **Yes** | MODEL_CARD scope; the model is **not** proposed for clinical use |

## Other information

| # | Item | Y/N/NA | Where |
|---|---|---|---|
| 40 | Registration number and name of registry | **NA** | *Justification:* a retrospective secondary analysis of a public de-identified imaging archive is not a clinical trial, so the ICMJE registration statement does not apply. An OSF Open-Ended Registration of the analysis plan is provided instead — ANALYSIS_PLAN §0 |
| 41 | Where the full study protocol can be accessed | **Yes** | [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), registered at OSF and archived at Zenodo |
| 42 | Sources of funding and other support; role of funders | — | *(manuscript)* |

---

## Limitations, stated plainly

These are the items above that resolve to No or Partial, collected so a reviewer
does not have to assemble them.

1. **No external validation of the classification arm.** CLAIM item 33 permits
   this if noted and justified. The justification is measured, not rhetorical:
   the entire open world holds ~327 EGFR-labelled CT cases with ~64 positives,
   and a test set of ~96 patients with ~17 positives yields a 95% CI half-width
   of ±0.154 at AUC 0.63 — an interval that spans chance and the published 0.80s
   at once. Such a test cannot confirm or refute anything, and this is stated in
   advance rather than discovered afterward. The **grounding** endpoint, which is
   primary, *is* externally validated.

2. **No inter-rater variability for the segmentations.** One contour per lesion
   exists in the source collection. A mask-perturbation sweep is reported as
   robustness and is explicitly not a substitute.

3. **The classification arm is not identifiable at this cohort size.** Riley's
   minimum n is 4,707 for the 512-feature frozen probe against 153 available.
   Consequently "the frozen probe matches the full model" is reported as
   ambiguous evidence, not as proof the architecture is redundant.

4. **Calibration is poor and is reported rather than repaired.** Slope 0.180;
   Brier 0.217–0.248 against an uncertainty floor of 0.193, so the raw
   probabilities are worse than predicting the base rate. No operating point is
   pre-specified for this reason, and any future external artifact must carry
   Platt or isotonic recalibration.

5. **Acquisition characteristics are incomplete.** Slice thickness, kernel and
   scanner vendor come from DICOM headers, and the local imaging was deleted to
   reclaim disk. Re-download from TCIA is required to populate that table.

6. **Single internal cohort, single institution, historical accrual.** The
   internal cohort is the Stanford/Palo Alto VA series. Generalisation beyond it
   is asserted only for grounding, and only to the one external cohort tested.
