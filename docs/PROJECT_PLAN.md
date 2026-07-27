# THEIA: 12-Month Project Plan

**Multi-modal, interpretable, externally-validated molecular profiling for NSCLC.**

## The reframe

The earlier 4-month plan optimized for "finishable." This one optimizes for "worth doing." With a year, THEIA stops being a single-cohort CT proof of concept and becomes a unified system for non-invasive molecular profiling that a clinician could actually audit and trust.

The shift is in the claim, not just the data volume. Today, imaging-based molecular prediction is black-box, single-modality, single-institution, and unaware of its own limits, which is exactly why none of it enters clinical workflow. THEIA's thesis is that a trustworthy system has to be interpretable (grounded), multi-modal (reason across CT and tissue), generalizable (validated across institutions), and calibrated (know when to defer to biopsy). Building and validating that system, and showing its grounding recovers reproducible radiogenomic associations, is the contribution.

## What THEIA is now

Five pillars, each a deliberate step up from v1:

1. **Multi-modal.** A CT encoder and an H&E pathology encoder feed shared grounding, classification, and generation heads. TCGA provides both modalities with matched mutation calls, and mutation-from-H&E is an established large-N task (Coudray et al., 2018, predicted EGFR, STK11 and others from lung histology). The model handles patients with one modality or both, trained with modality dropout so it degrades gracefully.

2. **Multi-gene.** An actionable NSCLC panel (EGFR, KRAS, and, as prevalence allows, ALK, MET, BRAF, STK11, TP53). Well-powered genes carry the headline; the rest are exploratory. Positive counts are reported honestly per gene.

3. **Externally validated.** Trained on pooled cohorts, tested on a held-out collaborator/private cohort that is never seen in training. This single lever is what separates mid-tier from top-tier, and it was impossible in 4 months.

4. **Uncertainty-aware.** Calibrated confidence with an abstention threshold. The headline clinical figure becomes "biopsies safely avoided at a given error tolerance," which is far stronger than a bare AUC.

5. **Discovery-generating.** Aggregated grounding maps form a data-driven radiogenomic atlas: reproducible imaging phenotypes linked to molecular alterations, cross-checked against known associations and mined for novel candidates.

## Positioning (why this clears the bar)

| Model | Classify | Generate | Ground | Multi-modal | External val | Discovery |
|-------|:--------:|:--------:|:------:|:-----------:|:------------:|:---------:|
| Glio-LLaMA-Vision (2026) | yes | yes | no | no | yes | no |
| NEVA (2026) | yes | no | yes | no | partial | no |
| **THEIA v2** | **yes** | **yes** | **yes** | **yes (CT + path)** | **yes** | **yes** |

Two contributions in one paper: a methodological and clinical result (the unified, grounded, multi-modal, externally-validated, calibrated system), and a scientific result (the radiogenomic atlas). That pairing is what Nature-family reviewers reward.

## Data

| Cohort | Role | Modalities | Notes |
|--------|------|-----------|-------|
| NSCLC-RADIOGENOMICS (TCIA) | train / internal | CT, PET | 211 patients, semantic annotations for generation supervision |
| TCGA-LUAD | train / internal | CT, H&E WSI | matched mutation calls, main pathology source |
| TCGA-LUSC | train / internal | CT, H&E WSI | squamous complement |
| NSCLC-Radiomics | train / internal | CT | additional CT volume |
| **Collaborator / private cohort** | **external test (held out)** | CT (+ path if available) | never trained on, load-bearing for the top-tier claim |
| RadGenome-Chest CT | grounding pretraining | CT | 665K grounded reports, no genomic labels |

Harmonize mutation-call formats across cohorts early. Patient-level splits, no leakage, and the external cohort stays sealed until the model is frozen.

## The 12-month plan

### Q1 (Months 1-3): Foundation and single-modality baseline
- Assemble and harmonize all cohorts. Compute per-gene positive counts across the pool and lock the headline vs exploratory gene split.
- Reproduce THEIA v1 (CT-only, grounded) as the internal baseline. The existing repo already does this, so Q1 is mostly data engineering plus running what exists.
- Start IRB determinations for the reader studies and the private-cohort data use agreement, plus de-identification. This is load-bearing, do not defer it.
- **Checkpoint:** harmonized multi-cohort dataset, working CT-only baseline with cross-validated AUC and grounding IoU.

### Q2 (Months 4-6): Pathology branch and multi-modal fusion
- Pathology encoder: tile the WSIs, use a frozen pathology foundation-model backbone (UNI or CONCH) with multiple-instance aggregation. Reproduce a Coudray-style mutation-from-H&E baseline as a sanity check.
- Multi-modal architecture: CT ROI tokens and WSI tile tokens both feed the shared grounding queries, the multi-gene classifier, and the generation head. Train with modality dropout so single-modality inference works.
- **Checkpoint:** multi-modal model beats each single-modality model on internal cross-validation, with grounding functioning in both modalities.

### Q3 (Months 7-9): Generalization, uncertainty, discovery
- External validation on the collaborator cohort and held-out public cohorts. Report the internal-to-external performance drop honestly. This is the make-or-break result.
- Uncertainty and deferral: calibration, abstention threshold, and the "biopsies safely avoided at X% error" curve.
- Discovery: aggregate grounding into the radiogenomic atlas, test cross-cohort reproducibility, cross-check known associations, flag novel candidates.
- Subgroup and fairness analysis by site, scanner, stage, and demographics.
- **Checkpoint:** external validation holds (or you learn exactly where it breaks), a deferral curve, and the first atlas.

### Q4 (Months 10-12): Clinical validation, writeup, submission
- Full multi-reader blinded study across cohorts: plausibility, grounding sensibility, usefulness, and whether THEIA's output changes reader decisions. Report inter-rater agreement.
- Complete ablations (grounding on/off, generation on/off, single vs multi-modal, with/without RadGenome pretraining), robustness, and fairness.
- Manuscript, preprint (arXiv and medRxiv the day results freeze), and submit. A single revision round inside 12 months is tight, so the submission itself lands by month 12.
- **Target venues:** Nature Communications, npj Precision Oncology, npj Digital Medicine, or Medical Image Analysis (methods-heavy framing). Nature Medicine or Nature Cancer are a reach that becomes real if external validation and the deferral result are both strong.

## Risks and mitigations

- **Pathology compute.** Gigapixel WSIs are the main new cost. A frozen foundation-model backbone plus tiling keeps it on one strong GPU. Confirm GPU and storage in Q1.
- **External cohort access.** Load-bearing. You have collaborator access, so lock the data use agreement and de-identification in Q1, not Q3.
- **Multi-gene underpowering.** Some genes will have too few positives. Restrict the headline to well-powered genes and label the rest exploratory.
- **Modality missingness.** Many patients have only one modality. Modality dropout during training handles it.
- **Scope creep over a year.** The quarterly checkpoints are the guardrail. Each quarter must produce a standalone result, so if the full vision slips you still have a paper. The multi-modal internal result alone (end of Q2) is publishable on its own.

## What the repo already supports

The current codebase (grounding, generation, classification, k-fold, evaluation, reader study) is the CT-only core, which is Q1. The multi-modal branch (pathology encoder, fusion, modality dropout, multi-gene panel, external-cohort evaluation, the atlas aggregation) is the Q2 to Q3 build. Config keys for the gene panel, pathology, modality handling, and external cohorts are stubbed in `configs/default.yaml` so the scope is visible even before the code lands.
