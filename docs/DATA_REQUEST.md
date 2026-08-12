# Data request specification — THEIA external cohort

Hand this to a prospective collaborator. Most data-sharing conversations stall
because the ask is vague ("do you have any lung CTs?"), which forces the other
party to do the scoping work. This states exactly what is needed, what is
optional, and what it is for.

## The ask, in one line

Retrospective, de-identified pre-treatment chest CT with **tumour segmentations**,
for NSCLC patients who had molecular testing. EGFR status is wanted but is no
longer the primary reason for the request — see below. Target n ≥ 300.

**Read this before the rest.** An earlier version of this document argued that
the bottleneck was labelled patients and that more of them would deliver an
accurate EGFR classifier. That argument is now known to be wrong, and it is
withdrawn rather than quietly edited. The Netherlands Cancer Institute has since
published 1,646 patients with 11,473 segmented lesions and NGS-confirmed EGFR
(Eur Radiol 2026, doi 10.1007/s00330-026-12601-9) and reports internal AUC
0.62–0.68 and external 0.55–0.63 — the same ceiling this project hits at n=153.
Ten times the patients did not move it. Anyone giving us data deserves to know
that up front.

## Required per patient

| Item | Detail | Why |
|---|---|---|
| **Pre-treatment chest CT** | DICOM. Axial, diagnostic (not the low-dose CT from a PET/CT, not a coronal reformat). ≤ 2.5 mm slice thickness. | The model reads axial slices; reformats break the slice axis silently. |
| **EGFR mutation status** | Mutant / wild-type, from tissue or plasma NGS or PCR. Variant detail welcome but not required. | The label. |
| **Anonymised patient ID** | Any stable key linking image to label. | Join key. |

That is the minimum viable request. Everything below improves the work but is
not a blocker — say so explicitly, because a long mandatory list kills replies.

## Strongly preferred

| Item | Why |
|---|---|
| **Tumor segmentation** (DICOM SEG or RTSTRUCT) | Supervises the grounding head. Without it the patient still trains the classifier via the AIM/centroid tier, but contributes nothing to localisation. |
| **A lesion centroid or slice number**, if no full segmentation | Cheap fallback — one click per patient in any viewer. Enough to place a crop. |
| **KRAS / ALK status** | Secondary endpoints. |
| **Histology, stage, age, sex, smoking status** | Subgroup and fairness analysis; reviewers ask. |
| **Scanner manufacturer / model / kernel** | Site and scanner effects are the main confound in multi-cohort radiomics. |

## Explicitly NOT needed

State this too — it materially lowers the perceived burden.

- No PET, no MRI, no contrast-phase series
- No pathology slides
- No treatment, response, or survival data
- No PHI: no names, MRNs, dates of birth, or institution identifiers.
  Study dates may be shifted; only the image-to-label link must survive.

## Numbers that make the case

Cite these to show the request is grounded, not speculative. Every one is
reproducible from `results/` in this repository with no data and no GPU.

**What does not work, stated first.** On NSCLC-Radiogenomics (153 labelled
patients, 40 EGFR-mutant) the pooled out-of-fold AUC is **0.618 ± 0.025** across
three seeds — against **0.794 for smoking status alone**. Adding the model on top
of five chart variables changes AUC by **−0.029 ± 0.015**, every seed's CI
including zero. A logistic regression on frozen features matches the full network
to three decimals. The probabilities score worse on Brier than predicting the
base rate for every patient. No fusion topology — hard voting, soft voting,
cross-fitted stacking — beats the chart model.

**Why more patients will not fix that.** Riley's minimum sample size for the
five-variable clinical model alone is **414**, and for the 512-feature frozen
probe **42,362**; this cohort has 153. And precision, not power, binds the
external test: at the observed prevalence a 95% CI half-width of ±0.05 needs
n=661, while the entire open world holds ~327 EGFR-labelled CT cases with ~64
positives, giving ±0.154 — an interval spanning chance *and* the published 0.80s
simultaneously.

**What does work, and what the data is actually for.** Attention supervised
against tumour segmentations localises the lesion, and it **transfers**. On 420
held-out NSCLC-Radiomics patients (Maastro, different country and scanner fleet)
the attention-mass lift is **+0.244**, beating a per-fold spatial shuffle in 25 of
15 evaluations, with pointing 0.609 against 0.476 for a centre prior and exactly
0.000 from a randomly-initialised head. None of those checkpoints had seen the
cohort, which is asserted mechanically in the test suite rather than claimed.

**So the request is for segmentations more than for labels.** A cohort with
tumour contours lets us test whether grounding holds across a third institution.
A cohort with EGFR status but no contours is much less useful to us now, which is
the opposite of what this document said a month ago.

**One thing we would find more valuable than either.** If your cohort has
*variant-level* EGFR calls rather than binary mutant/wild-type: restricting to
canonical activating variants changes 24% of positives in TCGA-LUAD and 60% in
TCGA-LUSC. Label fidelity is the lever the NKI group demonstrated, and it is
cheaper than sample size.

## What the collaborator gets

Decide with them, in writing, up front:

- Authorship position, and who is senior author
- Whether data stays on their infrastructure (we can send code to run in-house
  rather than receive images — often the fastest route past an IRB)
- Embargo before preprint
- Whether their site becomes the external validation cohort in the paper

## Logistics to settle early — not at the end

- **IRB / ethics**: retrospective de-identified imaging is often exempt or gets a
  waiver of consent, but their IRB decides, not us. Ask which pathway applies.
- **DUA**: institution-to-institution agreement. This has the longest lead time
  of anything in the project. Start it the week the conversation turns positive.
- **Transfer**: SFTP, institutional Box/Globus, or physical drive. Confirm the
  de-identification tool (e.g. the RSNA anonymiser, or DICOM Cleaner).
- **Sealed test set**: the external cohort must not be looked at until the model
  is frozen. Agree this explicitly so nobody peeks "just to sanity check".
