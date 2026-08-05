# Data request specification — THEIA external cohort

Hand this to a prospective collaborator. Most data-sharing conversations stall
because the ask is vague ("do you have any lung CTs?"), which forces the other
party to do the scoping work. This states exactly what is needed, what is
optional, and what it is for.

## The ask, in one line

Retrospective, de-identified pre-treatment chest CT plus EGFR mutation status
for NSCLC patients who had molecular testing. Target n ≥ 300.

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

Cite these to show the request is grounded, not speculative.

- Published EGFR-from-CT models reach **AUC ≈ 0.80** on external validation,
  but train on **400–950 patients** (Wu 2024 n=438; Gong 2023 n=818+131;
  Zhang 2024 n=424).
- The largest public cohort with CT **and** mutation status **and** a
  segmentation is NSCLC-Radiogenomics: **117 usable patients, 23 EGFR
  positives**. There is no second one.
- On that cohort our pooled out-of-fold AUC is **0.426, 95% CI [0.307, 0.546]**
  — an interval spanning chance. A permutation test over 200 label shuffles
  gives **p = 0.52**; a conventional radiomics baseline gives **p = 0.94**.
  Neither the model nor classical radiomics finds signal at this N.
- The learning curve is **rising** at our ceiling (AUC 0.478 → 0.549 → 0.610 as
  n goes 58 → 88 → 117, with the spread collapsing from ±0.12 to ±0.04). This
  is an underpowered signal, not an absent one.

The honest framing: **the bottleneck is labelled patients, and we have measured
that precisely rather than assumed it.**

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
