# Model card — THEIA

Research prototype. **Not a medical device**, not reviewed or approved by any
regulatory body, and not fit to inform clinical care. This card is written to be
readable by someone deciding whether to build on the work, so the limitations
come before the capabilities.

## What it is

A vision–language model that takes a lung CT crop centred on an annotated lesion
and produces three outputs: a per-gene mutation probability (EGFR, KRAS), an
attention map over the patch grid supervised against the tumour segmentation, and
a free-text rationale.

- **Vision encoder** BiomedCLIP ViT-B/16, frozen except the last 2 blocks
- **Grounding head** 8 learned region queries cross-attending to patch tokens
- **Classifier** MLP over pooled region embeddings
- **Generation head** BioGPT with LoRA, conditioned on a visual prefix
- **Input** 16 slices through the lesion, 224×224, lung window (−600/1500 HU),
  square crop at 2.5× the lesion bounding box with per-patient jitter

## What it was trained and evaluated on

**NSCLC-Radiogenomics** (TCIA): 158 preprocessed patients, 153 with a known EGFR
call, 40 mutant (26%). Western cohort. Segmentation-tier patients carry a tumour
mask; AIM-tier patients carry only an annotated point and are excluded from
grounding supervision.

Evaluation is nested 5-fold cross-validation stratified on EGFR, pooled
out-of-fold with within-fold rank normalisation, repeated over 3 seeds. No
external cohort has been tested.

## Performance

| | EGFR AUC |
|---|---|
| **THEIA, 3 seeds** | **0.617 ± 0.024** (range 0.597–0.643) |
| THEIA + peritumoral branch, 3 seeds | 0.675 ± 0.017 — suggestive, see RESULTS §1 |
| frozen BiomedCLIP + logistic regression, same folds | 0.617 ± 0.052 |
| clinical (age, sex, ethnicity, smoking, pack-years) | 0.764–0.805 |
| smoking status alone | 0.794 |

**Grounding, internal**, 15 scored folds: attention mass in ROI 0.362 ± 0.228
against a shuffled baseline of 0.037 — 9.7× chance, beating its own baseline in
12/15 folds (sign test p = 0.018). Pointing game 0.708 vs 0.037. All three
failing folds belong to the retrained seed-1337 run; see RESULTS §4.

**Grounding, external** — the primary endpoint. 420 held-out NSCLC-Radiomics
patients (Maastro Clinic, Netherlands), 30 evaluations: mass lift +0.297 ± 0.167,
beating the shuffle in 25/30. Pointing 0.688 against 0.476 for a centre prior and
0.046 for the shuffle; a randomly-initialised head returns +0.000.

**Calibration is poor and the probabilities should not be used.** Slope 0.180
against a perfect 1.0; Brier 0.217–0.248 against a base-rate floor of 0.193, so
the raw outputs are worse than predicting the prevalence for every patient. No
operating point is defined, and none should be inferred.

KRAS is at chance and is exploratory only. Gevaert et al. (Sci Rep 2017) also
report no significant KRAS model on this cohort.

## Limitations, in the order that matters

**It adds nothing to a clinician's existing information.** Paired per seed,
(clinical + THEIA) − clinical = **−0.028 ± 0.016**, with every seed's CI
including zero. Whatever EGFR signal it extracts from CT is already carried by
smoking status, which is free and in the chart.

**It does not beat a linear probe on its own frozen features** (0.617 ± 0.052 on
identical folds). The architecture is not currently earning its complexity.

**The rationale head must not be shown to anyone.** It confabulates — inventing
an age, a sex and a referral history absent from its input — and its text is not
significantly patient-specific (permutation p ≈ 0.07, 23% distinct). Both are
enforced by a gate in `theia.analysis.confabulation`; `build_cases` refuses to
produce reader materials that fail it.

**Single cohort, single site distribution, no external validation.** 40 positives
is small. The rerun spread of ±0.041 is comparable to many published effects in
this literature.

**Training is numerically unstable on Apple Silicon.** The backward pass through
the unfrozen ViT blocks can go NaN while every loss stays finite; an affected
fold trains not at all. It is detected and the fold is marked stalled rather than
reported. Prefer CUDA.

**Not validated for**: screening, patients without a lesion already identified
and annotated, non-adenocarcinoma histology, paediatric patients, scanners or
protocols unlike this cohort's, or any clinical decision.

## Intended use

Methods research on grounded radiogenomics, and as a measured baseline for
whether imaging adds to clinical variables for EGFR status. Explicitly **not**
intended to triage patients, to substitute for or defer molecular testing, or to
be run on patient data outside a research protocol.

## Ethical and data considerations

Trained on de-identified public TCIA data under that archive's Data Usage
Policies; no data is redistributed here. EGFR prevalence and its association
with smoking, sex and ethnicity differ substantially between populations — a
model this dependent on smoking-correlated signal should be expected to transfer
poorly to cohorts with a different smoking distribution, and that is a fairness
concern, not only an accuracy one. A confident but wrong molecular call could
delay or misdirect targeted therapy, which is the harm to design against.

## Reproducing

```bash
python -m theia.analysis.aggregate --pattern 'results/ms-s*.json'
python -m theia.analysis.incremental --pattern 'results/ms-s*.json'
```

Both run from archived out-of-fold predictions with no data, GPU or model
downloads. Full provenance in [RESULTS.md](RESULTS.md); the pre-specified plan
for external validation is in [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).
