# Pre-specified analysis plan — external validation

Written **before** any external cohort is obtained, and dated by its commit. The
point of fixing it now is that every degree of freedom left open until after the
data arrives is a degree of freedom that can be spent, unconsciously, on making
the result come out. This project has already measured how much that is worth:
the radiomics arm moves **0.136 AUC** across six defensible analytic choices, and
the same features and estimator move **0.066** between a flat and a nested
cross-validation protocol. Both are larger than the effect being looked for.

**Target venue:** *Radiology: Artificial Intelligence*, article type **Technical
Developments** (2000 words, 25 references, 6 figures, 2 tables). That journal
publishes no minimum-n or minimum-centre requirement; it asks for a sample-size
justification or power calculation, which §4 supplies. Precedent for a
single-institution mutation-status study without an external test set:
Jang et al., *Radiol Artif Intell* 2025;7(3):e240507 (n=274).

## 0. Registration

- OSF Open-Ended Registration ID: *(to be pasted on registration — see §8)*
- Zenodo DOI for the sealed code + model artifact: *(to be pasted at release)*

ClinicalTrials.gov registration is **not** applicable. The only registration item
in the checklist this journal accepts is CLAIM item 34 (ICMJE clinical-trial
registration), and a retrospective secondary analysis of a public de-identified
imaging archive is not a clinical trial.

## 1. Primary hypothesis

Changed from the previous version of this plan. The reason is stated in §7.

> **H₁ (primary): the grounding result generalises to an unseen cohort.**
> Attention-mass lift over a per-fold spatial shuffle, on held-out
> NSCLC-Radiomics patients, is > 0.

> **H₂ (co-primary, expected null): AUC(clinical + THEIA) − AUC(clinical) > 0**
> for EGFR mutation status.

H₂ was the sole primary endpoint until 2026-08-11. It is retained, but it is now
pre-declared as a **powered negative** rather than a hypothesis the project
expects to confirm, because two independent lines of evidence say it is not
answerable at any cohort size this project can reach:

1. Our own power table (§4): at an imaging AUC of 0.65, 80% power needs n > 1200,
   and even n = 1200 delivers only 0.62.
2. Rodríguez Sánchez et al., *Eur Radiol* 2026 (doi 10.1007/s00330-026-12601-9):
   **1,646 patients, 11,473 segmented lesions**, NGS-confirmed EGFR, reporting
   internal AUC 0.62–0.68 and external 0.55–0.63 — where their external cohort
   *is* NSCLC-Radiogenomics, this project's internal cohort. Ten times the
   patients and seventy-five times the lesions reproduce our 0.627.

Reporting H₂ as a null is therefore a finding, not a failure, and it must not be
softened into "future work with more data".

Primary estimator for H₁: mean attention-mass lift across evaluation folds, with
the per-fold shuffled baseline reported beside every number.
Primary estimator for H₂: paired difference in pooled out-of-fold AUC, paired
bootstrap CI over patients (2000 resamples, percentile, α = 0.05, two-sided).
Implemented in `theia.analysis.external_grounding` and `theia.analysis.incremental`.

## 2. Fixed comparators

The clinical model is **age, sex, ethnicity, smoking status, pack-years**, L1/L2
logistic regression with C and penalty chosen by inner cross-validation inside
the training portion only.

Deliberately excluded from it: %GG, tumour location and histology. Those are read
off the scan or the resection specimen; folding them in would quietly give the
"clinical" arm imaging and pathology information and make the imaging comparison
meaningless.

Grounding comparators, all three mandatory in every table:

| control | what it rules out |
|---|---|
| per-fold spatial shuffle | that a peaked map scores well regardless of where it points |
| centre prior | that "look at the middle" suffices — external ROI centroids sit at (0.50, 0.51) of the frame, sd ≈ 0.08, covering 5.8% of area |
| randomly-initialised head | that the metric measures architectural bias rather than learning |

## 3. Pre-specified decisions

| decision | fixed value | why it is fixed here |
|---|---|---|
| primary gene | EGFR | KRAS is at chance internally with 32 positives against EGFR's 40 — equal power, different outcome, so this is signal not power |
| primary endpoint | grounding external generalisation | §1 |
| co-primary | incremental ΔAUC vs clinical, pre-declared null | §1 |
| **primary analysis set** | **adenocarcinoma only** | all 43 EGFR-mutants are adenocarcinoma; the 35 squamous and 4 NSCLC-NOS patients are wild-type without exception, so 20 patients (13%) are guaranteed-negative on histology alone. `MODEL_CARD.md` already declares non-adenocarcinoma out of scope |
| supervision tier | segmented-only primary; full cohort with a `tier` covariate as sensitivity | `has_mask` scores 0.583 on its own through these folds, so tier is readable from pixels and is a confound, not a nuisance |
| acquisition sensitivity | ≤1.5 mm slice thickness (161/188 series), exploratory | pre-declared so it cannot be promoted later |
| training config | `configs/default.yaml` at the commit that seals the model | prevents post-hoc tuning against external data |
| seeds | 1337, 7, 42 | the same three used internally |
| headline statistic | across-seed mean ± sd | a single run's sd here is 0.041 |
| CV protocol | nested, 5-fold, stratified on EGFR | flat CV is worth +0.066 on identical features |
| pooling | within-fold rank normalisation, recomputed within any subgroup | a rank is a statement about the patients being compared |
| operating point | **none** — index test is continuous | at AUC 0.63 a 90%-sensitivity point has ~20% specificity and is clinically meaningless; STARD cross-tabulation is NA and the `_sens_spec` key is removed |
| stalled folds | excluded and reported as a count; the run is re-launched rather than analysed around | a fold that skipped every optimizer step never trained |
| missing labels | −1, masked from the loss and every metric | mapping unknown to wild-type invents negatives |
| external label definition | canonical activating variants only, harmonised via cBioPortal | ~10% of "EGFR-mutant" calls in TCGA/CPTAC are non-kinase-domain passengers (L62R, R222L, E545Q, K479I, H358R, E84K, I143L, A871G). The internal cohort has binary labels only; this asymmetry is disclosed, not hidden |
| class weighting | deep arm uses unweighted cross-entropy, baselines use `class_weight='balanced'` | disclosed as a modelling difference; every endpoint is AUC, and weighting barely moves a logistic regression's ranking |
| multiplicity | primary is H₁ alone; H₂ co-primary; everything else exploratory | — |
| pooling-sweep multiplicity | within-cohort paired bootstrap with Holm across the 10 arms | the across-seed sign test treats 10 re-splits of one 153-patient cohort as replicates, which they are not |

## 4. Sample size

From `theia.analysis.power`, simulating on the observed cohort's prevalence
(26%) and clinical-model strength:

| imaging AUC | delivered Δ | n for 80% power |
|---|---|---|
| 0.65 (current) | +0.027 | > 1200 |
| 0.75 | +0.085 | ~300 |
| 0.85 | +0.138 | < 150 |

**At the model's current strength H₂ is not worth running on any attainable
cohort.** That is why H₁ is now primary.

Treat those numbers as optimistic. The simulated arms are independent given the
labels; a real imaging model correlates with smoking status through ground-glass
morphology, so a real arm of the same standalone AUC delivers less.

For the external test set specifically, precision — not power — is the binding
constraint. The open world is ~327 EGFR-labelled CT cases with ~64 positives
(TCGA-LUAD 60/10, CPTAC-LUAD 36/10, NSCLC-Radiogenomics 153/40, squamous
collections ~59/1). A test set with ~17 positives has a Hanley–McNeil 95% CI of
roughly ±0.14 at AUC 0.63 — it spans chance *and* spans the published 0.80s.
**"External testing was attempted and is uninformative at the available scale" is
the pre-specified finding**, not a reason to defer.

## 5. What would falsify the project's premise

Stated in advance so it cannot be renegotiated later.

- The CI on the incremental ΔAUC lies entirely within ±0.02 → imaging adds
  nothing of clinical consequence over the chart, at this cohort size, and the
  radiogenomic framing should be abandoned rather than re-cut.
- A frozen-feature logistic regression matches the full model on matched folds
  → the architecture is not earning its complexity. **This is already true
  internally** (0.617 ± 0.052 vs 0.627 ± 0.041) and must be re-reported on the
  external cohort either way.
- **Grounding lift on the external cohort falls below +0.20, or beats its
  shuffled baseline in fewer than 10/14 (71%) of evaluation folds, or fails to
  beat the centre prior on pointing, or a randomly-initialised head produces a
  non-trivial lift** → the localisation claim does not hold and must be dropped,
  not softened. With H₁ primary, this is now the falsification that matters most.

  Note on the centre prior: it is a strong baseline on *mass* by construction and
  a weak one on *pointing* and area-matched IoU, which are invariant to its
  width. The gate is therefore set on pointing, where the prior has no built-in
  advantage, rather than on mass, where beating it slightly would mean little.

## 6. Reporting commitments

- The multi-seed mean ± sd is the headline. The pooled-over-seeds CI may appear
  beside it, never instead of it: it is tighter and absorbs no rerun variance.
- Every grounding number appears with all three controls from §2.
- The primary analysis set (segmented adenocarcinoma, n = 97, 23 positives)
  carries the headline; the full cohort appears as sensitivity. Both numbers are
  reported whichever way they fall — internally that costs **−0.056**
  (0.627 → 0.572).
- Calibration is reported: calibration plot, slope, intercept, Brier score and
  O:E ratio, computed from the archived per-patient probabilities.
- Stalled folds, excluded patients and any deviation from this plan are reported
  with counts.
- Negative and null results are reported in the same detail as positive ones.
- The rationale arm is **not** reported until it passes a confabulation check —
  it currently invents demographics absent from its input (see
  [RESULTS.md](RESULTS.md) §4b).
- Prior reports of this patient population are cited: Gevaert et al.
  *Radiology* 2012 and *Sci Rep* 2017, and Bakr et al. *Sci Data* 2018;5:180202.
  The accrual window is **not** computed from `clinical.csv`'s "CT Date" column —
  its values span 1989-11-07 to 1996-09-11 and are de-identification offsets, not
  real dates. The index-test-to-reference-standard interval is reported instead
  (median 38 days, IQR 19.5–65.5).

## 7. Deviations

Any departure from this plan is recorded here, with the date, the reason, and
whether it was decided before or after seeing the external outcome. **No external
cohort has been obtained, so every entry below predates any external outcome.**

**2026-08-11 — primary endpoint changed from H₂ to H₁.** Decided after seeing
internal results and the Rodríguez Sánchez et al. external replication, before
any external cohort was obtained. Reason in §1. H₂ is retained as co-primary and
pre-declared null rather than dropped, so the change cannot be read as discarding
an endpoint that failed.

**2026-08-11 — Gate A on the semantic oracle was reframed post-hoc.** The
pre-specified gate was `AUC(all semantic) − AUC(in-crop only) ≥ 0.06`; it
returned **+0.035, a FAIL**. The contrast actually intended — whole-lung features
against in-crop features, same folds, same estimator — returns +0.053 (10/10
seeds, decision tree) and +0.077 (9/10, logistic regression). The reframe is
better statistics, because the pre-specified version is a nested-model increment
whose variance at n = 158 with 100 features swamps the effect. It is nonetheless
**post-hoc** and is reported as such; the pre-specified failure is reported
alongside it.

**2026-08-11 — the field-of-view rebuild was cancelled.** Kill criterion K1 fired.
Probing frozen crop tokens recovers emphysema at 0.733, fibrosis 0.679 and airway
abnormality 0.626, against in-crop control attributes averaging 0.634 (whole-lung
mean 0.635, gap −0.001). The whole-lung parenchymal reads that drive Gevaert's
0.89 are already legible inside the existing `(bbox_mm + 24) × 2.5` crop, so
widening the field of view is not the lever.

**2026-08-11 — Gate D's fold threshold restated as a proportion.** The plan wrote
it as "10 of 14 evaluation folds", a count borrowed from the internal run's fold
total. The external evaluation runs over however many run × fold checkpoints
exist (currently 30), so a fixed count of 10 would be a far weaker bar than
intended. Restated as the same proportion, 10/14 = 71%. Decided after a 16-patient
smoke run confirmed the harness worked but **before** the full 420-patient result
was computed. The gate was also tightened, not loosened, in the same edit: it now
additionally requires beating the centre prior on pointing.

**2026-08-11 — the peritumoral architecture result is reported as suggestive, not
established.** `model.peritumoral_features` moves pooled EGFR AUC from 0.627 to
0.675 (+0.047, 3/3 seeds). Every per-seed paired bootstrap CI includes zero
(p = 0.59 / 0.21 / 0.10). Pooling seeds gives +0.028 (p = 0.34, n = 122) or
+0.068 (p = 0.048, n = 153) depending on whether seed 1337 is included — and
seed 1337's *baseline* had a stalled fold. Rather than choose the analysis that
clears p < 0.05, the degenerate baseline run was relaunched. Whichever way it
lands is what gets reported.

## 8. Actions required before registration

These are the author's, not the analysis's, and none can be performed by an
automated agent:

1. Create an OSF account and register this file as an Open-Ended Registration.
2. Authorise the GitHub–Zenodo integration and cut a release to mint the DOI.
3. Obtain written IRB determinations: not-human-subjects-research for the
   retrospective public-archive analysis, and exempt/approved for the reader
   study with reader consent language.
4. Accept the TCIA Data Usage Agreement needed to pull TCGA-LUAD and CPTAC-LUAD.

Registration happens **after** items 1–2 and **before** any external cohort is
opened. Paste the identifiers into §0 and commit.
