# Pre-specified analysis plan — external validation

Written **before** any external cohort is obtained, and dated by its commit. The
point of fixing it now is that every degree of freedom left open until after the
data arrives is a degree of freedom that can be spent, unconsciously, on making
the result come out. This project has already measured how much that is worth:
the radiomics arm moves **0.136 AUC** across six defensible analytic choices, and
the same features and estimator move **0.066** between a flat and a nested
cross-validation protocol. Both are larger than the effect being looked for.

## 1. Primary hypothesis

**Does imaging add to what a clinician already has?**

> H₁: AUC(clinical + THEIA) − AUC(clinical) > 0, for EGFR mutation status.

Primary estimator: the paired difference in pooled out-of-fold AUC, with a
paired bootstrap CI over patients (2000 resamples, percentile method, α = 0.05,
two-sided). Implemented in `theia.analysis.incremental`.

**Not** "THEIA beats chance", which is already established and clinically
uninteresting, and **not** "THEIA beats clinical", which has been measured and
is false on the internal cohort.

## 2. Fixed comparators

The clinical model is **age, sex, ethnicity, smoking status, pack-years**, L1/L2
logistic regression with C and penalty chosen by inner cross-validation inside
the training portion only.

Deliberately excluded from it: %GG, tumour location and histology. Those are read
off the scan or the resection specimen; folding them in would quietly give the
"clinical" arm imaging and pathology information and make the imaging comparison
meaningless.

## 3. Pre-specified decisions

| decision | fixed value | why it is fixed here |
|---|---|---|
| primary gene | EGFR | KRAS is at chance internally; reported as exploratory only |
| primary endpoint | incremental ΔAUC vs clinical | §1 |
| training config | `configs/default.yaml` at the commit that seals the model | prevents post-hoc tuning against external data |
| seeds | 1337, 7, 42 | the same three used internally |
| headline statistic | across-seed mean ± sd | a single run's sd here is 0.041 |
| CV protocol | nested, 5-fold, stratified on EGFR | flat CV is worth +0.066 on identical features |
| pooling | within-fold rank normalisation | folds are different models; raw scores are not comparable |
| stalled folds | excluded and reported as a count | a fold that skipped every optimizer step never trained |
| missing labels | −1, masked from the loss and every metric | mapping unknown to wild-type invents negatives |
| multiplicity | primary is EGFR incremental value alone | everything else is explicitly exploratory |

## 4. Sample size

From `theia.analysis.power`, simulating on the observed cohort's prevalence
(26%) and clinical-model strength:

| imaging AUC | delivered Δ | n for 80% power |
|---|---|---|
| 0.65 (current) | +0.027 | > 1200 |
| 0.75 | +0.085 | ~300 |
| 0.85 | +0.138 | < 150 |

**At the model's current strength this study is not worth running on any
attainable cohort.** Improving the imaging arm from 0.63 to 0.75 is worth more
than quadrupling the sample. That should drive sequencing: fix the model first,
then acquire.

Treat those numbers as optimistic. The simulated arms are independent given the
labels; a real imaging model correlates with smoking status through ground-glass
morphology, so a real arm of the same standalone AUC delivers less.

## 5. What would falsify the project's premise

Stated in advance so it cannot be renegotiated later:

- The CI on the incremental ΔAUC lies entirely within ±0.02 → imaging adds
  nothing of clinical consequence over the chart, at this cohort size, and the
  radiogenomic framing should be abandoned rather than re-cut.
- A frozen-feature logistic regression matches the full model on matched folds
  → the architecture is not earning its complexity. **This is already true
  internally** (0.617 ± 0.052 vs 0.627 ± 0.041) and must be re-reported on the
  external cohort either way.
- Grounding lift does not exceed its shuffled baseline in a majority of folds →
  the localisation claim does not hold and must be dropped, not softened.

## 6. Reporting commitments

- The multi-seed mean ± sd is the headline. The pooled-over-seeds CI may appear
  beside it, never instead of it: it is tighter and absorbs no rerun variance.
- Every grounding number appears with its per-fold shuffled baseline.
- Stalled folds, excluded patients and any deviation from this plan are reported
  with counts.
- Negative and null results are reported in the same detail as positive ones.
- The rationale arm is **not** reported until it passes a confabulation check —
  it currently invents demographics absent from its input (see
  [RESULTS.md](RESULTS.md) §4b).

## 7. Deviations

Any departure from this plan is recorded here, with the date, the reason, and
whether it was decided before or after seeing the external outcome.

*(none yet)*
