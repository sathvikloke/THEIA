# THEIA — results of record

What is measured, what it means, and which numbers are safe to quote. Written
because the project's framing changed materially and the evidence for that
change was scattered across commit messages, logs and a temp directory.

Cohort throughout: NSCLC-Radiogenomics, 158 preprocessed patients, 153 with a
known EGFR call, 40 of them mutant (26%). All estimates are pooled out-of-fold
over 5 nested folds, with within-fold rank normalisation, and bootstrap CIs.

---

## 1. The headline, and the number that undercuts it

| model | EGFR AUC | 95% CI |
|---|---|---|
| smoking status alone | **0.794** | [0.695, 0.882] |
| clinical (age, sex, ethnicity, smoking, pack-years) | 0.764–0.774 | [0.665, 0.859] |
| clinical + radiomics | 0.783 | [0.690, 0.869] |
| **THEIA** (run 12) | **0.660** | [0.556, 0.758] |
| radiomics | 0.526–0.662 | see §3 |

THEIA's CI excludes chance. It is also **beaten by a single chart variable.**

In this cohort never-smokers are 60.6% EGFR-mutant against 8.3% (current) and
18.8% (former); OR 7.69, Fisher exact p = 1.7e-6. This is the textbook
epidemiology, not an artefact.

## 2. The question that decides the paper

Imaging is only interesting if it adds something a clinician does not already
have. Paired bootstrap on identical folds, so every patient contributes to both
arms and the shared cohort variance cancels:

| comparison | ΔAUC | 95% CI | p |
|---|---|---|---|
| **(clinical + THEIA) − clinical** | **−0.006** | **[−0.066, +0.048]** | **0.832** |
| THEIA − clinical | −0.105 | [−0.220, +0.002] | 0.058 |
| THEIA − (clinical + radiomics) | −0.125 | [−0.235, −0.021] | 0.014 |
| THEIA − radiomics | +0.081 | [−0.044, +0.202] | 0.189 |

**THEIA adds nothing on top of clinical variables.** The interval is tight
around zero — this is a reasonably precise null, not an underpowered shrug: the
true increment lies between −6.6 and +4.8 AUC points. THEIA is significantly
worse than clinical+radiomics (p = 0.014), and beats radiomics alone
non-significantly.

This does not say the imaging is uninformative in principle. It says that on
158 patients, whatever EGFR signal this model extracts from CT is already
carried by smoking status.

## 3. Radiomics is dominated by analytic choices, not biology

Six pre-specifiable pipelines, same folds, same estimator family:

| feature bank | L2 | L1 | L1+L2 search |
|---|---|---|---|
| legacy, 18 features | 0.662 | 0.605 | 0.604 |
| rich, 61 features | 0.604 | 0.526 | 0.568 |
| clinical, 14 features | 0.774 | 0.773 | 0.764 |

Radiomics spans **0.136 AUC** across these choices; clinical spans **0.010**.
The spread on the radiomics arm is comparable to the entire effect size this
literature reports. Two independent attempts to strengthen it — more features,
then LASSO selection as the field does it — both made it worse.

Scope: six configurations, one cohort, n=153. This does **not** show published
radiomics results are wrong; cohorts differ (many are Asian cohorts with ~50%
EGFR prevalence against our 26%) and many use proper validation. It does show
that an estimate this sensitive to analytic choice needs nested validation
before it means anything.

## 4. Grounding

The methodological contribution, and it is real but unstable.

| run | grounding lift | folds localising | note |
|---|---|---|---|
| 6 | +0.235 | 4/5 | pre-clipping, log lost |
| 12 | +0.001 | 0/5 | **loss bug, see §5** |
| 13 | +0.180 | 2/5 | 2 folds died in warmup |

Every lift is reported against a per-fold shuffled baseline, and
`grounding_peak_ratio` flags attention that is flat. That matters: run 12 scored
0.65–1.00 on the *pointing game* while its attention map was literally constant.
A localisation metric without a chance baseline would have called that a success.

## 5. Runs and their trustworthiness

| run | config | EGFR | status |
|---|---|---|---|
| 6 | no pretraining | 0.656 [0.560, 0.747] | **weights destroyed**, log lost, divergence status unknown |
| 7 | — | — | fold 0 diverged to NaN |
| 8 | warm-start vision+grounding | 0.528 | **confounded** — 1–2 dead folds |
| 9 | warm-start grounding head | 0.597 [0.496, 0.696] | clean |
| 10 | no pretraining | — | discarded, fold 0 NaN at ep8 |
| 11 | + grad clipping | — | discarded, every step skipped |
| 12 | + clipping, conditioned loss | 0.660 [0.556, 0.758] | clean classification, **grounding dead** |
| 13 | + logit-BCE loss | 0.621 [0.513, 0.722] | 2/5 folds died in warmup |
| 14 | + warmup floor | running | — |

Run-to-run spread on essentially the same configuration is ~0.04 AUC (0.621,
0.656, 0.660). A single run's third decimal is not meaningful; the paper should
report a multi-seed mean.

## 6. Bugs that changed results

Recorded because each produced a plausible number rather than an error.

- **No gradient clipping, and off CUDA `GradScaler` is disabled**, so nothing
  skipped non-finite steps. One bad batch wrote NaN into the weights and the
  fold trained on garbage to completion. Killed folds in runs 7, 8 and 10.
- **Results were overwritten every run.** `checkpoints/fold{k}/` and
  `runs/cv_summary.json` are gitignored and unversioned; run 6, the best result
  the project had, survives only as a printed AUC. Runs are now namespaced and
  each writes a tracked `results/<run_id>.json` carrying out-of-fold predictions,
  which is enough to rebuild every curve without the weights.
- **A fixed point in the grounding loss, self-inflicted.** Normalising by the
  max pins the peak cell at the top of the range; the original `amax + 1e-6`
  left it at 0.9998, just under the BCE clamp. Replacing that with
  `clamp_min(1e-3)` put it exactly *on* the clamp, where the gradient is zero,
  and detaching the denominator removed the last escape route. A flat map then
  had every cell clamped at once: loss 9.30, gradient 0.000. Fixed by moving to
  `binary_cross_entropy_with_logits`, which has no clamp and therefore no dead
  zone.
- **Early stopping counted patience from epoch 0**, an untrained model at 4% of
  peak LR whose validation AUC is noise on ~24 patients. Run 13 lost 2 of 5
  folds to a lucky epoch-0 draw, one of them while grounding was still climbing.
- **Multi-segment DICOM SEG.** NSCLC-Radiomics bundles tumor, lungs, cord,
  oesophagus and heart into one object; reading it naively makes "tumor" mean
  the whole thorax, silently.
- **Single-class folds** make within-fold rank pooling meaningless while looking
  normal. Now warned.

## 7. What follows

The current framing — "THEIA predicts EGFR from CT" — is not supportable. §2 is
the reason, and a reviewer will find it in the first round.

Defensible directions, in order of strength:

1. **Incremental value, honestly reported.** The null in §2 is a real result,
   properly measured on identical folds with a paired test. It needs the larger
   cohort to become a *tight* null rather than a moderately tight one.
2. **The grounding method.** Attention supervised against masks, scored against
   per-fold shuffled baselines, with a flatness check that catches degenerate
   maps. §4 shows why that instrumentation matters.
3. **Measurement rigour as a contribution.** §3 and §6 together are an argument
   that this task's published effect sizes are within the range that analytic
   choices and silent bugs can produce.

What will not fix it: more architecture. The learning-curve and permutation
diagnostics already pointed at labelled data as the bottleneck, and §2 says the
imaging is competing with a variable that is free.
