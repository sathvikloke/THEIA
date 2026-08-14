# TRIPOD+AI checklist

Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for
reporting clinical prediction models that use regression or machine learning
methods. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378

TRIPOD+AI **supersedes TRIPOD 2015**, which should no longer be used. It applies
here because this is a prediction-model study, whatever else it also is. Item text
is quoted from the published checklist; the response column is ours.

This study is a **development-and-internal-evaluation** study for the
classification endpoint (items marked D and D;E) and an **evaluation** study for
the external attention endpoint. Items marked **E-only** are answered where the
external evaluation makes them meaningful and marked NA where they concern
evaluating a *prediction* model externally, which we do not do.

Page:line references are filled in against the typeset manuscript at submission.
Until then each row points at the artifact that satisfies it.

**Every NA carries a reason.** An unexplained NA is what a reviewer reads as an
omission rather than a decision.

---

## Title and abstract

| # | Item | Response |
|---|---|---|
| 1 | Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted | **Yes.** Title names the outcome (*EGFR* prediction), the population (CT cohorts, NSCLC), and that the finding is an evaluation, not a proposal |
| 2 | See TRIPOD+AI for Abstracts checklist | **Yes.** Structured abstract, Purpose / Materials and Methods / Results / Conclusion, 246 words |

## Introduction

| # | Item | Response |
|---|---|---|
| 3a | Explain the healthcare context and rationale for developing or evaluating the prediction model, including references to existing models | **Yes.** Introduction ¶1–2; the AUC 0.80–0.89 literature and the discrepant large external study are both cited |
| 3b | Describe the target population and the intended purpose of the prediction model in the context of the care pathway, including its intended users | **Yes.** Introduction ¶1 (TKI selection when tissue is limited). Discussion states the model is **not** proposed for clinical use, so there is no intended user |
| 3c | Describe any known health inequalities between sociodemographic groups | **Partial.** *EGFR* prevalence differs markedly by ancestry and smoking history, and Table 1 shows it in this cohort (Asian 22% of mutants vs 5% of wild-type). We report the cohort composition but do not review the inequality literature |
| 4 | Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both) | **Yes.** ANALYSIS_PLAN §1: H₁ external attention alignment, H₂ incremental value pre-declared null |

## Methods

| # | Item | Response |
|---|---|---|
| 5a | Describe the sources of data separately for the development and evaluation datasets, the rationale for using these data, and representativeness | **Yes.** Methods, *Study design and cohorts*. NSCLC-Radiogenomics (development and internal evaluation), NSCLC-Radiomics (external attention evaluation) |
| 5b | Specify the dates of the collected participant data, including start and end of participant accrual | **No.** *Reason:* the collection's CT dates span 1989–1996 and are de-identification offsets, not dates. Publishing a range derived from them would fabricate an accrual window. The index-test-to-reference-standard interval, which survives the offset, is reported instead: median 38 days (IQR, 19.5–65.5) |
| 6a | Specify key elements of the study setting including the number and location of centres | **Yes.** Single-centre surgical series (internal); single Dutch centre, radiotherapy-planning population (external) |
| 6b | Describe the eligibility criteria for study participants | **Yes.** Figure 1 and Supplement S2 give every exclusion with its reason |
| 6c | Give details of any treatments received, and how they were handled | **NA.** *Reason:* this is a diagnostic-classification study on pre-treatment imaging; no treatment variable enters any model |
| 7 | Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups | **Partial.** Preprocessing is fully specified (Methods, *Model*; Supplement S1) and is identical for every patient by construction. We did **not** test whether preprocessing failures differ by sociodemographic group; Supplement S2 reports that exclusions differ by **histology** (31.0% vs 11.1% squamous) |
| 8a | Clearly define the outcome that is being predicted and the time horizon, including how and when assessed, the rationale, and consistency across groups | **Partial.** Binary *EGFR* status by sequencing (Methods, *Reference standard*). **Consistency of assay across patients cannot be verified** — see 8b |
| 8b | If outcome assessment requires subjective interpretation, describe the qualifications and demographic characteristics of the outcome assessors | **No.** *Reason:* sequencing is not subjective, but the collection publishes no platform, panel, specimen type or variant-level call, so the assay cannot be characterised or harmonised per patient. Stated in Supplement S4 |
| 8c | Report any actions to blind assessment of the outcome to be predicted | **NA.** *Reason:* outcomes were assigned by the source investigators before this analysis existed and could not be influenced by it |
| 9a | Describe the choice of initial predictors and any pre-selection before model building | **Yes.** Imaging arm: no predictor selection, the encoder consumes the crop. Clinical comparator: five variables fixed a priori (age, sex, ethnicity, smoking status, pack-years) |
| 9b | Clearly define all predictors, including how and when they were measured | **Yes.** Methods, *Statistical analysis*; Supplement S1 |
| 9c | If predictor measurement requires subjective interpretation, describe the qualifications of the predictor assessors | **Partial.** The tumour segmentation that defines the crop was drawn by the source collection's reader. One contour per lesion exists, so interreader agreement cannot be computed; a mask perturbation sweep is reported as robustness |
| 10 | Explain how the study size was arrived at, and justify that it was sufficient. Include details of any sample size calculation | **Yes, and the answer is that it is not sufficient.** Results, *More patients would not settle it*: Riley's criterion requires 1159 for the clinical model on 14 fitted parameters; this cohort has 153. The size was not chosen — it is the collection's size |
| 11 | Describe how missing data were handled. Provide reasons for omitting any data | **Yes.** Labels: −1, masked from loss and every metric, never mapped to wild-type. Clinical: median-fill with a missingness indicator, explicit level for categorical. No patient is dropped for missingness |
| 12a | Describe how the data were used in the analysis, including whether the data were partitioned, considering sample size requirements | **Yes.** Nested five-fold CV stratified on *EGFR*, three seeds, patient-level disjoint |
| 12b | Describe how predictors were handled in the analyses (functional form, rescaling, transformation, standardisation) | **Yes.** Supplement S1 (HU windowing, resizing, jitter); clinical encoding in Methods |
| 12c | Specify the type of model, rationale, all model building steps including any hyperparameter tuning, and method for internal validation | **Yes.** Supplement S1 gives the full configuration and states that **no hyperparameter search was performed** and nothing was tuned against a reported endpoint |
| 12d | Describe if and how heterogeneity in model parameter values and performance was handled across clusters | **Yes, and it is a finding.** Heterogeneity across *training runs* dominates: the patient-level analysis is significant with runs fixed and not significant once the run is a random effect (Results) |
| 12e | Specify all measures and plots used to evaluate model performance | **Yes.** AUC with bootstrap CI, calibration slope/intercept, Brier with Murphy decomposition, O:E; attention mass / pointing / area-matched IoU each against three controls |
| 12f | Describe any model updating (eg, recalibration) arising from the model evaluation | **NA.** *Reason:* no recalibration was applied. Calibration is reported as measured (slope 0.19) rather than repaired, and the manuscript states any future externally validated artifact must carry explicit recalibration |
| 12g | For model evaluation, describe how the model predictions were calculated | **Yes.** Archived out-of-fold predictions in `results/*.json`; every reported number regenerates from them by `scripts/reproduce.sh` without imaging or a GPU |
| 13 | If class imbalance methods were used, state why and how, and any subsequent recalibration | **NA.** *Reason:* no class-imbalance method was used. Prevalence is 26.1%; folds are stratified on *EGFR*, which preserves rather than alters the balance. This is deliberate — resampling would further damage already-poor calibration |
| 14 | Describe any approaches used to address model fairness and their rationale | **No.** *Reason:* no fairness intervention was applied and none is claimed. With 40 positives, subgroup performance by ancestry or sex cannot be estimated at useful precision, and reporting it would invite over-interpretation. The cohort composition is in Table 1 |
| 15 | Specify the output of the prediction model. Provide details and rationale for any classification and how thresholds were identified | **Yes.** Output is a continuous probability. **No operating point is defined** and no classification is issued, because the calibration does not support a threshold; sensitivity and specificity are therefore not reported |
| 16 | Identify any differences between the development and evaluation data in healthcare setting, eligibility criteria, outcome, and predictors | **Yes.** Methods: different country, scanner fleet and clinical population (radiotherapy planning vs surgical). Note the external cohort is used **only** for the attention endpoint — it carries no *EGFR* labels, which is itself the reason the classification arm has no external validation |
| 17 | Name the institutional research board or ethics committee that approved the study and describe the participant informed consent or the waiver | **NO — OUTSTANDING.** This is the manuscript's remaining submission blocker. Methods carries a marked placeholder. Public de-identified TCIA data will almost certainly attract an exempt or not-human-subjects determination, but the board issues it, not the authors |

## Open science

| # | Item | Response |
|---|---|---|
| 18a | Give the source of funding and the role of the funders | **Outstanding.** Title page field, author action |
| 18b | Declare any conflicts of interest and financial disclosures for all authors | **Outstanding.** Title page and cover letter; ICMJE forms per author |
| 18c | Indicate where the study protocol can be accessed or state that a protocol was not prepared | **Partial.** [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) is public in-repo with its deviation log. The OSF registration ID is **pending** — §0 still reads *(to be pasted)* |
| 18d | Provide registration information, or state that the study was not registered | **Partial.** Not a clinical trial, so ICMJE trial registration does not apply. An OSF Open-Ended Registration is the substitute and is pending |
| 18e | Provide details of the availability of the study data | **Yes.** Both collections are public via The Cancer Imaging Archive under stated licences; no imaging is redistributed by us. Archived out-of-fold predictions are in the repository |
| 18f | Provide details of the availability of the analytical code | **Yes.** All analysis code is in the public repository, including the runs that failed and the deviations that went against us. Anonymized mirror at submission |

## Patient and public involvement

| # | Item | Response |
|---|---|---|
| 19 | Provide details of any patient and public involvement, or state no involvement | **None.** *Reason stated:* this is a secondary analysis of an existing public archive with no participant contact and no new data collection. No patient or public involvement occurred in design, conduct, reporting or dissemination |

## Results

| # | Item | Response |
|---|---|---|
| 20a | Describe the flow of participants, including the number with and without the outcome. A diagram may be helpful | **Yes.** Figure 1, generated from the pipeline rather than transcribed, reconciling 211 − 114 = 97 |
| 20b | Report the characteristics overall and for each data source, including key dates, key predictors, sample size, number of outcome events, and missing data. Report any differences across key demographic groups | **Yes.** Table 1 by *EGFR* status; Supplement S2 compares analysed with excluded patients and reports the histology difference |
| 20c | For model evaluation, show a comparison with the development data of the distribution of important predictors | **Partial.** The external cohort is characterised qualitatively (country, scanner fleet, treatment setting) and by ROI geometry. A predictor-distribution comparison is not meaningful: the external cohort has no *EGFR* labels and is used only for the attention endpoint |
| 21 | Specify the number of participants and outcome events in each analysis | **Yes.** 153 patients / 40 events for classification; 97 / 23 for the histology-restricted analysis; 420 for the external attention endpoint. Each arm's *n* is printed on Figure 2 because arms do not all cover the same patients |
| 22 | Provide details of the full prediction model to allow predictions in new individuals and third party evaluation, including restrictions | **Partial.** Code and configuration are public and the model is reconstructible. **Trained weights are not distributed**: the calibration is poor enough that a deployable artifact would be misused, and the manuscript says no operating point exists. Checkpoints are available on request for verification |
| 23a | Report model performance estimates with confidence intervals, including for any key subgroups | **Yes.** Bootstrap CIs throughout; across-seed mean ± SD is the headline, never a pooled CI alone. Histology subgroup reported (0.563 on segmented adenocarcinoma) |
| 23b | If examined, report results of any heterogeneity in model performance across clusters | **Yes.** Table 3 reports every run × fold evaluation individually rather than summarising them |
| 24 | Report the results from any model updating | **NA.** *Reason:* no model updating was performed (see 12f) |

## Discussion

| # | Item | Response |
|---|---|---|
| 25 | Give an overall interpretation of the main results, including issues of fairness, in the context of the objectives and previous studies | **Yes.** Discussion ¶1–4, including convergence with the largest external study and what the attention endpoint does and does not establish |
| 26 | Discuss any limitations of the study and their effects on biases, statistical uncertainty, and generalisability | **Yes.** Discussion, *Limitations*, plus Supplement S2 and S5. The binding limitation — that transfer does not survive treating the training run as a random effect — is stated against our own interest |
| 27a | Describe how poor quality or unavailable input data should be assessed and handled when implementing the model | **NA.** *Reason:* the model is not proposed for implementation. Stated explicitly in the Discussion |
| 27b | Specify whether users will be required to interact in the handling of the input data or use of the model, and what expertise is required | **NA.** *Reason:* as 27a. Note the model requires a lesion segmentation as input, so it could not be deployed without a reader or an upstream detector in any case |
| 27c | Discuss any next steps for future research, with a view to applicability and generalisability | **Yes.** Discussion: an independent *EGFR*-labelled external cohort is the only thing that would move the ceiling, and the measured precision bound says what it would need to be worth doing |

---

## Outstanding before submission

1. **Item 17, ethics.** The only hard blocker.
2. **Items 18a, 18b** — funding, conflicts, ICMJE forms.
3. **Items 18c, 18d** — OSF registration ID.

Everything else is answered or carries a written reason.
