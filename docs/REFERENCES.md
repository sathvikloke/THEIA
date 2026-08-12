# References

Every entry below was retrieved from PubMed and its metadata checked, not carried
over from a secondary source. Where a claim in this repository rests on a paper,
the claim is stated here next to the citation so the two cannot drift apart.

CLAIM item 40 asks for prior reports of the same patient population. Items 1–4
are those; the rest are the comparators and methods this project measures itself
against.

---

## Prior reports of this patient population

The internal cohort is the Stanford / VA Palo Alto series distributed on TCIA as
**NSCLC-Radiogenomics**. It has been reported before, and those reports are
required citations rather than optional ones.

**1. Bakr S, Gevaert O, Echegaray S, et al. A radiogenomic dataset of non-small
cell lung cancer.** *Scientific Data* 2018;5:180202.
[DOI](https://doi.org/10.1038/sdata.2018.202) · PMID 30325352

The dataset descriptor. 211 subjects; CT and PET/CT, semantic annotations against
a controlled vocabulary, tumour segmentation maps, gene mutation analysis, gene
expression microarrays, RNA-seq, and clinical data with survival. This is the
provenance for every number in this repository, and the source of both
supervision tiers: the segmentation maps and the AIM semantic annotations.

**2. Gevaert O, Echegaray S, Khuong A, et al. Predictive radiogenomics modeling
of EGFR mutation status in lung cancer.** *Scientific Reports* 2017;7:41674.
[DOI](https://doi.org/10.1038/srep41674) · PMID 28139704

**The single most important comparator for this project.** 186 cases, 89 semantic
image features annotated by a thoracic radiologist, decision tree, test-set AUC
**0.89** for EGFR.

Three things in it that directly shape this work:

* The final tree uses **four** variables: emphysema, airway abnormality,
  percentage ground-glass component, and tumour margin type. Two are whole-lung
  reads and two are lesion reads — which is what motivated the field-of-view
  experiment in `theia.analysis.semantic_oracle`.
* The **direction** matters and is easy to get backwards: emphysema or airway
  abnormality predicts **wild-type**; ground-glass component indicates
  **mutation**. This repository's earlier shorthand of "emphysema is the top
  predictor of EGFR" was directionally sloppy and is corrected here.
* They report **no statistically significant model for KRAS**. THEIA's KRAS arm
  is at chance with 32 positives against EGFR's 40 — equal power, different
  outcome. An independent group reaching the same conclusion on the same cohort
  is corroboration, not coincidence.

They also note that regularised logistic regression failed to find significant
performance on this cohort, which is why `semantic_oracle` scores a pruned
decision tree alongside logistic regression rather than assuming a model family.

**3. Gevaert O, Xu J, Hoang CD, et al. Non-small cell lung cancer: identifying
prognostic imaging biomarkers by leveraging public gene expression microarray
data — methods and preliminary results.** *Radiology* 2012;264(2):387–96.
[DOI](https://doi.org/10.1148/radiol.12111607) · PMID 22723499

The earlier radiogenomics work on this cohort (n=26 at that stage), establishing
the image-feature-to-metagene approach the collection was assembled to support.

**4. Nair VS, Gevaert O, Davidzon G, et al. Prognostic PET 18F-FDG uptake imaging
features are associated with major oncogenomic alterations in patients with
resected non-small cell lung cancer.** *Cancer Research* 2012;72(15):3725–34.
[DOI](https://doi.org/10.1158/0008-5472.CAN-11-3943) · PMID 22710433

PET arm of the same programme, n=25. Not used here — THEIA is CT-only — but it is
a prior report of overlapping patients and is cited for that reason.

---

## The external replication that reframes the project

**5. Rodríguez Sánchez DI, Middelkoop J, Vanneste T, et al. Quality over quantity:
biopsy-anchored CT radiogenomics models outperform all-lesion training in a
multi-tumour cohort despite a smaller sample size.** *European Radiology* 2026.
[DOI](https://doi.org/10.1007/s00330-026-12601-9) · PMID 42142113

Netherlands Cancer Institute. **1,646 patients, 11,473 segmented lesions**,
contrast-enhanced CT, EGFR status from next-generation sequencing. Internal
validation AUC **0.62–0.68**; external validation **0.55–0.63**.

Their external cohort is NSCLC-Radiogenomics — this project's *internal* cohort.
Ten times the patients and seventy-five times the lesions reproduce THEIA's
0.617. That is why [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) demotes incremental value
to a pre-declared null rather than treating it as an open question.

Their positive finding — that biopsy-anchored labels beat all-lesion labels
despite an order of magnitude less training data — is the same lever as this
project's variant harmonisation in `theia.data.variants`: label fidelity over
sample size.

---

## Comparators: EGFR from CT

Listed with n and the reported AUC, because the pattern across them is the
finding. The high numbers come from small single-centre cohorts; the one
large-cohort study with external validation (5, above) does not reproduce them.

| # | study | n | reported AUC |
|---|---|---|---|
| 6 | Huang L, et al. *Acad Radiol* 2025;32(8):4880–92. [DOI](https://doi.org/10.1016/j.acra.2025.04.029) | 826, 2 hospitals | external **0.889** (soft-voting fusion) |
| 7 | Shang Y, et al. *Radiol Med* 2023;128(12):1483–96. [DOI](https://doi.org/10.1007/s11547-023-01722-6) | 779 | external 0.701 |
| 8 | Lai R, et al. *Front Med* 2026;13:1868229. [DOI](https://doi.org/10.3389/fmed.2026.1868229) | 724, PET/CT | 0.862 |
| 9 | Yamazaki M, et al. *Br J Radiol* 2022;95(1140):20220374. [DOI](https://doi.org/10.1259/bjr.20220374) | 478 | 0.774 all-histology, **0.687 adenocarcinoma-only** |
| 10 | Wu J, et al. *Sci Rep* 2024;14:15877. [DOI](https://doi.org/10.1038/s41598-024-66751-1) | 438, 4 centres | external 0.809 |
| 11 | Le MN, et al. *Cancer Control* 2026;33:10732748261455532. [DOI](https://doi.org/10.1177/10732748261455532) | 200 | 0.87 (1 mm peritumoral) |

Note on **9**: its all-histology 0.774 falls to 0.687 when restricted to
adenocarcinoma. That is the same effect this project measures internally
(0.627 → 0.572 on segmented adenocarcinoma), and it is why comparisons in the
manuscript must be made against adenocarcinoma-only and external numbers rather
than against headline figures.

Studies 6–11 all find the peritumoral region informative, with optimal ring
widths reported at 1, 2, 3, 4, 6 and 15 mm. THEIA's independent peritumoral
result is therefore a **replication**, not a novel finding, and is described as
such.

---

## Methods

**12. Hanley JA, McNeil BJ. The meaning and use of the area under a receiver
operating characteristic (ROC) curve.** *Radiology* 1982;143(1):29–36.

The closed-form AUC standard error implemented in
`theia.analysis.power.hanley_mcneil_se`, used for the precision bound that
justifies the pre-specified "external testing is uninformative at the available
scale" finding. Validated against a bootstrap in `tests/test_power_precision.py`
rather than trusted.

**13. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required
for developing a clinical prediction model.** *BMJ* 2020;368:m441.

Minimum sample size, applied in `power.riley_min_n` only where its inputs are
defined — the 5-variable clinical model (needs 46, has 153) and the 512-feature
frozen probe (needs 4,707, has 153). Deliberately not quoted for the deep arm.

**14. Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in
Medical Imaging (CLAIM): a guide for authors and reviewers.** *Radiol Artif
Intell* 2020;2(2):e200029.

The reporting standard [CLAIM_CHECKLIST.md](CLAIM_CHECKLIST.md) is completed
against.

---

## Not independently verified

Recorded separately so it is clear which claims rest on a source this repository
has checked and which do not.

* **Jang et al., *Radiol Artif Intell* 2025;7(3):e240507** — cited in
  [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) §4.1 as precedent that this journal has
  accepted a single-institution, n=274 mutation-status study with no external
  test set. Surfaced during a literature audit; the DOI has **not** been
  independently confirmed against PubMed in this repository. Confirm before
  citing it in a manuscript.
* **Gevaert 2017's 0.89 as directly comparable to THEIA's numbers.** It is not,
  without qualification: that figure comes from 186 patients scored over 100
  random 70/30 splits, against this project's nested 5-fold CV on 153. Protocol
  differences of this size are worth 0.066 AUC on identical features here, so the
  two numbers are related but not interchangeable.
