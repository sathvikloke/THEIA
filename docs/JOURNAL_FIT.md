# Journal fit — audit of *Radiology: Artificial Intelligence*

Written 2026-08-13. **Scope of the audit, stated honestly:** the journal's whole
PubMed record was searched (608 records for 2025 onward alone). Titles and
abstracts were read for ~75 papers sampled across five strata: radiogenomics and
mutation prediction, null/cautionary/methodological findings, lung and NSCLC,
public-dataset-only studies, and the 40 most recent research articles. Author
instructions, journal policies and the About page were read directly. **No claim
below rests on a paper that was not read.**

Sources: [author instructions](https://pubs.rsna.org/page/ai/author-instructions),
[Handling Your Paper](https://pubs.rsna.org/page/ai/author-instructions/your-paper),
[About](https://pubs.rsna.org/page/ai/about), and PubMed.

---

## 1. Article type — the original choice was wrong

The journal defines **Technical Developments** as *"a brief description and
results of new algorithms, equipment, or datasets"* (2000 words, 25 refs, 6
figures, 2 tables).

This manuscript's Introduction says, in its own words, *"we do not propose a
better classifier."* Submitting an evaluation study under a category reserved for
new algorithms is a contradiction visible on page one, and Original Research
manuscripts are pre-reviewed by Editorial Board members for *"novelty, priority,
methodology quality, and subject-matter appropriateness"* before peer review —
so the mismatch would be caught at exactly the stage that rejects without review.

**Resolved:** switched to **Original Research** — *"new knowledge based on
original research"*, 3000 words, 35 refs, 6 figures, 4 tables. The change is also
substantively better: the body was 1631 words with a 264-word Discussion, which
is too thin to defend a contrarian claim. It is now 2368 words with room to
spare, and the Discussion carries an explicit account of what the localisation
endpoint does and does not establish.

## 2. Scope — the genre exists at this journal

Rad:AI publishes cautionary, confounder and null-adjacent work regularly. Papers
read for this audit:

| paper | year | why it matters here |
|---|---|---|
| Maleki et al., *Generalizability of Machine Learning Models: Quantitative Evaluation of Three Methodological Pitfalls*, 5(1):e220028 | 2022 | analytic choice > effect size, in a different domain. **Now cited** |
| Horvat et al., *Radiomics Beyond the Hype: A Critical Evaluation Toward Oncologic Clinical Use*, 6(4):e230437 | 2024 | critical evaluation of exactly this literature. **Now cited** |
| *Hurdles to AI Deployment: Noise in Schemas and "Gold" Labels* | 2023 | label-quality critique |
| *Structural MRI-based CAD Models for Alzheimer Disease: Insights into Misclassifications and Diagnostic Limitations* | 2025 | failure-analysis-first framing, n=3258 |
| *Performance of Lung Cancer Prediction Models for Screening-detected, Incidental, and Biopsied Pulmonary Nodules* | 2025 | comparative evaluation, no new model |
| *Impact of Exposure Parameters on Deep Learning Models in Chest Radiography* | 2026 | shortcut learning, **public data only** (MIMIC-CXR, MIDRC, EmoryCXR) |
| *AI Triage of Normal Chest Radiographs: A Silent Trial and Failure Analysis* | 2026 | negative operational finding |
| *Visit Frequency as a Potential Confounding Effect in Longitudinal Radiomic Models* | 2026 | confounder-first framing |

**A null is not what gets rejected here.** Public-data-only is also not
disqualifying.

## 3. Scale — this is the real risk, and it is not fixable

Every radiogenomics paper found in the journal has external validation:

| paper | development | external |
|---|---|---|
| Pediatric low-grade glioma *BRAF* subtyping | 214 | 112 (Children's Brain Tumor Network) |
| *IDH* two-stage radiomics framework | three public datasets (TCIA 227, UCSF, +1) | across datasets |
| Glioblastoma survival, multimodal | two institutional cohorts | two public test sets |

Recent Original Research cohort sizes across the 2026 issues: 1683 · 1941 · 3258
· 4069 · 7346 · 24,315 · 30,000+ · 35,008 · 104,364 · 313,966.

**This study: 153 labelled patients, primary analysis set 97, 23 positives.**

The closest structural counter-precedent found is Jang et al., *Unsupervised Deep
Learning for Blood-Brain Barrier Leakage Detection in Diffuse Glioma Using
Dynamic Contrast-enhanced MRI*, 7(3):e240507 (2025), doi:10.1148/ryai.240507 —
n=274, single institution, no external test set. But it is an image-reconstruction
paper in which *IDH* classification is a downstream check, and its headline result
is strongly **positive** (AUC 0.87 vs 0.81, P=.02). It establishes that a
single-institution study of this size *can* appear in the journal. It does not
establish that a null of this size can. **This citation was previously described
in `REFERENCES.md` as an n=274 EGFR study with no external test; that description
was wrong and has been corrected.**

Honest read: **desk rejection at Editorial Board pre-review is the single most
likely outcome, driven by cohort size rather than by the null.** The manuscript
argues that the cohort size cannot be increased usefully — the entire open world
of *EGFR*-labelled CT gives a 95% CI half-width of ±0.154 — but that argument has
to survive a reader who is not obliged to accept it.

## 4. House-style compliance — ten failures, all now fixed

| # | requirement | was | now |
|---|---|---|---|
| 1 | Structured abstract: **Purpose / Materials and Methods / Results / Conclusion** only | had a `Background` heading (that is *Radiology* style, not Rad:AI — verified against ~40 published abstracts and the instructions) | fixed |
| 2 | Summary Statement, 1–2 bold sentences, no abbreviations | absent | added |
| 3 | Key Points, up to 3, with summary data | absent | added |
| 4 | Date range of the study, in abstract Methods | absent | stated as unrecoverable, with the reason |
| 5 | Patient demographics in abstract Methods (`mean age, X years ± SD; N male`) | absent | added; computed by `theia.analysis.cohort.demographics`, not typed |
| 6 | Statistical software with version and manufacturer | absent | added (Python 3.13.9, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, pandas 2.2.3, statsmodels 0.14.4, PyTorch 2.10.0) |
| 7 | *P* value threshold for significance | absent (P values reported, threshold never stated) | added, two-sided P < .05 |
| 8 | IRB approval / waiver / exempt determination | absent — "did not constitute human-subjects research" is the authors' reading, not a board's determination | boxed **[AUTHOR ACTION REQUIRED]** in Methods and on the title page |
| 9 | Data sharing statement on the full title page | in-body only | on `title_page.tex` |
| 10 | **Double-anonymized review** — no names, institutions, acknowledgments or identifying URLs in the main file | `https://github.com/sathvikloke/THEIA` unblinded the author | replaced with an anonymized-mirror placeholder; real URL goes in the cover letter only |

Within limits after the rewrite: abstract 246/250 · body 2368/3000 · 6 figures/6
· 3 tables/4 · 15 references/35 · no uncited bibliography entries.

## 5. Two content risks, both now addressed in the manuscript

**The primary endpoint is not a clinical metric.** Attention mass, pointing
accuracy and area-matched IoU are not diagnostic accuracy, and neither CLAIM nor
STARD covers them. The Discussion now says this explicitly and states the
narrower claim being made — that plausible localisation is not evidence of
predictive validity, a dissociation only observable when the prediction fails.

**The endpoint passes by one fold** (11/15 = 73.3% against 71.4%). The rewrite
adds **Table 3**, reporting all 15 external evaluations individually. The
distribution is bimodal, not marginal: the weakest success has pointing accuracy
0.757 and the strongest failure has exactly 0.000, with nothing in between. The
endpoint is therefore insensitive to where the threshold sits within a wide
interval, while remaining sensitive to the fold count — which is the honest way
to state it.

## 6. Journal order

1. **Radiology: Artificial Intelligence** — Original Research. Written to spec.
2. **European Radiology Experimental** — Springer, open access, explicitly
   welcomes methodological and negative results. Same editorial family as
   *European Radiology*, where the converging Rodríguez Sánchez study appeared.
3. **Medical Physics** or **Journal of Medical Imaging** (SPIE).
4. **MIDL** — conference alternative, PMLR proceedings, does not preclude a
   journal version.

## 7. Still blocked on the author

IRB determination · OSF registration ID · Zenodo DOI · author list, affiliations,
degrees, ORCIDs · funding and conflicts (ICMJE forms) · statistical review, which
the journal advises before analysis and which has not happened · anonymized
repository mirror for review.
