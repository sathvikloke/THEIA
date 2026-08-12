# Outreach drafts

Drafts only. Nothing here has been sent, and nothing here can be sent by an
automated agent — every item needs the author to create the account, accept the
terms, or press send.

---

## 1. Netherlands Cancer Institute — collaboration enquiry

**Why this one first.** It is the only identified route to a cohort in the
thousands with CT and NGS-confirmed EGFR status, and it has the longest lead
time of anything on the critical path. It should go out before the science is
finished, not after.

**Recipients.** Corresponding author Regina Beets-Tan
(`r.beetstan@maastrichtuniversity.nl`), copying first author Diana Ivonne
Rodríguez Sánchez and Zuhir Bodalal (Dept. of Radiology, NKI, Amsterdam).

**The paper.** *Quality over quantity: biopsy-anchored CT radiogenomics models
outperform all-lesion training in a multi-tumour cohort despite a smaller sample
size.* European Radiology 2026. doi 10.1007/s00330-026-12601-9. PMID 42142113.
1,646 patients, 11,473 segmented lesions, internal AUC 0.62–0.68, external
0.55–0.63 — **their external cohort is NSCLC-Radiogenomics, which is ours.**

**Framing, and this matters.** Do not write as though they outperformed us. They
did not: 0.62–0.68 against our 0.617, from ten times the patients. The offer is a
convergence, not a request for rescue.

---

> **Subject:** Independent replication of your EGFR ceiling at n=153 — possible joint corrective?
>
> Dear Prof. Beets-Tan,
>
> I read *Quality over quantity* (Eur Radiol 2026) with more than the usual
> interest, because NSCLC-Radiogenomics — your external cohort — is the cohort my
> own work is built on.
>
> Working independently and without knowledge of your results, I get a pooled
> out-of-fold AUC of 0.617 ± 0.024 for EGFR from CT on those 153 patients, under
> nested cross-validation with within-fold rank pooling. Your 1,646 patients and
> 11,473 lesions give 0.62–0.68 internally and 0.55–0.63 externally. Two
> independent pipelines, an order of magnitude apart in sample size, landing in
> the same interval is a much stronger statement about the ceiling of
> lesion-anchored EGFR radiogenomics than either result is alone — particularly
> against a recent literature reporting 0.87–0.89 from single-centre cohorts of
> 200–800.
>
> Three things I have that might complement your cohort:
>
> 1. **A power analysis that explains the discrepancy.** A binormal simulation on
>    the observed prevalence shows that at an imaging AUC of 0.65, detecting
>    incremental value over a clinical model needs n > 1200 for 80% power, and
>    that at n = 150 the power is 0.15. The published 0.87s are not testable at
>    the cohort sizes they are reported from. The simulation reproduces our own
>    observed null as a consistency check.
>
> 2. **A localisation result that survives where discrimination does not, and
>    that is already externally validated.** Attention supervised against tumour
>    masks, evaluated on 420 held-out NSCLC-Radiomics patients (Maastro, so
>    different scanners and a radiotherapy-planning population): attention-mass
>    lift +0.297, beating a per-fold spatial shuffle in 25 of 30 evaluations.
>    Against controls — a centre prior scores 0.476 on the pointing game and a
>    randomly-initialised head returns exactly 0.000 lift, while the trained
>    heads reach 0.688. So the model localises the lesion reliably across
>    institutions even though it cannot genotype it, which is a cleaner
>    separation of those two claims than I have seen reported.
>
> 3. **A measurement of how much analytic choice is worth here.** On identical
>    features, our radiomics arm moves 0.136 AUC across six defensible
>    preprocessing choices, and 0.066 between flat and nested cross-validation.
>    Both exceed the effect sizes the field reports as findings.
>
> What I do not have is a cohort of your scale, and I do not think this
> literature gets corrected by another 150-patient study.
>
> Would you be open to a short conversation about whether a joint,
> multi-dataset methodological paper is of interest? I am thinking of the form
> your journal's readership responds to: two independent cohorts, a shared
> pre-specified protocol, and an explicit account of why the reported effect
> sizes do not survive it. I am happy to work entirely within your data
> governance — no imaging would need to leave your environment for the analyses
> that matter.
>
> Code and the full analysis plan are public at <REPO URL>, including the
> pre-registration and every negative result.
>
> With thanks for the paper, and for reading this far,
>
> <NAME>
> <AFFILIATION / independent researcher>
> <EMAIL>

---

**Before sending, fill in:** repository URL, name, affiliation (stating
"independent researcher" is fine and better than vagueness), email.

Every number in this draft is measured and current as of commit `31b4c1b`:
0.617 ± 0.024 from `results/CANONICAL.json`, the external grounding figures from
`results/external_grounding.json` (Gate D passed on all four pre-specified
criteria), and the analytic-choice spreads from `results/radiomics_sensitivity.json`.

**Consider attaching `figures/fig6_external_grounding.png`.** It is one panel
showing the external result against all three controls, and it makes the second
point land without the recipient having to take it on trust.

---

## 2. OSF pre-registration

**Type:** Open-Ended Registration, registering `docs/ANALYSIS_PLAN.md` at the
commit that seals it. Not a clinical-trial registry — CLAIM item 34 (ICMJE) is
not applicable to a retrospective secondary analysis of a public de-identified
imaging archive.

**Title.** Grounded radiogenomics on NSCLC-Radiogenomics: pre-specified analysis
plan for external validation of attention localisation and of the incremental
value of imaging over clinical variables.

**Description.**

> This registration fixes the analysis plan for a retrospective study predicting
> EGFR mutation status from lung CT on the public NSCLC-Radiogenomics collection
> (TCIA), and for its planned external evaluation.
>
> The primary hypothesis concerns **localisation**, not discrimination: that
> attention supervised against tumour masks generalises to an unseen cohort,
> measured as attention-mass lift over a per-fold spatial shuffle, with a centre
> prior and a randomly-initialised head as additional controls.
>
> A co-primary endpoint — whether imaging adds to a five-variable clinical model
> — is registered as a **pre-declared null**. The project's own power analysis
> shows 0.15 power at the observed effect size and cohort size, and an
> independent 1,646-patient cohort (Rodríguez Sánchez et al., Eur Radiol 2026)
> reproduces the same ceiling. Registering it as an expected null in advance is
> intended to prevent the result being reported later as an incidental
> disappointment or reframed as needing more data.
>
> The primary analysis set is restricted to adenocarcinoma, because every
> EGFR-mutant patient in this collection is an adenocarcinoma and the squamous
> and NSCLC-NOS patients are wild-type without exception; including them makes
> 13% of the cohort correctly classifiable on histology alone.
>
> Deviations from this plan, including two decided before any external cohort was
> obtained, are recorded in §7 of the registered document rather than silently
> applied.

**Attach:** `docs/ANALYSIS_PLAN.md`, `docs/MODEL_CARD.md`, `docs/RESULTS.md`.

**Steps (author only).**

1. Create an account at osf.io.
2. New Project → upload the three documents → Registrations → Open-Ended
   Registration.
3. Copy the registration GUID into `ANALYSIS_PLAN.md` §0 and commit.

---

## 3. Zenodo release

1. Sign in to zenodo.org with GitHub and authorise the integration.
2. Enable the toggle for the THEIA repository.
3. Cut a GitHub release (suggested tag `v1.0-preregistration`) — Zenodo mints the
   DOI automatically from it.
4. Paste the DOI into `ANALYSIS_PLAN.md` §0 and `CITATION.cff`, and commit.

Release notes draft:

> Sealed analysis plan and code state at pre-registration. Contains the full
> nested cross-validation pipeline, the grounding evaluation with its three
> controls, the radiomics and clinical comparators, the power analysis, and every
> archived run result including the negative ones. No external cohort had been
> obtained at this commit.
