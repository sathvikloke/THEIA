# RSNA / *Radiology: Artificial Intelligence* house style

Reference sheet the manuscript in `paper/` is formatted against.
`tests/test_manuscript_compliance.py` enforces the mechanical subset.

**Provenance.** Compiled 2026-08-13 by five independent investigations —
manuscript structure and cover letter, references, statistics and numbers,
figures and tables, language and terminology — each required to evidence every
rule from the author instructions or from published articles, then merged with
conflicts surfaced rather than silently resolved. Confidence tiers are carried
through:

| tier | meaning |
|---|---|
| **[V]** | verbatim from RSNA/RYAI author-facing documents |
| **[O]** | observed in typeset published articles |
| **[I]** | inferred — retained only where nothing better exists, and labelled |

**Read the OPEN QUESTIONS section at the end before relying on any single rule.**
`pubs.rsna.org` returns a login shell to automated fetchers, so much of the
verbatim text was recovered from Wayback snapshots (RYAI instructions last
archived June 2025). Changes since then cannot be ruled out. Two deliberate
deviations by this manuscript are recorded in `docs/JOURNAL_FIT.md`.

---

# Radiology: Artificial Intelligence (RSNA) — Consolidated House Style Spec

**Purpose:** single actionable reference for reformatting a LaTeX manuscript (Technical Development, 2000 words) to RYAI house style.

**Confidence tiers used throughout:**
- **[V]** verbatim-from-instructions — quoted from RSNA/RYAI author-facing documents
- **[O]** observed-in-published-papers — reverse-engineered from typeset RYAI articles
- **[I]** inferred — no direct source; retained only where it is the only information on a point that matters

---

## 1. MANUSCRIPT STRUCTURE

### 1.1 File layout

- **[V]** Submit the main manuscript as ONE document in exactly this order: (1) Abbreviated Title Page (Anonymized), (2) Abstract, (3) Main Body, (4) Acknowledgments (Anonymized, if any), (5) References, (6) Figure Legends, (7) Tables (embedded, one per page). Note Acknowledgments comes **before** References, and Tables are **last**.
- **[V]** Upload these six items as SEPARATE files, never inside the manuscript: Cover Letter; Full Title Page; point-by-point response letter (revisions only); Checklist (Original Research and Technical Development only); Figures; Supplemental Materials.
- **[V]** Main text file format: Word .docx (Office 2010+) **or** Adobe PDF. RYAI accepts PDF; the flagship *Radiology* journal does not — do not carry that stricter rule over.
- **[V]** LaTeX (.tex) is accepted for the **initial** submission only. Convert to Word at revision; .tex cannot be used for production.

### 1.2 Page setup

- **[V]** Double space all text.
- **[V]** Do not right-justify. (In LaTeX: `\raggedright`.)
- **[V]** Use Arial, Calibri, or Times New Roman at 11 pt.
- **[V]** Set special/mathematical characters and Greek letters in the symbol font; embed equations where cited in text.
- **[V]** **Do NOT number pages.** Page numbers are prohibited — ScholarOne adds them during PDF conversion. (In LaTeX: `\pagestyle{empty}`.)
- **[V]** Margins: the only requirement is "adequate margins." **No numeric margin value exists anywhere in the instructions.** Do not invent a 1-inch/2.5-cm rule.
- **[I]** Continuous line numbering is neither required nor forbidden — zero occurrences of "line number(s)" across the RYAI instructions, *Radiology* instructions, the *Radiology* checklist, and RSNA Editorial Policies. **Labeled inferred; retained because a LaTeX reformat must decide whether to load `lineno`.** Safe default: omit line numbers (nothing mandates them), or include them (nothing forbids them) — this is author's choice.

### 1.3 Section headings

- **[V]** Technical Development (2000 words) and Original Research (3000 words) use exactly four Main Body headings: **Introduction, Materials and Methods, Results, Discussion.** Subheadings may be added under Materials and Methods and Results.
- **[O]** Third-level subheadings are run-in, italic, and end with a period plus em dash: *External validation.—*, *Radiomics quality score.—*. Copyediting imposes this; write it that way.

### 1.4 Abbreviated Title Page (first page of main file, anonymized)

- **[V]** Exactly four items: (1) title of the manuscript; (2) Article Type; (3) Summary Statement of ≤255 characters in **boldface** — one sentence highlighting the key finding; (4) Key Points — up to 3 main points/conclusions including summary data, not repeating the Summary Statement, avoiding vague language and abbreviations (obvious ones like CT and MRI are fine); type "N/A" if none.
- **[V]** The checklist additionally names full title, article type, **five keywords**, and the abbreviation list (≤10) on this page.

### 1.5 Full Title Page (separate file)

- **[V]** Exactly nine fields: (1) title; (2) first and last names, middle initials, academic degrees, institutions of all authors; (3) name and street address of the institution where the work originated; (4) telephone, e-mail, and complete postal address of the corresponding author; (5) funding information; (6) Manuscript Type; (7) Word Count for Text; (8) unanonymized acknowledgments; (9) data sharing statement.
- **[V]** The data sharing statement must state: whether individual deidentified participant data will be shared; what data specifically; whether additional related documents will be available; when data become available and for how long; and by what access criteria. Identify proprietary data as proprietary and describe use/reuse restrictions.
- **[V]** Co-first / co-senior authorship must appear on the full title page **and** be requested and justified in the cover letter.
- **Not confirmed for RYAI:** superscript-numbered affiliations and a 15-word title limit are verbatim on the flagship *Radiology* page but absent from the archived RYAI text. Treat as *Radiology*-only.

### 1.6 Anonymization (double-anonymized review)

- **[V]** Anonymize the Abbreviated Title Page and Acknowledgments.
- **[V]** Do not mention the authors' institution or funding anywhere in the manuscript.
- **[V]** Anonymize author initials if they are readers.
- **[V]** Avoid self-references: no "In our prior study… (ref XX)", no "as we have previously described."

### 1.7 Supplemental material

- **[V]** Put all supplemental text, tables, and figures (with captions) together in ONE Word document uploaded as "Supplemental File for Review."
- **[V]** Upload movies/audio separately, labeled "Multimedia": MPEG/.mov/.avi/.wmv accepted, mp4 preferred, audio as mp3, each movie needs a legend, keep files under 30 MB.
- **[V]** Supplemental Materials are not copyedited.

---

## 2. COVER LETTER (separate file)

- **[V]** Six specified items; four flagged REQUIRED by the journal:
  1. Title of the manuscript.
  2. Complete author list.
  3. For manuscripts with two first authors, a brief statement of why equal contribution is warranted (**REQUIRED**).
  4. Description of any subject overlap with previously published works (**REQUIRED**).
  5. Description of any conflict of interest or industry support (**REQUIRED**).
  6. Confirmation of sole submission to *Radiology: Artificial Intelligence* (**REQUIRED**).
  Plus: explanation of circumstances if requesting fast-track and/or dual first authorship.
- **[V]** **Subject overlap statement** must cover prior published studies *and* work under review or in press elsewhere; explain what is new and different in the current analysis; upload PDFs of the overlapping articles during submission; and **also report the overlap in the Materials and Methods section**.
- **[V]** **Sole submission** means attesting that the same or similar material has not already been published by the authors and has not been and will not be submitted to another journal by them or by colleagues at their institution before it appears in RYAI. Similar material sent to advertising or news media must be indicated at submission with a copy provided.
- **[V]** (RSNA-wide Editorial Policies, applies to the whole *Radiology* suite) Describe any use of AI or AI-assisted technologies (LLMs, chatbots, image creators) in the study or in manuscript preparation/editing **in both the cover letter and the manuscript**, giving tool name, version, date of access, and manufacturer.
- **[V]** (RSNA-wide Editorial Policies) State which author(s) had control of the data and performed the analyses.
- **[V]** **Do not put suggested/opposed reviewers in the cover letter.** They are entered optionally in ScholarOne Step 5 ("Reviewers and Editors") via the Add Reviewer button.
- **[V]** **Do not write a "why this paper fits the journal" pitch.** The only suitability argument the instructions ask for is the separate fast-track letter, which must also be copied directly to radiology-ai@rsna.org.
- **[V]** Prior presentation / preprint disclosure is **mandatory at submission**, but the instructions do **not** name the cover letter as the required location. Disclose where and when earlier versions were presented, provide access to them, and — if a substantial portion was previously published in print or on the web — describe in detail how the present work differs. Disclose preprint DOI and licensing terms. Posting to arXiv does not count as prior publication.
- **[V]** Mechanics: paste the letter into the ScholarOne "Cover Letter" box at Step 6, or attach it as a file. Step 6 also requires pasting the Summary Statement into its own box and stating the number of tables and figures uploaded.

---

## 3. REFERENCES

### 3.1 In-text citation

- **[O]** Cite as arabic numerals in **round parentheses** — not superscript, not square brackets: `…a lack of clinically viable radiologic models (1).` (1,771 cross-references across 42 typeset articles; zero `<sup>`.)
- **[O]** Multiple citations: commas with **no space** — `(4,5)`, `(1,3,7)`. Three or more consecutive: en dash — `(2–6)`. Combined: `(2,4–6)`.
- **[V]** Number references in order of first mention. Never alphabetize.

### 3.2 Author names

- **[O]** List **all authors up to six**. With seven or more, list the first **three** followed by "et al." Never 4, 5, or 6 names + et al. (627 references with et al. after exactly 3; full lists peaked at exactly 6.)
- **[V]** The journal's own worked example lists exactly six authors with no et al., corroborating the ceiling.
- **[O]** Format: surname + space + initials, no periods or spaces inside initials, separated by `", "`, closed with a period. No `&`, no comma between surname and initials. → `DeGrave AJ, Janizek JD, Lee SI.`
- **[O]** Collective/group author: group name, **semicolon**, then named individuals, then et al. → `National Lung Screening Trial Research Team; Aberle DR, Adams AM, et al.`

### 3.3 Journal-article format

- **[O]** Template with exact punctuation:
  `Authors. Article title. Jrnl Abbrev Year;Volume(Issue):FirstPage–LastPage.`
  No comma between abbreviation and year; space before year; semicolon after year; issue in parentheses; colon before pages; terminal period.
- **[V]** Abbreviate periodical titles per NLM style, with **no periods**: `Nat Mach Intell`, `Radiol Artif Intell`, `N Engl J Med`.
- **[O]** **No italics and no bold anywhere in a reference** — journal name, volume, issue, pages all plain roman. (1,391 RSNA-deposited references contained 2 `<italic>` elements total, both inside article titles.)
- **[O]** Include the issue in parentheses when it exists; omit the parenthetical entirely when it does not. Never print empty parentheses.
- **[O]** E-locator articles: put the e-locator where the page range would go, no en dash → `Radiol Artif Intell 2021;3(4):e210011.`
- **[V]** Give complete, non-elided page numbers (1537-1544, not 1537-44). **[O]** In the typeset article the separator is an **en dash** (777 en dash vs 25 hyphen).
- **[O]** Reproduce the cited article's title capitalization as published — RYAI does not force sentence case or title case.
- **[V]** Bibliographic accuracy is the author's responsibility; reference hyperlinks are auto-generated and will not function if the data do not match.

### 3.4 DOIs — CONFLICT between lenses (resolved)

The `references` lens surfaced a genuine conflict inside its own evidence, and it must be stated:

- **[V]** The Instructions **require** the DOI: "It is important to include the article's DOI in the reference"; worked example ends `…2092-2093. doi: 10.1056/NEJMc070741`.
- **[O]** Typeset RYAI reference lists carry **no** DOIs for ordinary journal articles with volume and page data. ryai.240003 (20 refs), ryai.240050 (19 refs), ryai.250123 (33 refs) contain zero publisher-supplied DOIs; every DOI in their PMC XML is `assigning-authority="pmc"`.

**Resolution — the [V] instruction wins for what the author does; the [O] observation only predicts the typeset output.** These are not actually contradictory instructions: supply `doi: 10.xxxx/yyyy` at the end of every journal reference at submission (the instructions demand it and it drives hyperlink generation), and expect copyediting to strip it. Do not omit DOIs on the strength of the published-output evidence.

### 3.5 Non-journal reference types

- **[V]** **Web content:** `Author(s) (if any). Title of the page or content. Name or owner of the Web site. URL. Published <date>. Updated <date>. Accessed <date>.` Accessed date is mandatory; Published/Updated only if known. URL followed by a period, not bracketed, no "Available from:".
- **[V]** **Book chapter:** `Chapter authors. Chapter title. In: Editor(s), ed(s). Book title. Edition. City, State: Publisher, Year; pages.`
- **[O]** **Conference proceedings:** `Authors. Paper title. In: <Full proceedings name>. Publisher, Year; pages.` → `In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017; 3462–3471.`
- **[O]** **Preprint (preferred, current 2025 form):** `Authors. Title. arXiv YYYY. Preprint posted online Month D, YYYY; doi:10.48550/arXiv.NNNN.NNNNN.` The preprint DOI **is** retained in print.
- **[O]** **Preprint (older variant, avoid):** `Authors. Title. ArXiv NNNN.NNNNN [preprint]. Posted Month D, YYYY. Accessed Month D, YYYY.`
- **[O]** **Datasets:** no dedicated type exists. Cite either as the data-descriptor journal article (preferred where one exists — e.g. `Johnson AEW, Pollard TJ, Berkowitz SJ, et al. MIMIC-CXR… Sci Data 2019;6(1):317.`) or as web content naming the hosting platform as site owner (`… Kaggle. https://… Published 2019. Accessed September 2019.`).
- **[O]** **Software / code repositories:** cite as web content — title, owner/organization ("GitHub", "HicServices"), URL, Published/Accessed dates.

### 3.6 Reference limits

- **[V]** Technical Development: **25**. Original Research: 35. Data Resources 25; Review Articles 50; Special Reports 100; AI in Brief 20; Editorial Opinions 35; Invited Commentary 5–10; Letters to the Editor 5.
- **[O]** The Original Research cap of 35 is effectively advisory — 14 of 30 sampled Original Research articles exceed it (max 55). **But the two sampled Technical Development articles had 25 and 24 references, i.e. at or under the cap.** For a Technical Development manuscript, treat 25 as a hard ceiling.

---

## 4. STATISTICS AND NUMBERS

### 4.1 P values

- **[V]** Set as an uppercase **italic** *P*. Never "p value", "p-value", or "P-value". In prose the noun phrase is "*P* value" — italic *P*, roman "value", no hyphen.
- **[V]** **Drop the leading zero** on *P* and on any quantity that cannot exceed 1 (*P*, *r*): write `P = .03`, never `P = 0.03`; `P < .001`, never `P < 0.001`. (169/169 published P values had no leading zero.)
- **[O]** `P < .001` is the floor. Report exact values down to .001; anything smaller is `P < .001` — never `P < .0001`, `P = .0003`, or `P = 0`.
- **[O]** Two decimal places when *P* ≥ .01; three when *P* < .01. Keep three decimals if rounding to two would push a significant value across the threshold (retain `P = .046`, not `P = .05`).
- **[V]** A *P* value can never equal 1. The largest reportable value is `P > .99`.
- **[V]** Never report a bare *P*: give the regression coefficient or the group means alongside it.
- **[V]** Report results for **every** variable collected and analyzed, not only those reaching significance.

### 4.2 Confidence intervals

- **[O]** Write `95% CI:` then the two bounds separated by a **comma**: `95% CI: 0.61, 0.72`. Never a dash, hyphen, or "to". (131 of 146 occurrences used the colon form; zero used a dash between bounds.) Negative bounds keep the comma: `95% CI: -2.0, 2.0`.
- **[O]** This is imposed at copyediting — an author manuscript's `(95%CI, 0.88–0.92)` became `(95% CI: 0.88, 0.92)`. Submit in house style rather than relying on the copyeditor.
- **[O]** Nesting: parentheses outermost, square brackets for the inner CI → `(AUC, 0.86 [95% CI: 0.79, 0.92] vs 0.72 [95% CI: 0.62, 0.79]; P < .001)`.
- **[O]** Inside a table, strip the repeated `95% CI:` label; give bounds bare in parentheses, still comma-separated, and declare the meaning once in the Note.

### 4.3 Performance metrics

- **[O]** Report AUC to exactly **2 decimal places** (36/36 in-text point estimates), and pair the primary AUC with a 95% CI.
- **[O]** Comparing two AUCs: both point estimates with CIs, joined by `vs` (**no period** — 91 uses of "vs" vs 1 of "vs."), then the *P* value after a semicolon.
- **[V]** Abbreviate area under the ROC curve as **AUC, never AUROC**.

### 4.4 Dispersion, ranges, and demographics — CONFLICT flagged

- **[O]** Mean ± SD: `value unit ± SD`, unit attached to the mean, SD bare → `mean age, 61 years ± 9 [SD]`. Tag the dispersion measure as SD **at first use only** — square brackets when already inside parentheses, round parentheses in open prose — and omit the tag thereafter.
- **[O]** **Two orderings coexist in current issues** and the `statistics` lens found no rule discriminating them: `mean age, 61 years ± 9 [SD]` and label-first `mean age ± SD, 61.3 years ± 5.0`. Pick one and use it consistently throughout the manuscript.
- **[O]** IQR: literal label `IQR,` then the two quartiles joined by an **en dash** — not a comma (note this differs from CIs). Two accepted delimiters: `median age, 43 years; IQR, 31-55 years` or `median age, 74 years [IQR, 68-79]`.
- **[O]** Plain ranges: label with the word `range,` and join endpoints with an en dash → `age range, 11–64 years`, `(range, 0.77–0.93)`. Never present a bare dashed pair.
- **[V]/[O]** Open Results with a demographic sentence giving N, sex counts, and either mean age ± SD or median age with IQR. **[V]** "The first paragraph should summarize the demographics of your study population."
- **[V]** Give age (mean and range) separately for male and female groups where the design makes that relevant.

### 4.5 Percentages and counts

- **[V]** Every percentage must be accompanied by its numerator and denominator, in Results or a table — specifically for sensitivity, specificity, accuracy, PPV, and NPV.
- **[O]** Spell the fraction with the word **"of"**, never a slash: `44 of 823`, not `44/823`. (Zero slash forms in the corpus.)
- **[O]** Both orderings are acceptable and both appear in current issues: count-first `69 of 124 (56%)` and percentage-first `5.7% (17 of 297)`. The invariant is that numerator and denominator are present, not which comes first.
- **[O]** Spell out numbers below 10 in running text, including inside "N of D" fractions: `six of 17 (35%)`, `seven of 577`. Spell out any number opening a sentence: "Twenty-one studies…".
- **[O]** Thousands separator: use a **thin space**, not a comma, and only for numbers of five digits or more. Four-digit numbers take no separator: `7454`, `1205` — but `10 000`, `24 315`, `81 936`. Copyediting enforces this (`1,012` → `1012`).

### 4.6 Statistical Analysis paragraph (last paragraph of Materials and Methods)

- **[V]** State the statistical methods used; the statistical software with **version and manufacturer**; the *P* value used for significance; and any multiple-comparisons correction.
- **[V]** A statement justifying sample size and/or a power calculation is **strongly recommended**.
- **[V]** Give the initials of the statistical analyst if they are an author.

### 4.7 Symbols and italics

- **[O]** Italicize single-letter statistical symbols the same way as *P*: *r*, *t*, *d*, *n*, *k*. Write `r = 0.74`, `unpaired t tests` (no hyphen), `Cohen d`, `n = 800`. Greek symbols stay roman: `Fleiss κ = 0.404`.
- **[V]** Comparison symbols (>, <, ≤, ≥) may appear **only inside parentheses**; spell them out in running text — `lesion size (>2 cm)` or `lesion size greater than 2 cm`. This applies to *P* too: "a *P* value less than .05 indicated a statistically significant difference."
- **[V]** Use SI units for radiation measurements and laboratory values.

### 4.8 Abstract Results

- **[V]** The structured abstract's Results must mirror the Methods and carry actual numbers, percentages, and indicators of statistical significance — not qualitative claims.

---

## 5. FIGURES

### 5.1 Files and formats

- **[V]** Photographic/halftone (clinical) images: **PSD, TIF, AI, or EPS** (unflattened vector). **PNG is not accepted.**
- **[V]** Photographic/halftone: 300 dpi, sized 3–7 inches.
- **[V]** Graphs/illustrations/line art: .psd, .tif, .ppt, .ai, or .eps at **1200 dpi**, layers retained.
- **[V]** Do not send photographic/clinical image files as Word or PowerPoint documents (pixel resolution is lost).
- **[V]** **New submission:** combine all figures into ONE separate text document in text-citation order, with each legend immediately following its figure. (Note this does not contradict the "do not send as Word" rule, which governs the image files themselves.)
- **[V]** **Revision:** upload production-quality figures individually, with the figure number in each filename (`Figure 1.tif`, `Figure 2a.tif`, `Figure 3.eps`).
- **[V]** Revision: submit each multipart figure part as its own file with its own legend, and paste each legend into the ScholarOne caption box, **beginning with the word "Figure"** plus number and part (`Figure 1`, `Figure 2a`).

### 5.2 Figure legends — CONFLICT between lenses (resolved)

The `structure` lens flagged an internal conflict in the instructions and the `figures` lens restated only one side of it:

- **[V]** The single-document contents list names "Figure Legends" as an element of the **main manuscript file**.
- **[V]** The Figure Legends section says: "For new submissions, figure legends should appear directly after the relevant figure in the combined figure text document. For revised manuscripts, all figure legends should appear collectively on one or more pages at the end of the main document."

**Resolution — the specific Figure Legends rule overrides the generic contents list for new submissions**, because it addresses the new-submission case by name while the contents list is a generic enumeration. **Safest compliant practice: at new submission, place legends after each figure in the combined figure document AND keep a Figure Legends section in the main file.** At revision, legends go collectively at the end of the main document *and* into the ScholarOne caption boxes.

### 5.3 Legend content (mandatory)

- **[V]** Define every abbreviation used in the figure or the legend **in the legend** — never by reference to the text.
- **[V]** For images of human subjects: state patient **age, sex, and clinical history/disease**.
- **[V]** State the type of image, its plane, whether contrast material was used, the MR pulse sequence information, and the features to be observed.
- **[V]** Describe every label placed on the illustration; label all features described in the legend; use a different label for each feature.
- **[V]** Photomicrographs: give the stain and original magnification.
- **[V]** Drawings and graphs: state the important points to be observed.
- **[V]** Do not duplicate text material.
- **[O]** House convention for a patient-image legend: noun-phrase opener naming modality/view, then "in a [N]-year-old woman/man", then the finding → *"Digital mammography (DM) and digital breast tomosynthesis (DBT) images in a 61-year-old woman."* Describe the patient by age and sex only — never by identifier or "case number."
- **[O]** Legend sentences take the image as grammatical subject in the present tense: "image shows", "images show", "Diagram shows", "Graphs show" — not "Figure 1 shows" or "We show".
- **[O]** Call out annotations parenthetically: `(arrows)`, `(yellow)`.
- **[O]** Expand abbreviations in a trailing run-on sentence at the very end of the legend, alphabetical, comma-separated, `ABBR = expansion`, expansion lowercase unless a proper noun → `BI-RADS = Breast Imaging Reporting and Data System, CAD = computer-aided detection, CC = craniocaudal.`

### 5.4 Numbering, panels, and citation in text

- **[V]** Number figures `figure 1a, figure 1b, figure 2`, with no more than four components (a, b, c, d) per figure.
- **[O]** In print, panels are **uppercase letters in parentheses** and the whole multipanel figure carries ONE legend walking through panels in order: `(A) … (B) … (C) …`.
- **[O]** In running text, cite panels as `(Fig 1A)` — abbreviated "Fig", no period, letter closed up to the number, inside parentheses. Spell out "Figure 1A" at sentence start.
- **Conflict, minor:** the instructions themselves are internally inconsistent on panel-letter case — the numbering rule uses lowercase ("figure 1a", "components (a, b, c, d)") while the caption-box rule uses uppercase ("Figure 2A, Figure 2B"). **Published articles are uniformly uppercase, so use uppercase (A), (B) in legends and text**; the [O] evidence is stronger here because it reflects what the journal actually prints.
- **Conflict, minor:** the four-component cap is not enforced in recent issues (ryai.240017 has panels (A)–(F)). Follow the [V] cap of four unless the science demands more.

### 5.5 Image preparation

- **[V]** Crop to the area of interest but leave enough surrounding anatomy for assessment.
- **[V]** Use the same magnification across multiple images of a given modality or view.
- **[V]** Place labels/arrows on a separate unflattened layer; they must touch the edge of the feature labeled; do not use equilateral triangles for arrowheads.
- **[V]** Exclude all patient/subject identifying information; preserve anonymity throughout.

### 5.6 Counts and permissions

- **[V]** Technical Developments: **6 figures / 2 tables**. Original Research 6/4; Data Resources 3/2; Reviews 6/4; Special Reports 6/2; AI in Brief 4 combined; Letters none. Overflow figures go to Supplemental Material.
- **[V]** Written permission from the publisher (and the author where applicable) is required to reproduce any previously **published** figure or table; the journal actively discourages such reuse.
- **[V]** The RYAI instructions contain **no** clause requiring separate permission to display images from a public archive such as TCIA. Governing obligations come from the archive's own licence.
- **[V]** (TCIA policy) TCIA images are distributed already de-identified under CC BY 3.0/4.0 for most collections, so figure redistribution is permitted with attribution; a small number of collections impose commercial-use restrictions.
- **[V]** (TCIA policy) Cite the specific collection's persistent **DOI data citation**, not the web-page URL, and acknowledge the dataset. **[O]** RYAI papers in practice also cite the general TCIA reference (Clark et al, *J Digit Imaging* 2013).
- **[V]** (TCIA policy) Do not attempt to re-identify or contact participants — relevant if a figure would show a 3D surface rendering of a head or face.
- **[V]** IRB approval, patient informed consent, and (for U.S. studies) HIPAA compliance must be addressed in the **first paragraph of Materials and Methods**. This is the mechanism covering images shown in figures, including from retrospective or public cohorts.

---

## 6. TABLES

- **[V]** Prepare tables in Word as **editable text, not images**. Attach them to the end of the text document, one table per page, each with a title, numbered in Arabic numerals. **Tables do not go inline in Results.**
- **[V]** Do not use Word's merged-cell feature; use sub-headings instead.
- **[V]** Define every abbreviation used in a table in that table's own footnotes.
- **[V]** Tables must be understandable by themselves, without reference to the Results section.
- **[V]** Either Results or the tables must give numerators and denominators for all percentages.
- **[O]** Title format: `Table N:` label, then a **headline/title-case descriptive noun phrase with no terminal period** → `Table 2: Zero-Shot Classification Performance of Single-Sequence Vision Models on Normal versus Abnormal Binary Classification Task`.
- **[O]** The general footnote is one block opening with **`Note.—`** (the word Note, a period, an em dash, no spaces). Its first sentence declares what the numbers are, then the abbreviation expansions follow → `Note.—Unless otherwise indicated, data are numbers of patients, and data in parentheses are percentages. PFS = progression-free survival, WHO = World Health Organization.`
- **[O]** Abbreviations inside the Note are comma-separated `ABBR = expansion` pairs, **alphabetical**, expansions lowercase unless proper nouns. A table whose only footnote content is abbreviations still uses the `Note.—` prefix.
- **[O]** Cell-specific footnotes are separate from the `Note.—` block and use symbol markers in the order `*`, then `†`, then `‡`, placed after the Note.
- **[O]** Move repeated units, the dispersion measure, and the `95% CI:` label out of the cells and into the Note.

---

## 7. LANGUAGE AND TERMINOLOGY

### 7.1 Style basis and spelling

- **[V]** Copyediting conforms the manuscript to RYAI style, "based on widely accepted conventions of grammar and usage, the *American Medical Association Manual of Style*, and *Stedman's Medical Dictionary*." This editing may be substantive.
- **[O]** Use **American English** throughout: -ize/-ization, not -ise/-isation. `localization`, `normalization`, `randomized`, `analyzed`, `characterize`; `tumor`, `hemorrhage`, `edema`, `gray`, `labeled`, `color`, `modeling`, `pediatric`, `behavior`, `fiber`. (38-paper corpus: zero British forms in every pair tested.) No RSNA document states this rule explicitly — it rests on the AMA/Stedman's basis plus corpus evidence.
- **[O]** Use `center`/`multicenter`. "Centre" appears only inside proper nouns (institution names).

### 7.2 Imaging terminology

- **[V]** Use **section**, not slice, for cross-sectional CT/MRI: `section thickness`, `section interval`, `section gap`, `axial section`. Reserve "slice" for histologic tissue. **[O]** Enforcement tightened after 2020 — the only two "slice thickness" hits in a 38-paper corpus are both from 2020; every 2021–2025 paper uses "section thickness."
- **[V]** Report CT/MRI acquisition geometry as section thickness and section interval in Materials and Methods.
- **[O]** "Section" is also correct for document divisions ("the Materials and Methods section") — no conflict.
- **[V]** Use **"at"** for the modality/examination and **"on"** only for images: `detected at MRI`, `cancer detection at mammography`, but `on MRI scans`, `on radiographs`.
- **[V]** **Radiography** is the modality, **radiograph** is the image — do not write "x-ray" for either. "X-ray tube" and "x-ray beam" are correct.
- **[V]** **US = ultrasonography** (the modality). Do not abbreviate "ultrasound" when it means the waves (e.g., focused ultrasound). Write "US images", "US examinations".
- **[V]** Write **"signal intensity"**, never bare "signal", when describing contrast on an image. Change CT "density" to **"attenuation"**.
- **Conflict, unresolved for RYAI:** the *Radiology* blog states "*Radiology* journal specifically uses MRI scans, not MR images." **This could not be confirmed for RYAI** — the corpus contains both forms in body prose (MR images 49, MRI scans 41). Treat as a *Radiology*-only rule; either form is defensible in RYAI, but be internally consistent.

### 7.3 Word choice

- **[V]** Reserve **"significant"/"significantly"** for statistical significance only; otherwise use substantial, important, marked, or consequential. Exception: the established term "clinically significant prostate cancer."
- **[O]** For a null result write **"there was no evidence of a difference"**, not "no significant difference" or "the groups were similar." (15 uses of the preferred form vs 2 of the discouraged one.)
- **[V]** Use **"correlated"** only in the statistical sense and give the coefficient; otherwise use associated or compared.
- **[V]** Do not write **"cases"** where a more specific noun exists — use patients, participants, lesions, examinations, or images.
- **[V]** **Prospective study → "participants"; retrospective study → "patients."** Title the first Results paragraph "Patient Characteristics" or "Participant Characteristics" accordingly.
- **[O]** **Never call people "subjects."** Reserve "subject" for the adjectival sense ("subject to interrater variability"). Zero person-denoting uses across 38 RYAI full texts.
- **[V]** Distinguish **study sample** (a subset of the population), **population** (the group you generalize to), and **study cohort** (only appropriate for a longitudinal study that samples and observes a cohort over time).
- **[V]** Diseases and tumors **manifest**; only patients **present** (to a clinic or emergency department).
- **[V]** Change **multivariate/univariate → multivariable/univariable** unless the model truly has multiple dependent variables.
- **[V]** Do not coin abbreviations for adjusted statistics (aOR, aHR). Write "In the adjusted model, the OR was XX" or "after multivariable adjustment, the OR was XX."

### 7.4 Voice and self-reference

- **[O]** **First person plural ("we", "our") is accepted and standard** in RYAI — do not force everything into the passive. ("we" 591 times across 38/38 papers; "our" 344 times in 37/38.) Use it in Introduction, Methods, Results, and Discussion.
- **[V]** The one place third person is mandatory: **citing your own prior work**, because review is double-anonymized. Write "Smith et al have demonstrated", not "we previously demonstrated."
- **[V]** Delete all self-evaluation: **"novel", "unique", "ground-breaking", "first"**, and any claim of priority.
- **[V]** Write for a general radiologist, not a subspecialist. Avoid laboratory slang and clinical jargon.

### 7.5 Abbreviations

- **[V]** Hard cap of **10 abbreviations** for the whole paper; list them on the Abbreviated Title Page.
- **[V]** Define every abbreviation **twice** — at first use in the abstract **and** again at first use in the main text. The abstract definition does not carry over. Example: "cerebrospinal fluid (CSF)."
- **[V]** **Do not define** the standard imaging modalities: CT, MRI, US, SPECT, PET. Never write "magnetic resonance imaging (MRI)."
- **[O]** **Do** spell out "artificial intelligence (AI)" and "deep learning (DL)" at first use — they are not on the exempt list. (19 of 20 corpus papers that use AI or DL three or more times expand at first body use.)
- **[V]** **No abbreviations at all** in the Summary Statement or in "Implication for Patient Care." Avoid them in Key Points except obvious ones (CT, MRI).
- **[V]** Write out abbreviations again in the Summary, Key Results, and the first and last paragraphs of the Discussion — readers jump there first.
- **[V]** Each abbreviation on the list should appear at least **10 times** between Introduction and Discussion; otherwise write it out and drop it from the list.
- **[V]** Define abbreviations independently in each table footnote and each figure legend; tables and figures must stand alone.
- **[V]** Format abbreviation lists in table footnotes and figure legends **alphabetically**, with an equals sign, comma-separated: `BMI = body mass index, OR = odds ratio` — not in order of appearance.

### 7.6 Nomenclature

- **[O]** **Italicize gene symbols** (*IDH*, *IDH1*, *IDH2*, *BRAF*), including in the article title and in table/figure abbreviation footnotes. Leave the spelled-out gene/enzyme name roman ("isocitrate dehydrogenase") and leave protein-level variant designations roman (V600E). Gene symbols inside cited reference titles stay roman. Not stated in any RSNA document; inherited from the AMA Manual of Style.

---

## OPEN QUESTIONS

Nobody across the five lenses could establish the following. Each is a real gap, not an oversight.

**Access and provenance (affects everything above marked [V])**
1. `pubs.rsna.org` sits behind a Cloudflare bot challenge that returns a login shell to automated fetchers. All verbatim instruction text was recovered from Wayback snapshots (RYAI Instructions 2025-06-10; RSNA Editorial Policies 2025-06-13; *Radiology* Instructions 2026-06-05; *Radiology* Checklist 2022-07-16) or through the r.jina.ai text proxy. One live WebFetch of the RYAI instructions succeeded and corroborated every formatting rule. **The RYAI page has not been re-archived since June 2025, so changes between June 2025 and August 2026 cannot be ruled out.**
2. The **RYAI Manuscript Preparation Checklist** (`/page/ai/author-instructions/checklist`) could not be retrieved in full by three of the five lenses — no Wayback snapshot, live page blocked. If a line-numbering or margin requirement exists anywhere, that checklist is the one remaining place it could hide. (The sister *Radiology* checklist, retrieved in full from a 2022 snapshot, contains neither.)
3. `/page/ai/author-instructions/your-paper` ("Handling Your Paper") could not be fully retrieved; it covers editorial workflow, not formatting.

**Structure and formatting**
4. **Exact numeric margin requirement.** None found — only "adequate margins." Whether the editorial office enforces a de facto 1-inch/2.5-cm margin is unknown.
5. **Whether line numbering is expected.** Zero mentions across four documents. Neither mandated nor forbidden.
6. Whether the RYAI **Full Title Page requires superscript-numbered affiliations** and whether a **title word limit** applies. Both are verbatim on the flagship *Radiology* page (superscripts reflecting authorship order; 15-word title) but absent from archived RYAI text. One web search attributed the superscript rule to RYAI but this may be conflation.
7. Whether **prior-presentation and preprint disclosure must go specifically in the cover letter.** The duty to disclose is unambiguous; the required location is never stated.
8. Whether RYAI requires an **"Implication for Patient Care" section** for all Original Research. The instructions are internally inconsistent: the abbreviation prohibition references it, but the Manuscript Preparation section lists only Introduction / Materials and Methods / Results / Discussion.
9. **Author-count ceiling.** The archived RYAI page and RSNA Policies say "more than 20 authors may require Editor consent"; a live fetch reported 40+. Does not affect formatting.

**References**
10. **Whether journal names are italicized in the final typeset PDF** could not be confirmed by visual inspection. The "no italics" rule rests on markup-level evidence (zero `<italic>` wrapping `<source>`/`<volume>`/`<fpage>` across 1,391 RSNA-deposited references). A PMC-rendered fetch claimed italics and superscript in-text citations but was self-contradictory ("superscript numbers (e.g., [1], [2]-[6])") and contradicted by raw XML, so it was rejected.
11. **The Instructions contain no stated author-truncation ("et al.") rule at all.** The six-author / first-three rule is derived entirely from published practice plus the journal's own six-author worked example.
12. **No format is given for preprints, datasets, or software/code.** Those are reverse-engineered from published lists only, and the preprint form is not stable across issues (two variants observed).
13. **Whether DOIs are stripped at copyediting or simply not supplied by authors** could not be determined. (See §3.4 for the practical resolution.)
14. Whether the **reference caps are enforced at desk screening or purely advisory**. Nearly half of sampled Original Research articles exceed 35; no statement either way was found.
15. Whether RYAI permits citing **personal communications, abstracts, "in press" works, or unpublished data**. WebFetch returned NOT MENTIONED for each.
16. Reference-list evidence is limited to the ~43 RYAI articles in the PMC open-access subset; four were excluded as non-house-style. Subscription-only articles could not be inspected.

**Statistics**
17. The **Radiology Scientific Style Guide** and flagship *Radiology* Instructions are login-gated. Secondary summaries attribute several rules to them that could **not** be verified against a primary source or published text: **odds/risk/hazard ratios and their CIs rounded to hundredths**, and **correlation-coefficient digits scaled to sample size**. Do not rely on these.
18. Whether a **95% CI is formally mandatory** (versus merely near-universal) for AUC, sensitivity, and specificity. Empirically primary AUCs almost always carry one, but 19 of 36 in-text AUC mentions had no adjacent CI.
19. Whether **count-first vs percentage-first ordering** is governed by a stated rule keyed to context, or is author preference. Both appear in current issues.
20. **Decimal places for sensitivity, specificity, and accuracy percentages.** Published papers show both whole numbers and one decimal with no retrievable rule.
21. Whether the **mean±SD label-first vs value-first ordering** is governed by any rule. Both appear in current issues.
22. Whether RYAI has a **journal-specific statistics addendum or reviewer statistical checklist**. None surfaced.

**Figures and tables**
23. Whether the RSNA-specific **letter of informed consent for identifiable images** exists and whether "masking the eyes alone is insufficient" — described in a web-search snippet but not verifiable at source. Verify before relying on it.
24. **Color mode (RGB vs CMYK)** — no requirement stated anywhere; only "Color is acceptable for charts and graphs."
25. **No word or character limit for figure legends** is stated, and the instructions do not say whether legends count toward the manuscript word count.
26. Whether the **four-components-per-figure cap** is waived at editorial discretion or is stale text — recent issues publish (A)–(F).
27. Whether editors expect an explicit **in-legend or in-Acknowledgments provenance statement for a displayed public-archive (TCIA) image**. Nothing in RYAI addresses public archives by name; observed practice puts provenance in Materials and Methods and the reference list, not the legend.

**Language**
28. **No RSNA or RYAI author-facing document states the American-vs-British spelling rule.** Verified by full-text grep of five pages for "spelling", "American English", "US English", "British" — zero hits. The rule rests on the AMA/Stedman's basis plus a 38-paper corpus with zero British forms. (The sister journal *Radiology Advances* does state it explicitly, but that is a different journal.)
29. Whether the **section-vs-slice rule is hard-enforced for RYAI** or merely preferred. The verbatim rule comes from RSNA's official *Radiology* author blog, not the RYAI instructions; two 2020 RYAI papers published "slice thickness."
30. The **"cases", "participants vs patients", "AUC not AUROC", "aOR", and "10 occurrences per abbreviation" rules** come from the *Radiology* Scientific Style Guide. **RYAI has no equivalent public scientific style page.** RYAI shares the 10-abbreviation cap, the abstract-and-text definition rule, and the AMA/Stedman's basis, so these almost certainly apply, but none could be found restated on an RYAI URL.
31. The **AMA Manual of Style (11th ed.) is paywalled**, so its gene-nomenclature section could not be quoted. The gene-italics rule is evidenced only by observed RYAI markup plus RYAI's declaration that AMA is its style basis.
32. **Gene product (protein) handling** could not be positively evidenced from RYAI prose — no corpus paper discusses a named protein product in a way that would show it set roman.
33. **"MRI scans" vs "MR images"** — see §7.2. Unresolved for RYAI.