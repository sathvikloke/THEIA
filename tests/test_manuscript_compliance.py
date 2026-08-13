"""The manuscript has hard limits and required elements. These guard them.

Radiology: Artificial Intelligence enforces per-article-type caps (Original
Research: 3000 words Introduction-Discussion, 250-word abstract, 35 references,
6 figures, 4 tables) and a set of required elements that are easy to lose during
an edit -- a Summary Statement, Key Points, the P-value threshold, the statistical
software with versions.

Two of these are not style points. The journal uses DOUBLE-ANONYMIZED peer review,
so an author name or a personal repository URL surviving in the main file is a
desk-reject-grade error that no proofread reliably catches; and every Results
number the manuscript quotes must still match the archived JSON it came from,
which is how the stale-run bugs in this project have always announced themselves.

Run: pytest tests/test_manuscript_compliance.py -v
"""
from __future__ import annotations

import json
import os
import re

import pytest

MAIN = "paper/main.tex"
TITLE_PAGE = "paper/title_page.tex"
COVER_LETTER = "paper/cover_letter.tex"
FIGURES = "paper/figures.tex"
OVERLAY_PANELS = "figures/fig5_overlays_panels.json"
EXTERNAL = "results/external_grounding_canonical.json"

LIMIT_SUMMARY_STATEMENT = 255   # characters, not words

# Original Research. If the article type ever changes, change these together with
# the header comment in main.tex -- the point of the constant is that the limits
# and the declared type cannot drift apart silently.
LIMIT_ABSTRACT = 250
LIMIT_BODY = 3000
LIMIT_REFS = 35
LIMIT_FIGURES = 6
LIMIT_TABLES = 4

pytestmark = pytest.mark.skipif(not os.path.exists(MAIN),
                                reason="manuscript not present")


@pytest.fixture(scope="module")
def src() -> str:
    return open(MAIN).read()


def _words(tex: str) -> int:
    """Word count with LaTeX markup removed, citations counted as one word."""
    t = re.sub(r"%.*", "", tex)
    t = re.sub(r"\\(cite[a-z]*|ref|label|url)\{[^}]*\}", " X ", t)
    t = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", t)
    t = re.sub(r"[{}$\\_^&~]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


def _between(src: str, start: str, end: str) -> str:
    return src[src.index(start):src.index(end)]


# --- hard limits ----------------------------------------------------------

def test_abstract_within_limit(src):
    n = _words(_between(src, r"Abstract\par}", r"\section*{Introduction}"))
    assert n <= LIMIT_ABSTRACT, f"abstract is {n} words, limit {LIMIT_ABSTRACT}"


#: Author-action \fbox placeholders are deleted at submission and replaced by the
#: real ethics and AI-disclosure statements. Counting the instructions overstates
#: the body; ignoring them understates it, because the replacements do count. So
#: the boxes are excluded and this fixed allowance is charged instead.
PLACEHOLDER_ALLOWANCE = 150


def test_body_within_limit(src):
    body = _between(src, r"\section*{Introduction}", r"\section*{Acknowledgments}")
    boxes = re.findall(r"\\fbox\{\\parbox.*?\}\}", body, re.S)
    n = _words(body) - sum(_words(b) for b in boxes) + PLACEHOLDER_ALLOWANCE
    assert n <= LIMIT_BODY, (
        f"body is {n} words ({_words(body) - sum(_words(b) for b in boxes)} written "
        f"+ {PLACEHOLDER_ALLOWANCE} reserved for the placeholders), "
        f"limit {LIMIT_BODY}")


def test_summary_statement_within_character_limit(src):
    """RSNA caps the summary statement at 255 CHARACTERS, not words."""
    m = re.search(r"\\textbf\{Summary statement\.\}\s*\\textbf\{(.*?)\}\n\n",
                  src, re.S)
    assert m, "no summary statement found"
    text = " ".join(m.group(1).split())
    assert len(text) <= LIMIT_SUMMARY_STATEMENT, (
        f"summary statement is {len(text)} characters, "
        f"limit {LIMIT_SUMMARY_STATEMENT}")


def test_counts_within_limits(src):
    assert len(re.findall(r"\\includegraphics", src)) <= LIMIT_FIGURES
    assert len(re.findall(r"\\begin\{table\}", src)) <= LIMIT_TABLES
    assert len(re.findall(r"\\bibitem", src)) <= LIMIT_REFS


def test_no_uncited_references(src):
    """An uncited bibliography entry reads as a padded reference list."""
    keys = re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", src)
    uncited = [k for k in keys
               if not re.search(r"\\citep\{[^}]*" + re.escape(k), src)]
    assert not uncited, f"uncited references: {uncited}"


# --- required elements ----------------------------------------------------

@pytest.mark.parametrize("label,pattern", [
    ("summary statement", r"\\textbf\{Summary statement\.\}"),
    ("key points", r"\\textbf\{Key points\}"),
    # Spelled out, not "P < .05": RSNA allows comparison symbols only inside
    # parentheses, so the significance threshold must be written in words.
    ("P-value threshold", r"\\textit\{P\} values less than \.05"),
    ("statistical software with version", r"Python 3\.\d+\.\d+"),
    ("ethics statement", r"\\subsection\*\{Ethics\}"),
    ("CLAIM cited", r"\\citep\{claim2020\}"),
    ("abstract demographics", r"mean age, \d+ years"),
    ("accrual-window statement", r"[Nn]o accrual window"),
    ("keywords", r"\\textbf\{Keywords:\}"),
])
def test_required_element_present(src, label, pattern):
    assert re.search(pattern, src), f"missing required element: {label}"


def test_abstract_uses_journal_headings(src):
    """Purpose / Materials and Methods / Results / Conclusion -- no Background.

    Radiology proper takes a Background heading; Radiology: AI does not. This is
    the kind of difference that survives a careful read because both look right.
    """
    abstract = _between(src, r"Abstract\par}", r"\section*{Introduction}")
    for h in ("Purpose", "Materials and Methods", "Results", "Conclusion"):
        assert re.search(r"\\textbf\{" + h + r":", abstract), f"missing {h}"
    assert not re.search(r"\\textbf\{Background", abstract), \
        "Radiology: AI abstracts have no Background heading"


def test_section_order_matches_rsna(src):
    """Order: body, acknowledgments, references, figure legends, tables.

    Acknowledgments before References is the counterintuitive one, and Tables
    last. Getting this wrong is invisible when reading a compiled PDF.
    """
    order = ["\\section*{Introduction}", "\\section*{Materials and Methods}",
             "\\section*{Results}", "\\section*{Discussion}",
             "\\section*{Acknowledgments}", "\\section*{References}",
             "\\section*{Figure Legends}", "\\section*{Tables}"]
    positions = [src.index(s) for s in order]
    assert positions == sorted(positions), (
        "sections are out of RSNA order: "
        f"{[order[i] for i in sorted(range(len(order)), key=lambda k: positions[k])]}")


def test_no_page_numbers(src):
    """RSNA prohibits page numbers -- ScholarOne adds its own at PDF build."""
    assert "\\pagestyle{empty}" in src, "page numbers must be suppressed"


def test_double_spaced_and_ragged_right(src):
    assert "\\doublespacing" in src
    assert "\\raggedright" in src


# --- house style: statistics, spelling, terminology ------------------------

def test_p_values_have_no_leading_zero(src):
    bad = re.findall(r"P\}?\s*\$?\s*[<>=]\s*\$?\s*0\.\d", src)
    assert not bad, f"P values must drop the leading zero (P = .03): {bad}"


def test_confidence_intervals_use_the_colon_comma_form(src):
    """95% CI: lower, upper -- never a dash or 'to' between the bounds."""
    bad = re.findall(r"95\\%\s*CI[^:]{0,3}[-–]", src)
    assert not bad, f"CIs must read '95% CI: lower, upper': {bad}"


def test_american_spelling(src):
    """RYAI copyedits to the AMA Manual of Style; published corpus is 100% US.

    Reference titles are excluded: a cited article's title is reproduced as
    published, so 'multi-tumour cohort' in a British journal's title is correct.
    """
    body = src[:src.index("\\section*{References}")]
    british = ["localis", "generalis", "tumour", "randomis", "normalis",
               "summaris", "artefact", "labelled", "behaviour", "colour",
               "modelling", "recognis", "organis"]
    hits = [w for w in british if re.search(w, body, re.I)]
    assert not hits, f"British spellings in the manuscript body: {hits}"


def test_people_are_patients_not_subjects(src):
    """RSNA: never call people 'subjects' in a retrospective imaging study.

    'not-human-subjects-research' is the regulatory term of art and is allowed.
    """
    body = re.sub(r"not-human-subjects-research", "", src)
    body = re.sub(r"%.*", "", body)
    assert not re.search(r"\bsubjects\b", body), \
        "use 'patients', never 'subjects'"


def test_no_self_evaluation(src):
    body = src[:src.index("\\section*{References}")]
    bad = re.findall(r"\b(novel|unique|ground-?breaking|first-ever)\b",
                     body, re.I)
    assert not bad, f"RSNA forbids claims of novelty or priority: {bad}"


def test_section_not_slice(src):
    body = re.sub(r"%.*", "", src)
    assert not re.search(r"\bslices?\b", body), \
        "RSNA: use 'section', not 'slice', for cross-sectional images"


def test_abbreviation_list_within_cap(src):
    """RSNA caps a manuscript at 10 abbreviations."""
    m = re.search(r"\\textbf\{Abbreviations:\}(.*?)\n\n", src, re.S)
    assert m, "no abbreviation list on the abbreviated title page"
    n = len(re.findall(r"=", m.group(1)))
    assert n <= 10, f"{n} abbreviations listed, cap is 10"


# --- anonymization --------------------------------------------------------

@pytest.mark.parametrize("token", ["github.com", "sathvikloke", "@gmail",
                                   "Stanford", "Maastro", "Palo Alto"])
def test_main_file_is_anonymized(src, token):
    """Double-anonymized review: no identifying strings in the main file.

    Institution names are included because naming the contributing centre of a
    public collection identifies the cohort but not the authors -- yet a reviewer
    guide asks for them to be generalised anyway, and the manuscript says "a
    Dutch centre" and "a surgical series" instead.
    """
    assert token not in src, f"{token!r} in the anonymized main file"


def test_title_page_exists_and_carries_the_identifying_material():
    assert os.path.exists(TITLE_PAGE), "separate full title page is required"
    tp = open(TITLE_PAGE).read()
    for field in ("Funding", "Conflicts of interest", "Data sharing statement",
                  "Institutional review board", "Informed consent"):
        assert field in tp, f"title page missing: {field}"


def test_cover_letter_carries_every_required_item():
    """RSNA names four REQUIRED items; Editorial Policies add two more."""
    assert os.path.exists(COVER_LETTER), "a cover letter is required"
    cl = open(COVER_LETTER).read()
    for field in ("Subject overlap", "Conflicts of interest", "Sole submission",
                  "artificial intelligence", "Control of the data",
                  "Prior presentation"):
        assert field in cl, f"cover letter missing required item: {field}"


def test_cover_letter_omits_reviewer_suggestions():
    """Suggested/opposed reviewers go in ScholarOne Step 5, not the letter."""
    cl = re.sub(r"%.*", "", open(COVER_LETTER).read())
    assert not re.search(r"[Ss]uggested reviewers|[Oo]pposed reviewers", cl), \
        "reviewer preferences belong in ScholarOne, not the cover letter"


def test_ai_use_disclosed_in_manuscript_as_well_as_cover_letter(src):
    """RSNA requires the AI-use disclosure in BOTH places, not one."""
    assert re.search(r"[Uu]se of artificial intelligence", src), \
        "manuscript is missing the AI-use disclosure"


# --- figure legends --------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(OVERLAY_PANELS),
                    reason="overlay figure not generated")
def test_overlay_legend_matches_the_patients_actually_shown(src):
    """RSNA requires age and sex in a legend showing human images.

    The panel demographics are emitted beside the PNG by
    theia.analysis.figures, so a legend typed by hand cannot silently drift
    from the checkpoint and fold that produced the panels.
    """
    panels = json.load(open(OVERLAY_PANELS))["panels"]
    flat = " ".join(src.split())
    for p in panels:
        noun = "man" if p["sex"] == "male" else "woman"
        phrase = f"({p['panel']}) {p['age']}-year-old {noun}"
        assert phrase in flat, f"Figure 6 legend missing or wrong: {phrase}"


def test_every_figure_legend_is_present_in_both_files(src):
    assert os.path.exists(FIGURES), "figures ship as a separate file"
    figs = open(FIGURES).read()
    for n in range(1, 7):
        assert re.search(rf"\\textbf\{{Figure {n}:", src), \
            f"main file missing legend for Figure {n}"
        assert re.search(rf"\\textbf\{{Figure {n}:", figs), \
            f"figure file missing legend for Figure {n}"


# --- the numbers still match the archive ----------------------------------

@pytest.mark.skipif(not os.path.exists(EXTERNAL), reason="no external results")
def test_primary_endpoint_matches_archive(src):
    """Table 3's summary row must equal what the canonical run actually produced."""
    d = json.load(open(EXTERNAL))
    rows = d["rows"]
    assert d["n_folds"] == len(rows)

    quoted_lift = float(re.search(r"lift\s*\n?of \$\+(0\.\d+)\$", src).group(1))
    assert abs(quoted_lift - d["mean_mass_lift"]) < 5e-4, (
        f"manuscript quotes +{quoted_lift} but the archive has "
        f"{d['mean_mass_lift']:.4f}")

    beat = sum(1 for r in rows
               if r["grounding_mass"] > r["grounding_mass_shuffled"])
    assert beat == d["folds_beating_shuffle"]
    # Collapse newlines: LaTeX wraps prose, so the phrase is often split.
    flat = " ".join(src.split())
    assert f"{beat} of {len(rows)}" in flat, (
        f"manuscript must say '{beat} of {len(rows)}' evaluations")


@pytest.mark.skipif(not os.path.exists(EXTERNAL), reason="no external results")
def test_bimodality_claim_holds():
    """The Discussion claims no evaluation sits between 0.000 and 0.757 pointing.

    This is the answer to "your primary endpoint passes by one fold", so it has
    to keep being true rather than have been true once.
    """
    rows = json.load(open(EXTERNAL))["rows"]
    pointing = sorted(r["grounding_pointing"] for r in rows)
    middling = [p for p in pointing if 0.0 < p < 0.757]
    assert not middling, f"evaluations in the claimed empty band: {middling}"
