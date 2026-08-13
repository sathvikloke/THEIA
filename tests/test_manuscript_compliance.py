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
EXTERNAL = "results/external_grounding_canonical.json"

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
    n = _words(_between(src, r"\section*{Abstract}", r"\section{Introduction}"))
    assert n <= LIMIT_ABSTRACT, f"abstract is {n} words, limit {LIMIT_ABSTRACT}"


def test_body_within_limit(src):
    n = _words(_between(src, r"\section{Introduction}",
                        r"\section*{Data and code availability}"))
    assert n <= LIMIT_BODY, f"body is {n} words, limit {LIMIT_BODY}"


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
    ("Summary Statement", r"Summary Statement"),
    ("Key Points", r"Key Points"),
    ("P-value threshold", r"P < \.05"),
    ("statistical software with version", r"Python 3\.\d+\.\d+"),
    ("ethics statement", r"subsection\{Ethics\}"),
    ("CLAIM cited", r"\\citep\{claim2020\}"),
    ("abstract demographics", r"mean age, \d+ years"),
    ("accrual-window statement", r"[Nn]o accrual window"),
])
def test_required_element_present(src, label, pattern):
    assert re.search(pattern, src), f"missing required element: {label}"


def test_abstract_uses_journal_headings(src):
    """Purpose / Materials and Methods / Results / Conclusion -- no Background.

    Radiology proper takes a Background heading; Radiology: AI does not. This is
    the kind of difference that survives a careful read because both look right.
    """
    abstract = _between(src, r"\section*{Abstract}", r"\section{Introduction}")
    for h in ("Purpose", "Materials and Methods", "Results", "Conclusion"):
        assert re.search(r"\\textbf\{" + h + r"\.", abstract), f"missing {h}"
    assert not re.search(r"\\textbf\{Background", abstract), \
        "Radiology: AI abstracts have no Background heading"


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
    assert f"{beat} of {len(rows)} external evaluations" in flat, (
        f"manuscript must say '{beat} of {len(rows)} external evaluations'")


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
