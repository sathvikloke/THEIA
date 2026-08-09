"""The documented numbers must match the archives they came from.

Every headline in this project has been restated across README, RESULTS and the
model card, and several were quietly superseded while the prose kept the old
figure -- the config cited a KRAS AUC from a run four configurations out of date,
and RESULTS led with a single-run delta after the multi-seed one existed. Prose
does not have tests unless someone writes them.
"""
import glob
import json
from statistics import mean, stdev

import pytest

DOCS = ["README.md", "docs/RESULTS.md", "docs/MODEL_CARD.md"]


def _text():
    # Docs use a Unicode minus; normalise so a real match is not missed.
    return {f: open(f).read().replace("−", "-") for f in DOCS}


def _multiseed(key):
    vals = []
    for p in sorted(glob.glob("results/ms-s*.json")):
        v = json.load(open(p))["pooled"].get(key)
        if v is not None and v == v:
            vals.append(v)
    return vals


@pytest.mark.parametrize("key,label", [("egfr_auc", "EGFR"), ("kras_auc", "KRAS")])
def test_headline_auc_in_docs_matches_the_archives(key, label):
    vals = _multiseed(key)
    if len(vals) < 2:
        pytest.skip("need >= 2 archived seeds")
    m, s = f"{mean(vals):.3f}", f"{stdev(vals):.3f}"
    text = _text()
    assert any(m in t for t in text.values()), f"{label} mean {m} appears in no doc"
    assert any(s in t for t in text.values()), f"{label} sd {s} appears in no doc"


def test_incremental_delta_in_docs_matches_the_archive():
    """The number the whole project turns on must be the archived one."""
    blob = json.load(open("results/incremental_value.json"))
    m, s = f"{blob['delta_mean']:.3f}", f"{blob['delta_sd']:.3f}"
    text = _text()
    assert any(m in t for t in text.values()), f"delta mean {m} appears in no doc"
    assert any(s in t for t in text.values()), f"delta sd {s} appears in no doc"


def test_no_superseded_figures_survive_in_prose():
    """Specific stale numbers that were once headlines and are no longer true."""
    stale = {
        "0.522": "an old KRAS pooled AUC, superseded by the multi-seed 0.509",
        "0.656 [0.560, 0.747]": "run 6's EGFR, superseded by 0.627 +/- 0.041",
    }
    for f, t in _text().items():
        for bad, why in stale.items():
            # RESULTS.md documents run history on purpose; the ban is on the
            # files a newcomer reads first.
            if f == "docs/RESULTS.md":
                continue
            assert bad not in t, f"{f} still cites {bad} — {why}"


def test_the_config_does_not_cite_a_superseded_number():
    cfg = open("configs/default.yaml").read()
    assert "0.522" not in cfg, "config still cites the superseded KRAS AUC"
