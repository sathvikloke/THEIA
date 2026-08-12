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
CANONICAL = "results/CANONICAL.json"


def _text():
    # Docs use a Unicode minus; normalise so a real match is not missed.
    return {f: open(f).read().replace("−", "-") for f in DOCS}


def _multiseed(key):
    """Pooled metric across the CANONICAL headline runs.

    Deliberately not a glob. `results/ms-s*.json` used to define the headline set,
    and it silently included a run whose fold 1 stalled -- scored on 122 of 153
    patients, reporting the highest AUC of the three seeds. That kept 0.627 alive
    in three documents after the retrained run had moved the number to 0.617. The
    set is now named in one file so it cannot drift again.
    """
    runs = json.load(open(CANONICAL))["headline_runs"]
    vals = []
    for p in runs:
        v = json.load(open(p))["pooled"].get(key)
        if v is not None and v == v:
            vals.append(v)
    return vals


def test_canonical_runs_all_exist_and_none_stalled():
    """The headline set must not contain a degenerate run, ever again."""
    blob = json.load(open(CANONICAL))
    for p in blob["headline_runs"]:
        d = json.load(open(p))
        stalled = [i for i, f in enumerate(d["folds"])
                   if f.get("stalled") or f["test"].get("egfr_auc") is None]
        assert not stalled, f"{p} has stalled fold(s) {stalled} and cannot be canonical"
        assert d["pooled"]["egfr_n"] == 153, (
            f"{p} scored {d['pooled']['egfr_n']} patients, not the full 153")


def test_superseded_runs_are_excluded_from_the_headline_set():
    blob = json.load(open(CANONICAL))
    for p in blob.get("superseded", {}):
        assert p not in blob["headline_runs"], f"{p} is both superseded and canonical"


def test_canonical_file_headline_matches_the_archives_it_names():
    """The convenience copy in CANONICAL.json must not drift from the runs."""
    blob = json.load(open(CANONICAL))
    vals = _multiseed("egfr_auc")
    assert round(mean(vals), 3) == blob["headline"]["mean"]
    assert round(stdev(vals), 3) == blob["headline"]["sd"]


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
        "0.656 [0.560, 0.747]": "run 6's EGFR, superseded by the canonical headline",
        "0.627 +/- 0.041": "the headline before the stalled seed-1337 run was "
                           "retrained; superseded by 0.617 +/- 0.024",
        "0.627 ± 0.041": "same, with a Unicode plus-minus",
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
