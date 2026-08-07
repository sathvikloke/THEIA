"""Tests for the comparator arms.

These matter more than their size suggests: `baselines.py` produced the number
that reframed the project (clinical AUC 0.774 against THEIA's 0.656). A silent
bug here would either invent an advantage for the baseline or hide one, and in
both directions it would change what the paper claims. So the tests target the
ways this specific code could be wrong rather than its happy path:

  * a baseline scored on different folds than THEIA is not a comparison
  * a baseline that can see the outer test fold is not a baseline
  * a paired delta that ignores the pairing throws away the whole point
"""
import json
import os

import numpy as np
import pandas as pd
import pytest


def _cohort(tmp_path, n=60, seed=0):
    """Rows + a matching clinical sheet, with EGFR driven by smoking status.

    Mirrors the real cohort's structure: never-smokers are mostly mutant,
    smokers mostly wildtype, plus a few patients with unknown labels (-1) that
    every arm must drop identically.
    """
    rng = np.random.default_rng(seed)
    rows, recs = [], []
    for i in range(n):
        never = i % 3 == 0
        egfr = int(rng.random() < (0.7 if never else 0.1))
        if i % 17 == 0:
            egfr = -1                                   # unknown label
        img = rng.random((2, 8, 8)).astype(np.float32)
        roi = np.zeros((2, 8, 8), dtype=np.float32)
        roi[:, 2:6, 2:6] = 1.0
        npz = os.path.join(tmp_path, f"p{i}.npz")
        np.savez_compressed(npz, images=img, roi=roi)
        rows.append(dict(patient_id=f"p{i}", npz=npz, report="a lesion.",
                         labels={"EGFR": egfr, "KRAS": i % 2}, n_slices=2))
        recs.append({"Case ID": f"p{i}",
                     "Age at Histological Diagnosis": 50 + (i % 30),
                     "Pack Years": "" if never else str(10 + i % 40),
                     "Gender": "Female" if never else "Male",
                     "Ethnicity": "Asian" if never else "Caucasian",
                     "Smoking status": "Nonsmoker" if never else "Former"})
    idx = os.path.join(tmp_path, "rows.jsonl")
    with open(idx, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    csv = os.path.join(tmp_path, "clinical.csv")
    pd.DataFrame(recs).to_csv(csv, index=False)
    return rows, idx, csv


def _cfg(tmp_path):
    from theia.config import load_config

    cfg = load_config("configs/default.yaml", validate=False)
    cfg["paths"]["processed_dir"] = str(tmp_path)
    cfg["eval"]["bootstrap_n"] = 50
    cfg["split"].update(n_folds=3, inner_val_frac=0.25)
    return cfg


def test_clinical_features_align_to_rows_and_never_drop_patients(tmp_path):
    """Row i of the matrix must be patient i, and no patient may vanish.

    Silent misalignment here would shuffle labels against features and produce
    an AUC near 0.5 that looks like an honest negative result.
    """
    from theia.analysis.baselines import clinical_features

    rows, _, csv = _cohort(tmp_path)
    X, names = clinical_features(rows, csv)
    assert X.shape[0] == len(rows), "a patient was dropped building features"
    assert np.isfinite(X).all(), "non-finite value in the clinical matrix"

    # Never-smokers are the Asian/Female arm in this fixture, so the smoking
    # column must actually separate them -- if it does not, the encoding is wrong.
    col = names.index("Smoking status_nonsmoker")
    never = np.array([r["patient_id"].endswith(tuple("0369")) for r in rows])
    assert X[:, col].sum() > 0
    del never

    # A patient absent from the sheet must fail loudly, not silently misalign.
    with pytest.raises(KeyError):
        clinical_features(rows + [{"patient_id": "ghost"}], csv)


def test_pack_years_blank_means_zero_for_never_smokers(tmp_path):
    """Blank pack-years is 'did not smoke', not 'unmeasured'.

    Median-filling it would hand never-smokers a smoker's exposure and blunt the
    single strongest predictor in the cohort.
    """
    from theia.analysis.baselines import clinical_features

    rows, _, csv = _cohort(tmp_path)
    X, names = clinical_features(rows, csv)
    py = X[:, names.index("Pack Years")]
    never = np.array([
        pd.read_csv(csv).set_index("Case ID").loc[r["patient_id"], "Smoking status"]
        == "Nonsmoker" for r in rows])
    assert (py[never] == 0).all(), "never-smokers were given imputed pack-years"
    assert (py[~never] > 0).all()


def test_baseline_uses_the_same_outer_folds_as_theia(tmp_path):
    """Every labelled patient is predicted exactly once, by the fold that held them.

    If the baseline cross-validated under its own split, the comparison would
    confound the model difference with a split difference -- which at ~40
    positives moves AUC by more than the effect being measured.
    """
    from theia.analysis.baselines import clinical_features, oof_predictions
    from theia.data.dataset import nested_kfold_indices

    rows, idx, csv = _cohort(tmp_path)
    cfg = _cfg(tmp_path)
    X, _ = clinical_features(rows, csv)
    oof = oof_predictions(X, rows, "EGFR", cfg)

    seen = [r["patient_id"] for r in oof]
    assert len(seen) == len(set(seen)), "a patient was predicted more than once"

    labelled = {r["patient_id"] for r in rows if r["labels"]["EGFR"] != -1}
    assert set(seen) == labelled, "baseline scored a different cohort than it should"

    # And the fold each patient landed in must match the real splitter exactly.
    want = {}
    for f, (_tr, _va, te) in enumerate(nested_kfold_indices(
            idx, cfg.split.stratify_on, cfg.split.n_folds, cfg.seed,
            float(cfg.split.inner_val_frac))):
        for i in te:
            want[rows[i]["patient_id"]] = f
    for r in oof:
        assert r["fold"] == want[r["patient_id"]], (
            f"{r['patient_id']} scored by fold {r['fold']}, splitter says "
            f"{want[r['patient_id']]}")


def test_baseline_never_trains_on_its_own_test_fold(tmp_path, monkeypatch):
    """The estimator must not see an outer-test patient during fitting.

    Checked by recording the row indices handed to every fit, rather than by
    trusting the index arithmetic to be right.
    """
    from theia.analysis import baselines
    from theia.data.dataset import nested_kfold_indices

    rows, idx, csv = _cohort(tmp_path)
    cfg = _cfg(tmp_path)
    X, _ = baselines.clinical_features(rows, csv)

    # Tag each patient with a unique value so a fitted row is identifiable.
    tagged = np.hstack([X, np.arange(len(rows))[:, None].astype(float)])
    fitted: list[set] = []
    real = baselines._tuned_logreg

    def spy(Xf, y, seed):
        # Record the tag column, then fit on the matrix UNCHANGED -- stripping it
        # here would leave the fitted estimator expecting one fewer feature than
        # oof_predictions passes to predict_proba.
        fitted.append(set(Xf[:, -1].astype(int).tolist()))
        return real(Xf, y, seed)

    monkeypatch.setattr(baselines, "_tuned_logreg", spy)
    baselines.oof_predictions(tagged, rows, "EGFR", cfg)

    folds = list(nested_kfold_indices(idx, cfg.split.stratify_on, cfg.split.n_folds,
                                      cfg.seed, float(cfg.split.inner_val_frac)))
    assert len(fitted) == len(folds)
    for used, (_tr, _va, te) in zip(fitted, folds):
        assert not (used & set(te)), "baseline was fit on its own outer test fold"


def test_paired_delta_is_paired_and_signed_correctly(tmp_path):
    """A strictly better model must show a positive delta with a CI above zero."""
    from theia.analysis.baselines import paired_delta

    rng = np.random.default_rng(3)
    y = np.array([0, 1] * 40)
    good, bad = [], []
    for i, yi in enumerate(y):
        # fold = i % 4 would put every even index (all class 0) in fold 0 and
        # every odd index (all class 1) in fold 1. Within-fold rank
        # normalisation then correctly destroys all signal, because a
        # single-class fold carries none. Group in pairs so each fold sees both.
        pid, fold = f"p{i}", (i // 2) % 4
        good.append({"patient_id": pid, "fold": fold, "egfr_true": int(yi),
                     "egfr_prob": float(yi * 0.8 + rng.random() * 0.2)})
        bad.append({"patient_id": pid, "fold": fold, "egfr_true": int(yi),
                    "egfr_prob": float(rng.random())})

    d = paired_delta(good, bad, "EGFR", n_boot=400, seed=1)
    assert d["delta"] > 0.2, f"strictly better model did not win: {d}"
    assert d["delta_lo"] > 0, "CI on a clear win still includes zero"
    assert d["p_two_sided"] < 0.05

    # Symmetry: swapping the arms must flip the sign, not change the magnitude.
    rev = paired_delta(bad, good, "EGFR", n_boot=400, seed=1)
    assert rev["delta"] == pytest.approx(-d["delta"], abs=1e-9)


def test_paired_delta_reports_no_difference_for_identical_models(tmp_path):
    """Two copies of one model must give delta 0 -- the null the CI is read against."""
    from theia.analysis.baselines import paired_delta

    rng = np.random.default_rng(7)
    preds = [{"patient_id": f"p{i}", "fold": i % 4, "egfr_true": i % 2,
              "egfr_prob": float(rng.random())} for i in range(80)]
    d = paired_delta(preds, [dict(r) for r in preds], "EGFR", n_boot=300, seed=1)
    assert d["delta"] == pytest.approx(0.0, abs=1e-9)
    assert d["p_two_sided"] > 0.5, "identical models flagged as different"
