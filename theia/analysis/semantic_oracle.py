"""How much EGFR signal is in the radiologist's semantic reads, and WHERE is it?

This is the experiment that decides whether widening THEIA's field of view is
worth a pipeline rebuild.

The motivation is a specific number. Gevaert et al. (Sci Rep 2017) report AUC
0.89 for EGFR on this same collection -- far above THEIA's 0.627 and above every
deep model in the modern literature -- from *semantic* features scored by a
radiologist. Their top predictors were emphysema and airway abnormality.

Those are whole-lung parenchymal reads. THEIA's crop is `(bbox_mm + 24) * 2.5`
around the lesion, so diffuse emphysema and airway disease elsewhere in the lung
are physically outside the pixels the model ever sees. If the 0.89 lives mostly
in those features, the model is not underfitting -- it is looking in the wrong
place, and the fix is a bigger field of view. If instead the signal is in the
nodule-morphology features, those ARE in the crop, and a rebuild buys nothing.

So every feature is tagged `in_crop` or `whole_lung` by where a radiologist would
have to look to score it, and the ablation is the whole point:

    all features      -> the ceiling this cohort's semantics support
    in_crop only      -> what THEIA can in principle already reach
    whole_lung only   -> what a wider crop would newly expose

GATE A (pre-specified): all >= 0.78 AND (all - in_crop) >= 0.06.

Two protocol notes, both load-bearing:

* The estimator is a pruned decision tree, not logistic regression. Gevaert
  states plainly that on this cohort "regularized logistic regression modeling
  failed to find a significant performance". Scoring their feature set with the
  model family they report failing would measure the wrong thing.

* Scoring runs through `baselines.oof_predictions`, i.e. THEIA's own nested
  folds. Gevaert used 100 random 70/30 splits on 186 patients. A shortfall
  against 0.89 is therefore information about protocol, not necessarily a defect
  -- read the ablation *contrast*, which is internally consistent, rather than
  the absolute number.

Run: python -m theia.analysis.semantic_oracle
"""
from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import pandas as pd

from theia.data.aim import _tag, _val

# Where a radiologist has to look to score the attribute. The split is by
# anatomy, decided before any AUC was computed, and it is the entire experiment
# -- so it is written out explicitly rather than inferred from a keyword.
WHOLE_LUNG_LABELS = {
    "Emphysema",
    "Primary Emphysema Pattern",
    "Primary Distribution",
    "Primary Emphysema Laterality",
    "Overall Emmphysema Severity",        # sic, the collection's spelling
    "Secondary Emmphysema Pattern",       # sic
    "Secondary Emhysema Distribution",    # sic
    "Secondary Emphysema Laterality",
    "Fibrosis",
    "Fibrosis Type",
    "Anatomic Fibrosis Distribution",
    "Axial Fibrosis Distribution",
    "Centrilobular Nodules - Diffuse (RB type nodules)",
    "Nodules in Contralateral Lung (gretater than 4mm noncalcified)",   # sic
    "Lung Parencyma Features",            # sic
}

IN_CROP_LABELS = {
    "Nodule Attenuation",
    "Nodule Margins-Primary Pattern",
    "Nodule Margins-Secondary Pattern",
    "Nodule Shape",
    "Nodule Calcification",
    "Nodule Periphery",
    "Nodule Internal Features",
    "Axial Location",
    "Nodule Associated Findings",
    "Satellite Nodules in Primary Lesion Lobe (greater than 4mm noncalcified)",
    # A lesion-lobe nodule sits inside a 2.5x lesion crop often enough that
    # counting it as reachable is the conservative call: it can only shrink the
    # whole_lung contribution this experiment is trying to demonstrate.
    "Nodules in Non-Lesion Lobe Same Lung (greater than 4mm noncalcified)",
}

# Labels that legitimately repeat per lesion; the others keep one value.
MULTI_LABELS = {"Nodule Associated Findings", "Lung Parencyma Features"}


def parse_labelled(path: str) -> dict[str, list[str]]:
    """label -> [values], keeping every value of the repeatable labels.

    `aim.parse_aim` flattens both repeatable labels into one `findings` list,
    which loses which label a term came from -- and that is exactly the
    distinction this module is built on ("airway abnormality" is whole-lung,
    "attachment to pleura" is in-crop). So the tags are read again here rather
    than guessing the provenance of a term downstream.
    """
    root = ET.parse(path).getroot()
    out: dict[str, list[str]] = defaultdict(list)
    for el in root.iter():
        if _tag(el) != "ImagingObservationCharacteristic":
            continue
        label = value = None
        for child in el:
            if _tag(child) == "label":
                label = _val(child)
            elif _tag(child) == "typeCode":
                display = ""
                for sub in child.iter():
                    if _tag(sub) == "displayName" and sub.get("value"):
                        display = sub.get("value").strip()
                        break
                value = (child.get("codeSystem") or display
                         or child.get("code") or "").strip()
        if not label or not value:
            continue
        if label in MULTI_LABELS or label not in out:
            out[label].append(value)
    return dict(out)


def load_semantics(aim_dir: str) -> dict[str, dict[str, list[str]]]:
    out = {}
    for fn in sorted(os.listdir(aim_dir)):
        if fn.lower().endswith(".xml"):
            try:
                out[os.path.splitext(fn)[0]] = parse_labelled(os.path.join(aim_dir, fn))
            except ET.ParseError as exc:
                print(f"[oracle] skip {fn}: {exc}")
    return out


def build_matrix(rows: list[dict], sem: dict, clinical_csv: str
                 ) -> tuple[np.ndarray, list[str], list[str]]:
    """One-hot semantic features aligned to `rows`. Returns (X, names, groups).

    A patient with no AIM file gets all-zero indicators plus an explicit
    `__no_aim` column, rather than being dropped: dropping would change the
    cohort between arms and make the AUCs incomparable to THEIA's.
    """
    ids = [r["patient_id"] for r in rows]

    vocab: dict[str, list[str]] = {}
    for label in sorted(WHOLE_LUNG_LABELS | IN_CROP_LABELS):
        vals = sorted({v for s in sem.values() for v in s.get(label, [])})
        if vals:
            vocab[label] = vals

    cols, names, groups = [], [], []
    for label, vals in vocab.items():
        grp = "whole_lung" if label in WHOLE_LUNG_LABELS else "in_crop"
        for v in vals:
            cols.append([1.0 if v in sem.get(p, {}).get(label, []) else 0.0 for p in ids])
            names.append(f"{label}={v}")
            groups.append(grp)

    cols.append([0.0 if p in sem else 1.0 for p in ids])
    names.append("__no_aim")
    groups.append("in_crop")

    # %GG is read off the scan inside the lesion, so it is an in-crop feature.
    df = pd.read_csv(clinical_csv).set_index("Case ID", drop=False)
    gg = pd.to_numeric(df.reindex(ids)["%GG"], errors="coerce")
    if gg.notna().any():
        cols.append(gg.fillna(gg.median()).to_numpy().tolist())
        names.append("%GG")
        groups.append("in_crop")

    return np.array(cols, dtype=np.float64).T, names, groups


def _tuned_tree(X: np.ndarray, y: np.ndarray, seed: int):
    """Pruned decision tree, depth and leaf size chosen by inner CV.

    Pruning is not optional at 153 patients and 40 positives: an unpruned tree
    memorises the fold. The search runs strictly inside the fitting portion.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    def make(depth, leaf, alpha):
        return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                      ccp_alpha=alpha, class_weight="balanced",
                                      random_state=seed)

    n_pos = int(y.sum())
    n_splits = max(2, min(5, n_pos, len(y) - n_pos))
    best, best_auc = (3, 10, 0.0), -1.0
    for depth in (2, 3, 4, 5, None):
        for leaf in (5, 10, 20):
            for alpha in (0.0, 0.005, 0.01, 0.02):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                scores = []
                for a, b in skf.split(X, y):
                    if len(set(y[a].tolist())) < 2 or len(set(y[b].tolist())) < 2:
                        continue
                    m = make(depth, leaf, alpha).fit(X[a], y[a])
                    scores.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
                if scores and float(np.mean(scores)) > best_auc:
                    best_auc, best = float(np.mean(scores)), (depth, leaf, alpha)
    return make(*best).fit(X, y)


def score(X: np.ndarray, rows: list[dict], cfg, gene: str, seeds: list[int],
          fit_fn) -> tuple[float, float, list[float]]:
    from theia.analysis.baselines import oof_predictions
    from theia.engine.evaluate import pooled_metrics

    aucs = []
    for s in seeds:
        cfg["seed"] = s
        oof = oof_predictions(X, rows, gene, cfg, fit_fn=fit_fn)
        aucs.append(pooled_metrics(oof, [gene], cfg.eval.bootstrap_n)[f"{gene.lower()}_auc"])
    return mean(aucs), (stdev(aucs) if len(aucs) > 1 else 0.0), aucs


def main() -> None:
    from theia.analysis.baselines import _tuned_logreg
    from theia.analysis.diagnostics import load_rows
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--seeds", default="1337,7,42,3,11,23,71,101,202,404")
    ap.add_argument("--out", default="results/semantic_oracle.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    sem = load_semantics(os.path.join(cfg.paths.raw_dir, "aim"))
    clinical = os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv")

    X, names, groups = build_matrix(rows, sem, clinical)
    have = sum(1 for r in rows if r["patient_id"] in sem)
    print(f"[oracle] {len(rows)} patients, {have} with an AIM read, "
          f"{X.shape[1]} features "
          f"({groups.count('in_crop')} in_crop / {groups.count('whole_lung')} whole_lung)")

    g = np.array(groups)
    arms = {
        "all semantic": X,
        "in_crop only": X[:, g == "in_crop"],
        "whole_lung only": X[:, g == "whole_lung"],
    }

    results = {}
    for est_name, fit_fn in (("tree", _tuned_tree), ("logreg", _tuned_logreg)):
        print(f"\n[oracle] estimator = {est_name}")
        for arm, Xa in arms.items():
            m, s, per = score(Xa, rows, cfg, a.gene, seeds, fit_fn)
            results[f"{est_name} | {arm}"] = {
                "mean": m, "sd": s, "per_seed": per, "dim": int(Xa.shape[1])}
            print(f"[oracle]   {arm:18s} {m:.3f} +/- {s:.3f}  (d={Xa.shape[1]})")

    tree_all = results["tree | all semantic"]["mean"]
    tree_crop = results["tree | in_crop only"]["mean"]
    lift = tree_all - tree_crop
    gate = bool(tree_all >= 0.78 and lift >= 0.06)

    print(f"\n[oracle] GATE A: all={tree_all:.3f} (need >=0.78), "
          f"whole-lung contribution={lift:+.3f} (need >=0.06) -> "
          f"{'PASS' if gate else 'FAIL'}")

    json.dump({"gene": a.gene, "seeds": seeds, "n": len(rows), "n_with_aim": have,
               "feature_names": names, "feature_groups": groups,
               "arms": results,
               "gate_a": {"all": tree_all, "in_crop": tree_crop,
                          "whole_lung_contribution": lift, "passed": gate}},
              open(a.out, "w"), indent=2)
    print(f"[oracle] wrote {a.out}")


if __name__ == "__main__":
    main()
