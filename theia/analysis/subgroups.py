"""Re-score the archived headline on the subgroups a reviewer will ask about.

Two facts about this cohort make the pooled number softer than it looks, and both
are visible from the archived predictions without retraining anything.

HISTOLOGY. Every EGFR-mutant patient in the collection is an adenocarcinoma; the
35 squamous and 4 NSCLC-NOS patients are wild-type without exception. So a model
that learns nothing about EGFR but learns to recognise squamous morphology still
scores above chance, because ~13% of the labelled cohort is guaranteed-negative on
histology alone. `docs/MODEL_CARD.md` already declares non-adenocarcinoma out of
scope while the analysis cohort contains it -- this resolves that contradiction by
measuring what it costs.

SUPERVISION TIER. 119 of 158 patients have a pixel segmentation; the rest are
AIM-only, where the crop comes from an annotated point and the ROI is written as
zeros. The two tiers differ in crop geometry, and `has_mask` is separately
measurable from the pixels, so tier is a confound rather than a nuisance.

Neither correction requires new compute -- the per-patient probabilities are in
`results/*.json` under folds[i]['oof']. Rank normalisation is redone within each
subgroup, because ranks are only meaningful relative to the patients being scored.

Run: python -m theia.analysis.subgroups
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_oof(path: str, gene: str = "egfr") -> list[dict]:
    d = json.load(open(path))
    return [r for f in d.get("folds", []) for r in (f.get("oof") or [])
            if r.get(f"{gene}_true") in (0, 1) and r.get(f"{gene}_prob") is not None]


def rank_within_fold(recs: list[dict], key: str) -> dict[str, float]:
    """Rank-normalise inside each fold, over whatever patients are present.

    Recomputed per subgroup rather than reused from the full cohort: a rank is a
    statement about a patient's position among the patients being compared, so
    carrying over full-cohort ranks into a subgroup would silently import
    information about the patients that were just excluded.
    """
    by_fold: dict[int, list[dict]] = {}
    for r in recs:
        by_fold.setdefault(r.get("fold", 0), []).append(r)
    out = {}
    for group in by_fold.values():
        p = np.array([r[key] for r in group], dtype=float)
        ranks = p.argsort().argsort() / max(len(p) - 1, 1)
        for r, v in zip(group, ranks):
            out[r["patient_id"]] = float(v)
    return out


def auc_on(recs: list[dict], keep: set[str], gene: str = "egfr") -> tuple[float, int, int]:
    sub = [r for r in recs if r["patient_id"] in keep]
    if not sub:
        return float("nan"), 0, 0
    ranked = rank_within_fold(sub, f"{gene}_prob")
    ids = sorted(ranked)
    y = np.array([next(r[f"{gene}_true"] for r in sub if r["patient_id"] == i) for i in ids])
    if len(set(y.tolist())) < 2:
        return float("nan"), len(ids), int(y.sum())
    return float(roc_auc_score(y, [ranked[i] for i in ids])), len(ids), int(y.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default="results/ms-s1337.json,results/ms-s42.json,results/ms-s7.json")
    ap.add_argument("--gene", default="egfr")
    ap.add_argument("--out", default="results/subgroups.json")
    a = ap.parse_args()

    from theia.config import load_config
    cfg = load_config(a.config)

    rows = [json.loads(l) for l in open(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))]
    has_mask = {r["patient_id"] for r in rows if r.get("has_mask")}
    clin = pd.read_csv(os.path.join(cfg.paths.raw_dir, "clinical", "clinical.csv"))
    hist = dict(zip(clin["Case ID"], clin["Histology "].astype(str).str.strip()))
    adeno = {p for p, h in hist.items() if h == "Adenocarcinoma"}

    runs = [p for p in a.runs.split(",") if os.path.exists(p)]
    subsets = {
        "full cohort (current headline)": None,
        "adenocarcinoma only": adeno,
        "segmented only": has_mask,
        "segmented adenocarcinoma": adeno & has_mask,
    }

    table: dict[str, dict] = {}
    print(f"{'subset':<32} {'n':>4} {'pos':>4}   " + "  ".join(f"s{os.path.basename(p)[4:-5]:>6}" for p in runs) + "     mean")
    for name, keep in subsets.items():
        aucs, n, npos = [], 0, 0
        for p in runs:
            recs = load_oof(p, a.gene)
            k = {r["patient_id"] for r in recs} if keep is None else keep
            auc, n, npos = auc_on(recs, k, a.gene)
            aucs.append(auc)
        good = [x for x in aucs if not np.isnan(x)]
        m = mean(good) if good else float("nan")
        s = stdev(good) if len(good) > 1 else 0.0
        table[name] = {"n": n, "n_pos": npos, "per_run": aucs, "mean": m, "sd": s}
        cells = "  ".join(f"{x:>7.3f}" for x in aucs)
        print(f"{name:<32} {n:>4} {npos:>4}   {cells}   {m:.3f} +/- {s:.3f}")

    base = table["full cohort (current headline)"]["mean"]
    print()
    for name, v in table.items():
        if name.startswith("full"):
            continue
        print(f"[subgroups] {name:<28} {v['mean'] - base:+.3f} vs the headline")

    n_excl = len(runs) and (table["full cohort (current headline)"]["n"]
                            - table["adenocarcinoma only"]["n"])
    print(f"\n[subgroups] dropping non-adenocarcinoma removes {n_excl} patients, "
          f"all EGFR-wildtype -- {100*n_excl/max(table['full cohort (current headline)']['n'],1):.0f}% "
          "of the cohort was guaranteed-negative on histology alone.")

    json.dump({"gene": a.gene, "runs": runs, "subsets": table}, open(a.out, "w"), indent=2)
    print(f"[subgroups] wrote {a.out}")


if __name__ == "__main__":
    main()
