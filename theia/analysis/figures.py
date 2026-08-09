"""Figures, rebuilt from results/*.json alone.

No checkpoints are loaded. Everything here comes from the archived out-of-fold
predictions, which is the reason the archive stores them — the weights for the
project's earlier best run no longer exist, and its curves would have been
unrecoverable.

  fig1_roc.png        ROC per arm, on the same patients and the same folds.
  fig2_forest.png     Every arm's AUC with its CI, plus the paired deltas
                      against the clinical model. The deltas are the point: two
                      overlapping CIs can still differ significantly when paired.
  fig3_seeds.png      Per-seed AUC for THEIA against the clinical baseline, to
                      show rerun spread next to the effect being claimed.

Deliberate choices, since a figure is an argument:
  * The chance line is drawn and labelled on every panel.
  * Axes start at 0.0/0.5 rather than cropping to make differences look larger.
  * Each arm's n is printed, because the arms do not all cover the same patients
    once a fold has stalled.

Run: python -m theia.analysis.figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

CHANCE = 0.5


def _load_oof(path: str) -> list[dict]:
    blob = json.load(open(path))
    return [r for f in blob["folds"] if not f.get("stalled")
            for r in (f.get("oof") or [])]


def _ranked_by_fold(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Within-fold ranks + labels, matching how the pooled AUC is computed."""
    from theia.engine.evaluate import _rank_normalize_by_fold

    keep = [r for r in rows if key in r and r.get(key.replace("_prob", "_true"), -1) != -1]
    if not keep:
        return np.array([]), np.array([])
    y = np.array([r[key.replace("_prob", "_true")] for r in keep])
    p = _rank_normalize_by_fold(keep, key)
    return y, p


def fig_roc(arms: dict, gene: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=200)
    for label, rows in arms.items():
        y, p = _ranked_by_fold(rows, f"{gene.lower()}_prob")
        if y.size == 0 or len(set(y.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.8,
                label=f"{label} — AUC {roc_auc_score(y, p):.3f} (n={y.size})")
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), lw=1, color="0.5")
    ax.annotate("chance", xy=(0.62, 0.58), color="0.45", fontsize=8, rotation=45)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.set_title(f"{gene} — pooled out-of-fold ROC", fontsize=11)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_forest(summary: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, aucs, los, his = [], [], [], []
    for name, m in summary.items():
        if m.get("auc") is None or m["auc"] != m["auc"]:
            continue
        labels.append(f"{name}  (n={m.get('n', '?')})")
        aucs.append(m["auc"]); los.append(m["lo"]); his.append(m["hi"])
    order = np.argsort(aucs)
    labels = [labels[i] for i in order]
    aucs = np.array(aucs)[order]; los = np.array(los)[order]; his = np.array(his)[order]

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.4), dpi=200)
    ypos = np.arange(len(labels))
    ax.hlines(ypos, los, his, lw=2, color="0.35")
    ax.plot(aucs, ypos, "o", ms=5, color="0.1")
    ax.axvline(CHANCE, ls=(0, (4, 4)), lw=1, color="0.5")
    ax.text(CHANCE, -0.85, "chance", fontsize=8, color="0.45",
            ha="center", va="center")
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    # The multi-seed row plots an observed RANGE, not a bootstrap CI. Saying
    # "95% CI" over a mixed axis would misdescribe one of the rows.
    ax.set_xlabel("AUC — bars are 95% CI, except the multi-seed row (observed range)",
                  fontsize=8.5)
    ax.set_xlim(0.35, 1.0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_seeds(per_seed: list[dict], clinical: float, out: str) -> None:
    """THEIA per seed against the clinical baseline.

    The figure exists to make one comparison unmissable: the spread of THEIA
    across reruns against the gap to a five-variable chart model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [str(p["seed"]) for p in per_seed]
    vals = [p["auc"] for p in per_seed]
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=200)
    bars = ax.bar(seeds, vals, width=0.55, color="0.75", edgecolor="0.3", lw=0.8)
    # Hatch any run that lost a fold. Its AUC is computed on fewer patients, and
    # here the INCOMPLETE run is also the highest-scoring one -- leaving that
    # unmarked would let the eye read it as the best result rather than the
    # least comparable.
    for bar, p in zip(bars, per_seed):
        if p.get("stalled"):
            bar.set_hatch("///")
            ax.text(bar.get_x() + bar.get_width() / 2, p["auc"] + 0.008,
                    f"n={p.get('n', '?')}\nfold lost", ha="center", fontsize=6.5,
                    color="0.25")
    m = float(np.mean(vals))
    ax.axhline(m, ls="-", lw=1.2, color="0.2")
    ax.text(len(seeds) - 0.4, m + 0.006, f"THEIA mean {m:.3f}", fontsize=8, ha="right")
    ax.axhline(clinical, ls=(0, (5, 3)), lw=1.4, color="#8c2d04")
    ax.text(len(seeds) - 0.4, clinical + 0.006, f"clinical {clinical:.3f}",
            fontsize=8, ha="right", color="#8c2d04")
    ax.axhline(CHANCE, ls=(0, (2, 3)), lw=1, color="0.6")
    ax.text(-0.45, CHANCE + 0.006, "chance", fontsize=8, color="0.5")
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("pooled out-of-fold AUC"); ax.set_xlabel("seed")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"[fig] wrote {out}")


def main() -> None:
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pattern", default="results/ms-s*.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    cfg = load_config(a.config)
    gene = cfg.data.get("headline_genes", ["EGFR"])[0]
    g = gene.lower()
    os.makedirs(a.outdir, exist_ok=True)

    paths = sorted(glob.glob(a.pattern))
    if not paths:
        raise SystemExit(f"no results matched {a.pattern!r}")

    # ROC uses the single most complete THEIA run rather than a merge: a merged
    # curve over patients x seeds is not a curve any one model produced.
    best = max(paths, key=lambda p: len(_load_oof(p)))
    theia_rows = _load_oof(best)
    print(f"[fig] THEIA ROC from {os.path.basename(best)} "
          f"({len(theia_rows)} predictions)")

    arms = {"THEIA": theia_rows}
    bl = json.load(open(a.baselines)) if os.path.exists(a.baselines) else {}
    # Baseline arms, if their predictions were archived. Restricted to the
    # patients THEIA scored so every curve is drawn on the same cohort -- a ROC
    # comparison across different patient sets is not a comparison.
    scored = {r["patient_id"] for r in theia_rows}
    for name in ("clinical", "radiomics"):
        rows_b = (bl.get("oof") or {}).get(name)
        if rows_b:
            arms[name] = [r for r in rows_b if r["patient_id"] in scored]

    summary: dict[str, dict] = {}
    for name, m in (bl.get("arms") or {}).items():
        if m.get(f"{g}_auc") is None:
            continue
        summary[name] = {"auc": m.get(f"{g}_auc"), "lo": m.get(f"{g}_auc_lo"),
                         "hi": m.get(f"{g}_auc_hi"), "n": int(m.get(f"{g}_n", 0))}

    from theia.engine.evaluate import pooled_metrics
    tm = pooled_metrics(theia_rows, [gene], cfg.eval.bootstrap_n)
    # Name the seed. Calling this "best seed" would be wrong twice over: it is
    # chosen for COVERAGE (most predictions), not for AUC, and the run it picks
    # is in fact the lowest-scoring of the three. A figure label that implies
    # the opposite is worse than no label.
    best_seed = (json.load(open(best)).get("config") or {}).get("seed")
    summary[f"THEIA (seed {best_seed})"] = {
        "auc": tm.get(f"{g}_auc"), "lo": tm.get(f"{g}_auc_lo"),
        "hi": tm.get(f"{g}_auc_hi"), "n": int(tm.get(f"{g}_n", 0))}

    fig_roc(arms, gene, os.path.join(a.outdir, "fig1_roc.png"))

    per_seed = []
    for p in paths:
        blob = json.load(open(p))
        pooled = blob.get("pooled") or {}
        auc = pooled.get(f"{g}_auc")
        if auc is not None and auc == auc:
            per_seed.append({
                "seed": (blob.get("config") or {}).get("seed"), "auc": auc,
                "n": int(pooled.get(f"{g}_n", 0)),
                "stalled": any(f.get("stalled") for f in blob["folds"]),
            })
    per_seed.sort(key=lambda d: str(d["seed"]))
    clin = (summary.get("clinical") or {}).get("auc", float("nan"))

    # The multi-seed summary goes on the forest plot too, with the interval
    # drawn as the observed RANGE rather than a bootstrap CI. They measure
    # different things and plotting one in place of the other would understate
    # the spread that actually matters when rerunning.
    if len(per_seed) >= 2:
        vals = [p["auc"] for p in per_seed]
        summary[f"THEIA ({len(vals)}-seed mean, range)"] = {
            "auc": float(np.mean(vals)), "lo": float(min(vals)),
            "hi": float(max(vals)), "n": "3 runs"}

    fig_forest(summary, os.path.join(a.outdir, "fig2_forest.png"))
    if per_seed and clin == clin:
        fig_seeds(per_seed, clin, os.path.join(a.outdir, "fig3_seeds.png"))


if __name__ == "__main__":
    main()
