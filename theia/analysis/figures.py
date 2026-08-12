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
  fig4_grounding.png  Each fold's attention mass joined to its OWN shuffled
                      baseline. Paired, because chance depends on how much of
                      the crop the tumour fills in that fold.
  fig5_overlays.png   Attention over the CT with the tumour outlined. Needs a
                      checkpoint (--overlay_ckpt); everything else does not.

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


def fig_grounding(rows: list[dict], out: str) -> None:
    """Per-fold attention mass against each fold's OWN shuffled baseline.

    Paired, not two independent distributions: the chance level depends on how
    much of the crop the tumour occupies in that fold, so a single global
    baseline would be wrong. Drawing the pairing is the whole point -- the claim
    is "every fold beats its own chance level", and a bar chart of means cannot
    show that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array([r["grounding_mass"] for r in rows])
    s = np.array([r["grounding_mass_shuffled"] for r in rows])
    order = np.argsort(m)
    m, s = m[order], s[order]
    x = np.arange(len(m))

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    for i in x:
        ax.plot([i, i], [s[i], m[i]], color="0.75", lw=1.2, zorder=1)
    ax.scatter(x, s, s=22, color="#8c8c8c", zorder=2, label="shuffled baseline")
    ax.scatter(x, m, s=26, color="#08519c", zorder=3, label="model")
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{i+1}" for i in x], fontsize=7)
    ax.set_xlabel(f"held-out fold (n={len(m)}, sorted by model)", fontsize=9)
    ax.set_ylabel("attention mass inside tumour", fontsize=9)
    ax.set_ylim(0, max(m.max() * 1.15, 0.1))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"mass {m.mean():.3f} vs {s.mean():.3f} chance "
                 f"({m.mean()/s.mean():.1f}x), {int((m>s).sum())}/{len(m)} folds",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_external_grounding(blob: dict, out: str) -> None:
    """The primary endpoint: grounding on a cohort the model never saw.

    Drawn with all three controls in every panel, because the trained bars alone
    would overstate the result. The centre prior is the one that matters: these
    crops are lesion-centred, so "look at the middle" is a genuinely strong
    baseline on attention MASS and a weak one on pointing and area-matched IoU.
    Showing all three metrics side by side is what makes that visible -- the
    trained model's margin over the prior is small on the left panel and large on
    the other two, and a reader should be able to see that rather than be told it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = blob["rows"]
    cp, rnd = blob["control_centre_prior"], blob["control_random_init"]
    live = [r for r in rows if r["grounding_pointing"] > 0.05]
    keys = ["grounding_mass", "grounding_pointing", "grounding_iou"]
    titles = ["attention mass in ROI", "pointing game", "area-matched IoU"]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=200)
    for ax, k, title in zip(axes, keys, titles):
        vals = [np.mean([r[k] for r in rows]), np.mean([r[k] for r in live]),
                cp[k], np.mean([r[k + "_shuffled"] for r in rows]), rnd[k]]
        labels = ["trained\n(all 30)", "trained\n(25 live)", "centre\nprior",
                  "spatial\nshuffle", "random\ninit"]
        colours = ["#08519c", "#3182bd", "#e6550d", "#8c8c8c", "#bdbdbd"]
        ax.bar(range(5), vals, color=colours, width=0.68)
        # Per-fold spread on the trained bars only; the controls are single
        # numbers computed once over the whole cohort, so an error bar there
        # would imply a variability that was never measured.
        spread = np.std([r[k] for r in rows])
        ax.errorbar(0, vals[0], yerr=spread, color="0.2", capsize=3, lw=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("score", fontsize=9)
    fig.suptitle(
        f"External validation: {blob['n_external']} held-out NSCLC-Radiomics patients "
        f"(Maastro, NL) — mass lift {blob['mean_mass_lift']:+.3f}, "
        f"beats shuffle {blob['folds_beating_shuffle']}/{blob['n_folds']}",
        fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out); plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_overlays(ckpt: str, cfg, out: str, n: int = 6) -> None:
    """Attention over the CT for held-out patients, with the tumour outlined.

    Qualitative panels are easy to cherry-pick, so these are the FIRST n patients
    of the fold in index order -- not the best-scoring ones -- and the selection
    rule is stated in the caption line printed below. The tumour contour is drawn
    from the ROI so a reader can see where the attention should be, rather than
    being asked to take the overlay on trust.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from theia.data.dataset import RadiogenomicsDataset, collate, nested_kfold_indices
    from theia.models.theia_model import Theia
    from theia.runtime import resolve_device

    device = resolve_device("auto")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = Theia(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)

    rp = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    seed = int((state.get("cfg") or {}).get("seed", cfg.seed))
    fold = int(state.get("fold", 0))
    splits = list(nested_kfold_indices(rp, cfg.split.stratify_on, cfg.split.n_folds,
                                       seed, float(cfg.split.inner_val_frac)))
    held = list(splits[fold][2])
    ds = RadiogenomicsDataset(rp, cfg.data.target_genes)

    picks, used = [], 0
    for i in held:
        item = ds[i]
        if float(item["roi"].sum()) <= 0:      # AIM-tier: no mask to outline
            continue
        picks.append((i, item))
        used += 1
        if used >= n:
            break

    fig, axes = plt.subplots(2, (len(picks) + 1) // 2, figsize=(2.1 * len(picks), 4.6),
                             dpi=200)
    for ax, (i, item) in zip(np.array(axes).ravel(), picks):
        with torch.no_grad():
            out_d = model(collate([item], genes=model.genes), device)
        a = out_d["attn_maps"][0].amax(0).float().cpu().numpy()
        a = (a - a.min()) / (np.ptp(a) + 1e-6)
        img = item["images"].numpy()
        roi = item["roi"].numpy()
        k = int(roi.reshape(roi.shape[0], -1).sum(1).argmax())
        mid, msk = img[k, 0], roi[k, 0]
        ky = max(mid.shape[0] // a.shape[0], 1)
        kx = max(mid.shape[1] // a.shape[1], 1)
        big = np.kron(a, np.ones((ky, kx)))[: mid.shape[0], : mid.shape[1]]
        ax.imshow(mid, cmap="gray")
        ax.imshow(big, cmap="inferno", alpha=0.42)
        ax.contour(msk, levels=[0.5], colors="#39d353", linewidths=0.9)
        ax.set_axis_off()
    for ax in np.array(axes).ravel()[len(picks):]:
        ax.set_axis_off()
    fig.suptitle("attention (hot) with tumour outlined (green) — first "
                 f"{len(picks)} masked patients of held-out fold {fold}, not "
                 "selected on score", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"[fig] wrote {out}")


def main() -> None:
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pattern", default="results/ms-s*.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--overlay_ckpt", default=None,
                    help="checkpoint for qualitative attention panels")
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

    gfolds = [f["test"] for p in paths for f in json.load(open(p))["folds"]
              if not f.get("stalled") and f["test"].get("grounding_mass") is not None]
    if gfolds:
        fig_grounding(gfolds, os.path.join(a.outdir, "fig4_grounding.png"))
    if a.overlay_ckpt:
        fig_overlays(a.overlay_ckpt, cfg, os.path.join(a.outdir, "fig5_overlays.png"))

    ext = "results/external_grounding.json"
    if os.path.exists(ext):
        fig_external_grounding(json.load(open(ext)),
                               os.path.join(a.outdir, "fig6_external_grounding.png"))
    else:
        print(f"[fig] skipping fig6: {ext} not found "
              "(run theia.analysis.external_grounding)")


if __name__ == "__main__":
    main()
