"""Which pooling of frozen patch tokens carries the EGFR signal?

Motivation, from a measurement rather than a hunch. The frozen-feature baseline
mean-pools every patch token in the crop — but `context_factor: 2.5` was chosen
deliberately so the tumour occupies only **3.9%** of that crop. So roughly 96%
of the pooled vector is chest wall, vessel and aerated lung, and whatever
signal the lesion carries is diluted ~25x before the classifier sees it.

That is a specific, testable explanation for why THEIA (0.627 ± 0.041) does not
beat a linear probe on its own frozen features (0.617 ± 0.052): both pool the
same diluted representation.

Every variant here is scored through **exactly THEIA's nested folds**, per seed,
so a win is comparable to the numbers already reported and not to a friendlier
protocol. Tokens are extracted once and cached; the variants are then cheap, so
this sweeps in minutes where retraining sweeps in days.

The mask used is the ROI already stored in each .npz, downsampled to the patch
grid by the same `roi_to_grid` the grounding loss uses — so "tumour patch" means
here exactly what it means during training.

Run: python -m theia.analysis.pooling_sweep
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean, stdev

import numpy as np
import torch


def extract_grids(cfg, rows: list[dict], device) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient patch tokens and the tumour mask on the same grid.

    Returns (tokens [P, N, D], grid_mask [P, N]). Slice aggregation stays the
    encoder's own masked mean, so this isolates the SPATIAL pooling question.
    """
    from theia.engine.losses import roi_to_grid
    from theia.models.backbone import VisionEncoder

    enc = VisionEncoder(cfg.model.vision_encoder, cfg.model.vision_dim,
                        freeze=True).to(device).eval()
    toks, masks = [], []
    for i, r in enumerate(rows):
        with np.load(r["npz"]) as b:
            img = torch.from_numpy(b["images"]).float()
            roi = torch.from_numpy(b["roi"]).float()
        x = img.unsqueeze(1).unsqueeze(0)                      # [1,S,1,H,W]
        with torch.no_grad():
            t, grid = enc.encode(x.to(device))                 # [1,N,D]
        h, w = grid
        g = roi_to_grid(roi.unsqueeze(1).unsqueeze(0), h, w, "cpu")[0]   # [h,w]
        toks.append(t[0].float().cpu().numpy())
        masks.append(g.reshape(-1).numpy())
        if (i + 1) % 40 == 0:
            print(f"[pool]   {i+1}/{len(rows)}", flush=True)
    return np.stack(toks), np.stack(masks)


def _dilate(m: np.ndarray, h: int, w: int, iters: int = 1) -> np.ndarray:
    from scipy import ndimage

    g = m.reshape(-1, h, w) > 0.5
    out = np.stack([ndimage.binary_dilation(x, iterations=iters) for x in g])
    return out.reshape(len(m), -1).astype(np.float32)


def _control_regions(msk: np.ndarray, ring: np.ndarray, h: int, w: int,
                     seed: int = 1337) -> tuple[np.ndarray, np.ndarray]:
    """Size-matched shapes and rings at a random INTERIOR location.

    The control has to differ from the real region in location and nothing else.
    np.roll wraps a shape but binary_dilation clips at the array border, so an
    unconstrained shift makes control rings systematically smaller -- measured at
    -3.7%, 22 of 158 -- and a smaller control carries less information, inflating
    the very gain it exists to rule out. Shifts are constrained so the dilated
    shape stays inside the grid, giving an exact size match on every patient with
    a ring.
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    shifted, randring = [], []
    for m1, real_ring in zip(msk, ring):
        g = m1.reshape(h, w) > 0.5
        want = int(real_ring.sum())
        ys, xs = np.nonzero(g)
        if ys.size == 0:
            shifted.append(np.zeros(h * w, np.float32))
            randring.append(np.zeros(h * w, np.float32))
            continue
        best, best_gap = None, None
        for _ in range(60):
            lo_y, hi_y = -(ys.min()) + 1, max(h - ys.max() - 1, -(ys.min()) + 2)
            lo_x, hi_x = -(xs.min()) + 1, max(w - xs.max() - 1, -(xs.min()) + 2)
            sh = np.roll(np.roll(g, int(rng.integers(lo_y, hi_y)), 0),
                         int(rng.integers(lo_x, hi_x)), 1)
            r = ndimage.binary_dilation(sh, iterations=1) & ~sh
            gap = abs(int(r.sum()) - want)
            if best_gap is None or gap < best_gap:
                best, best_gap = (sh, r), gap
            if gap == 0:
                break
        sh, r = best
        shifted.append(sh.reshape(-1).astype(np.float32))
        randring.append(r.reshape(-1).astype(np.float32))
    return np.stack(shifted), np.stack(randring)


def build_variants(tok: np.ndarray, msk: np.ndarray, h: int, w: int) -> dict:
    """Feature matrices, one per pooling strategy."""
    def pool(weights: np.ndarray, how: str = "mean") -> np.ndarray:
        out = []
        for t, wt in zip(tok, weights):
            sel = t[wt > 0.5]
            if sel.size == 0:                 # no tumour patch at grid resolution
                sel = t                       # fall back to the whole crop
            out.append(sel.max(0) if how == "max" else sel.mean(0))
        return np.stack(out)

    ones = np.ones_like(msk)
    dil = _dilate(msk, h, w, 1)
    ring = np.clip(dil - msk, 0, 1)           # peritumoral shell only

    whole = pool(ones)
    tumour = pool(msk)
    tumour_max = pool(msk, "max")
    peri = pool(ring)

    # CONTROLS. A peritumoral win is exactly the kind of result that is a fluke
    # at n=153, so it has to beat regions that are matched on everything except
    # location before it means anything.
    shifted, randring = _control_regions(msk, ring, h, w)

    peri2 = pool(np.clip(_dilate(msk, h, w, 2) - msk, 0, 1))

    return {
        "whole_mean (current baseline)": whole,
        "tumour_mean": tumour,
        "tumour_max": tumour_max,
        "peritumoral_mean": peri,
        "tumour_mean + whole_mean": np.hstack([tumour, whole]),
        "tumour_mean + peritumoral": np.hstack([tumour, peri]),
        "tumour_mean + tumour_max": np.hstack([tumour, tumour_max]),
        "peritumoral (2-ring)": peri2,
        "CONTROL shifted-tumour_mean": pool(shifted),
        "CONTROL shifted-ring_mean": pool(randring),
    }


def score(X: np.ndarray, rows: list[dict], cfg, gene: str, seeds: list[int]) -> tuple:
    """Pooled OOF AUC per seed, through THEIA's own nested folds.

    Also returns the per-patient seed-averaged rank, which is what the paired
    test needs -- an AUC alone cannot support a bootstrap over patients.
    """
    from theia.analysis.baselines import _ranked, oof_predictions
    from theia.engine.evaluate import pooled_metrics

    aucs, ranks = [], []
    for s in seeds:
        cfg["seed"] = s
        oof = oof_predictions(X, rows, gene, cfg)
        aucs.append(pooled_metrics(oof, [gene], cfg.eval.bootstrap_n)[f"{gene.lower()}_auc"])
        ranks.append(_ranked(oof, f"{gene.lower()}_prob"))
    ids = sorted(set.intersection(*[set(d) for d in ranks]))
    avg = {i: float(np.mean([d[i] for d in ranks])) for i in ids}
    return mean(aucs), (stdev(aucs) if len(aucs) > 1 else 0.0), aucs, avg


def holm_paired(arm_ranks: dict[str, dict], truth: dict[str, int], baseline: str,
                n_boot: int = 5000, seed: int = 1337) -> dict:
    """Each arm vs the baseline arm: paired bootstrap over PATIENTS, Holm-corrected.

    The sweep scores 10 variants on the same outer folds and reports the best, so
    two corrections are needed and they are different corrections.

    Multiplicity: 10 comparisons against one baseline, so p-values are adjusted by
    Holm-Bonferroni. Holm rather than Bonferroni because the arms are strongly
    correlated -- they are poolings of the same tokens -- and Bonferroni would be
    needlessly conservative.

    Unit of resampling: the earlier version of this sweep reported "10/10 seeds"
    as though seeds were replicates. They are not. Ten seeds are ten re-splits of
    ONE 153-patient cohort, so they measure split variance, not sampling variance,
    and a sign test over them answers a question nobody asked. Patients are the
    unit that generalises, so the bootstrap resamples patients and the seed spread
    is reported separately as variance.
    """
    from sklearn.metrics import roc_auc_score

    ids = sorted(set(truth).intersection(*[set(v) for v in arm_ranks.values()]))
    y = np.array([truth[i] for i in ids])
    base = np.array([arm_ranks[baseline][i] for i in ids])
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]
    idx = [i for i in idx if len(set(y[i].tolist())) > 1]

    raw = {}
    for name, r in arm_ranks.items():
        if name == baseline:
            continue
        a = np.array([r[i] for i in ids])
        obs = roc_auc_score(y, a) - roc_auc_score(y, base)
        d = np.array([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], base[i])
                      for i in idx])
        raw[name] = {"delta": float(obs),
                     "ci_lo": float(np.percentile(d, 2.5)),
                     "ci_hi": float(np.percentile(d, 97.5)),
                     "p_raw": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

    # Holm-Bonferroni, step-down.
    order = sorted(raw, key=lambda k: raw[k]["p_raw"])
    m = len(order)
    running = 0.0
    for rank, name in enumerate(order):
        adj = min(1.0, raw[name]["p_raw"] * (m - rank))
        running = max(running, adj)          # enforce monotonicity
        raw[name]["p_holm"] = running
    return raw


def main() -> None:
    from theia.analysis.diagnostics import load_rows
    from theia.config import load_config
    from theia.runtime import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gene", default=None)
    ap.add_argument("--cache", default="results/_token_cache.npz")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated. More than the 3 archived seeds: with\nonly 3 the sd is too wide to rank variants, and features are cached so\nextra seeds are nearly free.")
    ap.add_argument("--out", default="results/pooling_sweep.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    gene = a.gene or cfg.data.get("headline_genes", ["EGFR"])[0]
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
    else:
        seeds = sorted({(json.load(open(p)).get("config") or {}).get("seed", cfg.seed)
                        for p in glob.glob("results/ms-s*.json")}) or [cfg.seed]
    print(f"[pool] {len(rows)} patients, gene={gene}, seeds={seeds}")

    if os.path.exists(a.cache):
        z = np.load(a.cache)
        tok, msk = z["tok"], z["msk"]
        print(f"[pool] loaded cached tokens {tok.shape}")
    else:
        print("[pool] extracting frozen tokens (once)...", flush=True)
        tok, msk = extract_grids(cfg, rows, resolve_device("auto"))
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        np.savez_compressed(a.cache, tok=tok, msk=msk)
        print(f"[pool] cached {tok.shape} -> {a.cache}")

    n = tok.shape[1]
    h = w = int(round(n ** 0.5))
    frac = float(msk.mean())
    print(f"[pool] grid {h}x{w}, tumour occupies {100*frac:.1f}% of patches\n")

    variants = build_variants(tok, msk, h, w)
    results, arm_ranks = {}, {}
    base = None
    for name, X in variants.items():
        m, s, per, avg = score(X, rows, cfg, gene, seeds)
        results[name] = {"mean": m, "sd": s, "per_seed": per, "dim": int(X.shape[1])}
        arm_ranks[name] = avg
        if base is None:
            base = m
        print(f"[pool] {name:32s} {m:.3f} +/- {s:.3f}   ({m-base:+.3f} vs baseline, "
              f"d={X.shape[1]})")

    baseline_name = next(iter(variants))
    truth = {r["patient_id"]: int(r.get("labels", {}).get(gene.upper(), -1)) for r in rows}
    truth = {k: v for k, v in truth.items() if v in (0, 1)}
    paired = holm_paired(arm_ranks, truth, baseline_name)

    print(f"\n[pool] paired bootstrap over PATIENTS vs '{baseline_name}', "
          f"Holm-corrected across {len(paired)} arms:")
    print(f"  {'arm':<32} {'delta':>7}  {'95% CI':>18} {'p_raw':>7} {'p_holm':>7}")
    for name in sorted(paired, key=lambda k: -paired[k]["delta"]):
        v = paired[name]
        star = " *" if v["p_holm"] < 0.05 else ""
        print(f"  {name:<32} {v['delta']:>+7.3f}  "
              f"[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] {v['p_raw']:>7.3f} "
              f"{v['p_holm']:>7.3f}{star}")
    print("\n[pool] seed spread is reported as variance, not replication: ten seeds are "
          "ten\n       re-splits of one 153-patient cohort, so they measure split "
          "variance. Patients\n       are the unit that generalises, so they are what "
          "the bootstrap resamples.")

    json.dump({"gene": gene, "seeds": seeds, "tumour_patch_frac": frac,
               "variants": results, "baseline_arm": baseline_name,
               "paired_vs_baseline_holm": paired}, open(a.out, "w"), indent=2)
    print(f"\n[pool] wrote {a.out}")
    best = max(results.items(), key=lambda kv: kv[1]["mean"])
    print(f"[pool] best: {best[0]} at {best[1]['mean']:.3f} "
          f"({best[1]['mean']-base:+.3f} over the current pooling)")


if __name__ == "__main__":
    main()
