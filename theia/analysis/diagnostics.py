"""Is the null result about the model, or about N?

The full run returned a pooled out-of-fold EGFR AUC of 0.426 (95% CI
[0.307, 0.546]) over 117 patients with 23 positives. That is chance. Before
committing months to more data or a bigger architecture, four cheap diagnostics
separate the possible causes.

The design point: a permutation test needs hundreds of refits, which is
impossible at ~3 h per end-to-end run. So features are extracted ONCE from a
frozen encoder, and every diagnostic runs on the cached matrix in seconds. That
is also a fairer test of "is there signal" than fine-tuning 88M parameters on 76
patients, which mostly measures how fast the model memorises.

  1. linear probe      frozen pretrained features + L2 logistic regression.
                       Right capacity for this N. If this beats the fine-tuned
                       model, the problem is capacity, not signal.
  2. permutation test   shuffle the labels N times, rebuild the null distribution
                       of pooled AUC, report an empirical p-value. This is the
                       definitive "is anything being extracted" test.
  3. radiomics baseline handcrafted first-order + shape + GLCM texture features,
                       same folds, same estimator. What the field would do.
                       If radiomics finds signal and the network does not, the
                       network is the problem.
  4. learning curve     AUC against training-set size. A rising curve at n=94
                       means more data helps and roughly predicts how much. A
                       flat curve near 0.5 means it will not.

Run: python -m theia.analysis.diagnostics
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def load_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


@torch.no_grad()
def deep_features(cfg, rows: list[dict], device) -> np.ndarray:
    """Mean-pooled patch tokens from the FROZEN pretrained vision encoder."""
    from theia.models.backbone import VisionEncoder

    enc = VisionEncoder(cfg.model.vision_encoder, cfg.model.vision_dim, freeze=True)
    enc = enc.to(device).eval()
    feats = []
    for r in rows:
        with np.load(r["npz"]) as b:
            img = torch.from_numpy(b["images"]).float().unsqueeze(1).unsqueeze(0)
        tok, _ = enc.encode(img.to(device))           # [1, N, D]
        feats.append(tok.mean(dim=1).squeeze(0).float().cpu().numpy())
    return np.stack(feats)


def radiomic_features(rows: list[dict]) -> np.ndarray:
    """First-order + shape + GLCM texture over the segmented region.

    Not PyRadiomics (which does not install cleanly on Python 3.13), but the same
    families of descriptor: intensity distribution, lesion geometry, and
    grey-level co-occurrence texture. Enough to answer "would a conventional
    radiomics pipeline find signal here".
    """
    from scipy import stats
    from skimage.feature import graycomatrix, graycoprops

    out = []
    for r in rows:
        with np.load(r["npz"]) as b:
            img, roi = b["images"], b["roi"]
        m = roi > 0.5
        vals = img[m] if m.any() else img.ravel()

        f = [
            vals.mean(), vals.std(), np.median(vals),
            np.percentile(vals, 10), np.percentile(vals, 90),
            float(stats.skew(vals)), float(stats.kurtosis(vals)),
            float(stats.entropy(np.histogram(vals, bins=32, range=(0, 1))[0] + 1e-9)),
            float(m.mean()),                                  # lesion volume fraction
        ]
        # geometry on the largest-area slice
        areas = m.reshape(m.shape[0], -1).sum(1)
        k = int(areas.argmax())
        sl = m[k]
        if sl.any():
            ys, xs = np.where(sl)
            h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
            area = sl.sum()
            f += [float(h), float(w), float(min(h, w) / max(h, w)),
                  float(area), float(4 * np.pi * area / max((2 * (h + w)) ** 2, 1))]
        else:
            f += [0.0] * 5
        # GLCM texture on the same slice, lesion bounding box
        patch = (img[k] * 255).astype(np.uint8)
        g = graycomatrix(patch, distances=[1, 3], angles=[0, np.pi / 2],
                         levels=256, symmetric=True, normed=True)
        for prop in ("contrast", "homogeneity", "energy", "correlation"):
            f.append(float(graycoprops(g, prop).mean()))
        out.append(f)
    return np.nan_to_num(np.array(out, dtype=np.float64))


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def pooled_oof_auc(X: np.ndarray, y: np.ndarray, seed: int = 1337,
                   n_folds: int = 5, C: float = 1.0) -> float:
    """Pooled out-of-fold AUC from an L2 logistic regression.

    Predictions are rank-normalised within fold before pooling, exactly as
    theia.engine.evaluate.pooled_metrics does, because each fold is a different
    fitted model and their score scales are not comparable.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p.argsort().argsort() / max(len(p) - 1, 1)   # within-fold ranks
    return float(roc_auc_score(y, oof))


def permutation_test(X: np.ndarray, y: np.ndarray, n_perm: int = 200,
                     seed: int = 0) -> dict:
    """Empirical null for pooled AUC under shuffled labels."""
    rng = np.random.default_rng(seed)
    observed = pooled_oof_auc(X, y)
    null = np.array([pooled_oof_auc(X, rng.permutation(y), seed=1337 + i)
                     for i in range(n_perm)])
    # two-sided on |AUC - 0.5|, +1 correction so p is never exactly 0
    p = (np.sum(np.abs(null - 0.5) >= abs(observed - 0.5)) + 1) / (n_perm + 1)
    return dict(observed=observed, null_mean=float(null.mean()),
                null_sd=float(null.std()),
                null_p95=float(np.percentile(np.abs(null - 0.5), 95) + 0.5),
                p_value=float(p), n_perm=n_perm)


def learning_curve(X: np.ndarray, y: np.ndarray, fractions=(0.25, 0.5, 0.75, 1.0),
                   repeats: int = 12, seed: int = 0) -> list[dict]:
    """Pooled AUC against training-set size, to see whether more data would help."""
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        scores = []
        for r in range(repeats):
            idx = rng.permutation(len(y))[: max(int(round(frac * len(y))), 12)]
            if len(set(y[idx].tolist())) < 2:
                continue
            try:
                scores.append(pooled_oof_auc(X[idx], y[idx], seed=1337 + r))
            except ValueError:
                continue
        if scores:
            rows.append(dict(frac=frac, n=int(round(frac * len(y))),
                             auc_mean=float(np.mean(scores)),
                             auc_sd=float(np.std(scores)), repeats=len(scores)))
    return rows


def main() -> None:
    from theia.config import load_config
    from theia.runtime import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--n_perm", type=int, default=200)
    a = ap.parse_args()

    cfg = load_config(a.config)
    rows = load_rows(os.path.join(cfg.paths.processed_dir, "rows.jsonl"))
    gene = a.gene
    keep = [r for r in rows if int(r["labels"].get(gene, -1)) != -1]
    y = np.array([int(r["labels"][gene]) for r in keep])
    print(f"[diag] {gene}: {len(y)} labelled, {int(y.sum())} positive", flush=True)

    device = resolve_device("auto")
    print(f"[diag] extracting frozen-encoder features on {device}...", flush=True)
    Xd = deep_features(cfg, keep, device)
    print(f"[diag] deep features {Xd.shape}", flush=True)
    Xr = radiomic_features(keep)
    print(f"[diag] radiomic features {Xr.shape}", flush=True)

    results = {"gene": gene, "n": int(len(y)), "n_pos": int(y.sum())}
    for name, X in (("deep_frozen", Xd), ("radiomics", Xr),
                    ("deep+radiomics", np.hstack([Xd, Xr]))):
        auc = pooled_oof_auc(X, y)
        results[f"{name}_auc"] = auc
        print(f"[diag] {name:16s} pooled OOF AUC = {auc:.3f}", flush=True)

    print(f"[diag] permutation test ({a.n_perm} shuffles) on deep features...", flush=True)
    results["permutation_deep"] = permutation_test(Xd, y, a.n_perm)
    print(f"[diag] permutation test on radiomics...", flush=True)
    results["permutation_radiomics"] = permutation_test(Xr, y, a.n_perm)

    print("[diag] learning curves...", flush=True)
    results["learning_curve_deep"] = learning_curve(Xd, y)
    results["learning_curve_radiomics"] = learning_curve(Xr, y)

    os.makedirs(cfg.paths.runs_dir, exist_ok=True)
    out = os.path.join(cfg.paths.runs_dir, f"diagnostics_{gene}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[diag] -> {out}", flush=True)


if __name__ == "__main__":
    main()
