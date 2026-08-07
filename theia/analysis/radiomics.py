"""A radiomics baseline strong enough for the comparison to be fair.

The 18-feature extractor in `diagnostics.py` was built to answer "is there any
signal at all". Used as the paper's comparator it scores EGFR at 0.662, while
published CT-radiomics work on the same task reports 0.79-0.85. Losing to a
weak baseline proves nothing, so this rebuilds the arm properly.

What the previous version left on the table, in rough order of expected value:

  HU is recoverable, and was being thrown away.
      The .npz holds crops already windowed to [0,1]. That is a linear map of
      Hounsfield units, so it inverts exactly: HU = v*width + (center -
      width/2). Absolute density is not a nuisance here, it is the signal --
      ground-glass opacity is one of the best-established imaging correlates of
      EGFR mutation, and it is defined by an HU range. Working in [0,1] made
      that inexpressible.

  Peritumoral tissue.
      Texture in the rim outside the lesion carries mutation signal in the
      literature. The old extractor only ever looked inside the mask.

  Texture beyond one slice and one scale.
      GLCM on a single slice at two distances, versus multi-distance
      multi-angle GLCM plus run-length features plus Laplacian-of-Gaussian
      energy at three scales, aggregated across every slice the tumor touches.

  Real 3D shape.
      Bounding-box height and width on the largest slice are not shape
      descriptors. Volume, surface area, sphericity, extent and elongation are.

The whole point of this module is that if THEIA cannot beat THIS, the honest
conclusion is that the deep model is not earning its complexity on this cohort.
"""
from __future__ import annotations

import numpy as np

# Ground-glass sits roughly between aerated lung and soft tissue. The exact
# cut-points vary by paper; these are the common radiological convention and are
# reported as fractions so the choice is visible rather than buried.
GG_LO, GG_HI = -700.0, -300.0
SOLID_LO = -300.0


def to_hu(img01: np.ndarray, center: float, width: float) -> np.ndarray:
    """Invert the display window applied in preprocessing.

    preprocess.window_hu maps [center - width/2, center + width/2] onto [0,1]
    and clips. The inverse is exact for everything that was not clipped, which
    is the whole lung field at the configured window.
    """
    return img01 * width + (center - width / 2.0)


def _first_order(v: np.ndarray, prefix: str) -> dict:
    if v.size == 0:
        return {}
    from scipy import stats

    p = np.percentile(v, [10, 25, 50, 75, 90])
    hist = np.histogram(v, bins=32)[0].astype(float)
    hist /= max(hist.sum(), 1.0)
    return {
        f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std()),
        f"{prefix}_p10": float(p[0]), f"{prefix}_p25": float(p[1]),
        f"{prefix}_median": float(p[2]), f"{prefix}_p75": float(p[3]),
        f"{prefix}_p90": float(p[4]), f"{prefix}_iqr": float(p[3] - p[1]),
        f"{prefix}_skew": float(stats.skew(v)), f"{prefix}_kurt": float(stats.kurtosis(v)),
        f"{prefix}_entropy": float(stats.entropy(hist + 1e-12)),
        f"{prefix}_energy": float(np.mean(v.astype(np.float64) ** 2)),
        f"{prefix}_mad": float(np.mean(np.abs(v - v.mean()))),
        f"{prefix}_range": float(v.max() - v.min()),
    }


def _density_fractions(hu: np.ndarray, m: np.ndarray) -> dict:
    """Ground-glass and solid fractions — the automated analogue of %GG.

    The clinical sheet carries %GG as a radiologist's eyeball estimate and it is
    a known EGFR correlate. Measuring it from the mask makes it an imaging
    feature the model could in principle have learned, rather than a chart
    variable.
    """
    v = hu[m]
    if v.size == 0:
        return {"gg_frac": 0.0, "solid_frac": 0.0, "aerated_frac": 0.0}
    return {
        "gg_frac": float(((v >= GG_LO) & (v < GG_HI)).mean()),
        "solid_frac": float((v >= SOLID_LO).mean()),
        "aerated_frac": float((v < GG_LO).mean()),
    }


def _shape3d(m: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict:
    """Volume, surface, sphericity, extent, elongation over the 3D mask."""
    if not m.any():
        return {k: 0.0 for k in ("vol", "surf", "sphericity", "extent",
                                 "elongation", "bbox_fill")}
    vol = float(m.sum())
    # Surface area by counting exposed faces — mesh-free and adequate at this
    # resolution, where a marching-cubes surface would be dominated by voxel
    # aliasing anyway.
    surf = 0.0
    for ax in range(3):
        d = np.diff(m.astype(np.int8), axis=ax)
        surf += float((d != 0).sum())
    surf += float(m.take([0], axis=0).sum() + m.take([-1], axis=0).sum())
    sphericity = float((np.pi ** (1 / 3)) * ((6 * vol) ** (2 / 3)) / surf) if surf else 0.0

    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0) + 1
    bbox = np.prod(hi - lo)
    ext = float(vol / bbox) if bbox else 0.0
    # Principal axes from the inertia tensor, following PyRadiomics' definitions.
    # Elongation compares the two LARGEST axes and flatness the smallest against
    # the largest, and both are needed: a flat slab and a sphere have identical
    # elongation (1.0 each, since their two major axes are equal) and are told
    # apart only by flatness. Reporting elongation alone silently collapsed that
    # distinction.
    c = idx.mean(0)
    d = (idx - c).astype(np.float64)
    cov = (d.T @ d) / max(len(d), 1)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = float(np.sqrt(ev[1] / ev[0])) if ev[0] > 1e-9 else 0.0
    flat = float(np.sqrt(max(ev[2], 0.0) / ev[0])) if ev[0] > 1e-9 else 0.0
    return {"vol": vol, "surf": surf, "sphericity": sphericity, "extent": ext,
            "elongation": elong, "flatness": flat, "bbox_fill": ext}


def _glcm(slice_u8: np.ndarray, prefix: str) -> dict:
    from skimage.feature import graycomatrix, graycoprops

    g = graycomatrix(slice_u8, distances=[1, 2, 4], levels=64, symmetric=True,
                     normed=True, angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    out = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "ASM"):
        vals = graycoprops(g, prop)
        out[f"{prefix}_{prop}_mean"] = float(np.nanmean(vals))
        out[f"{prefix}_{prop}_range"] = float(np.nanmax(vals) - np.nanmin(vals))
    return out


def _run_length(slice_u8: np.ndarray, m: np.ndarray, prefix: str) -> dict:
    """Short/long-run emphasis and grey-level non-uniformity, horizontal runs.

    A compact stand-in for GLRLM: coarse textures give long runs, fine ones
    short. Computed only inside the mask so background never contributes.
    """
    q = (slice_u8 // 16).astype(np.int16)          # 16 grey levels
    q[~m] = -1
    runs: list[int] = []
    lev: list[int] = []
    for row in q:
        cur, n = -1, 0
        for v in row:
            if v == cur and v >= 0:
                n += 1
            else:
                if cur >= 0 and n:
                    runs.append(n); lev.append(cur)
                cur, n = v, 1 if v >= 0 else 0
        if cur >= 0 and n:
            runs.append(n); lev.append(cur)
    if not runs:
        return {f"{prefix}_sre": 0.0, f"{prefix}_lre": 0.0, f"{prefix}_glnu": 0.0,
                f"{prefix}_rlnu": 0.0, f"{prefix}_runs": 0.0}
    r = np.array(runs, dtype=np.float64)
    counts = np.bincount(np.array(lev), minlength=16).astype(np.float64)
    rcounts = np.bincount(r.astype(int), minlength=2).astype(np.float64)
    n = r.size
    return {
        f"{prefix}_sre": float(np.mean(1.0 / r ** 2)),
        f"{prefix}_lre": float(np.mean(r ** 2)),
        f"{prefix}_glnu": float((counts ** 2).sum() / n ** 2),
        f"{prefix}_rlnu": float((rcounts ** 2).sum() / n ** 2),
        f"{prefix}_runs": float(n),
    }


def _log_energy(img: np.ndarray, m: np.ndarray) -> dict:
    """Laplacian-of-Gaussian energy at three scales — multi-scale edge content."""
    from scipy import ndimage

    out = {}
    for s in (1.0, 2.0, 4.0):
        f = ndimage.gaussian_laplace(img.astype(np.float64), sigma=s)
        v = f[m]
        out[f"log{int(s)}_mean"] = float(np.abs(v).mean()) if v.size else 0.0
        out[f"log{int(s)}_std"] = float(v.std()) if v.size else 0.0
    return out


def features_for_patient(npz_path: str, center: float, width: float) -> dict:
    """All features for one patient, as a name -> value dict."""
    from scipy import ndimage

    with np.load(npz_path) as b:
        img01, roi = b["images"], b["roi"]
    hu = to_hu(img01, center, width)
    m = roi > 0.5

    feats: dict[str, float] = {}
    if not m.any():
        # AIM-tier patients carry no mask. Their ROI is written as zeros, and
        # inventing one here would fabricate a lesion boundary. Fall back to the
        # whole crop and flag it, so the estimator can separate the two tiers.
        m = np.ones_like(roi, dtype=bool)
        feats["has_mask"] = 0.0
    else:
        feats["has_mask"] = 1.0

    # Peritumoral rim: dilate in-plane, subtract the lesion.
    dil = ndimage.binary_dilation(m, structure=np.ones((1, 5, 5)), iterations=2)
    rim = dil & ~m

    feats.update(_first_order(hu[m], "core"))
    feats.update(_first_order(hu[rim] if rim.any() else hu[m], "rim"))
    feats.update(_density_fractions(hu, m))
    feats.update(_shape3d(m))
    feats.update(_log_energy(hu, m))

    # Texture aggregated over the slices the tumor actually occupies, rather
    # than the single largest one -- a lesion is not its median cross-section.
    areas = m.reshape(m.shape[0], -1).sum(1)
    order = np.argsort(areas)[::-1]
    chosen = [int(k) for k in order[:3] if areas[k] > 0] or [int(order[0])]
    acc: dict[str, list[float]] = {}
    for k in chosen:
        # Quantise within the slice's own range so contrast is not dominated by
        # how much air happens to be in the crop.
        sl = hu[k]
        lo, hi = np.percentile(sl[m[k]] if m[k].any() else sl, [1, 99])
        u8 = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
        u8 = (u8 * 63).astype(np.uint8)
        for d in (_glcm(u8, "glcm"), _run_length(u8 * 4, m[k], "rl")):
            for kk, vv in d.items():
                acc.setdefault(kk, []).append(vv)
    for kk, vv in acc.items():
        feats[kk] = float(np.mean(vv))

    return feats


def extract(rows: list[dict], center: float, width: float
            ) -> tuple[np.ndarray, list[str]]:
    """Feature matrix aligned to `rows`, plus the feature names."""
    dicts = [features_for_patient(r["npz"], center, width) for r in rows]
    names = sorted({k for d in dicts for k in d})
    X = np.array([[d.get(n, 0.0) for n in names] for d in dicts], dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names
