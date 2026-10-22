"""The one place this repository turns scores into ranks.

DEF-0001. Every rank in this repository used to be built with
`values.argsort().argsort()`, which assigns *distinct* ranks to *equal* scores.
Tied predictions therefore inherited whatever order their rows happened to
arrive in, and a quantity that must depend only on the data acquired a
dependency on row order.

Two consequences, both measured rather than argued:

* a **constant** predictor — one that says the same thing about every patient —
  scored 0.4567 instead of the 0.5000 it must score, because its n identical
  outputs were spread across 0, 1, ... n-1;
* a three-level categorical predictor on 80 patients ranged **0.4280 – 0.5524**
  under row permutation alone. That spread, 0.124, is larger than most of the
  contrasts this paper argues about.

The imaging predictions are continuous and contain no within-fold ties, so they
were unaffected. The comparators this paper measures itself *against* — smoking
status, pack-years, site, annotation tier — are mostly ties, and moved.

The fix is the mid-rank (fractional rank) convention: every member of a tie
group receives the average of the positions that group occupies. It is the
convention the Mann-Whitney U statistic is defined in, so `rank_normalize`
composes with `roc_auc_score` to give exactly the Mann-Whitney AUC.

Nothing in this module may import from the rest of `theia`: everything that
ranks depends on it.
"""
from __future__ import annotations

import numpy as np


def midrank(values) -> np.ndarray:
    """0-based ranks with tied values averaged.

    ``[0.2, 0.2, 0.2, 0.9] -> [1.0, 1.0, 1.0, 3.0]`` — the three tied entries
    share the mean of positions 0, 1 and 2, and the distinct entry keeps
    position 3.

    Permutation-equivariant by construction: permuting the input permutes the
    output identically, which `argsort().argsort()` was not.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    out = np.empty(n, dtype=float)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j)   # mean of the positions i..j
        i = j + 1
    return out


def rank_normalize(values) -> np.ndarray:
    """Mid-ranks rescaled to [0, 1] — the vector every pooled AUC reads.

    A fold of all-tied scores maps to 0.5 everywhere rather than spreading
    across the interval, so it contributes no ordering to the pooled ranking
    instead of contributing a fabricated one. A fold of one patient is the same
    degenerate case and gets the same 0.5; under the old rule it silently
    received rank 0.0 and was pooled as if it were that fold's lowest-risk
    patient.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return np.full(v.size, 0.5)
    return midrank(v) / (v.size - 1)


def rank_normalize_groups(values, groups) -> np.ndarray:
    """`rank_normalize` applied within each group, in one pass.

    `groups` is a per-element label — the fold, almost always. Each fold is a
    different fitted model whose probability scale is not comparable with the
    others', so ranks are taken within fold and only then pooled.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    out = np.empty(v.size, dtype=float)
    for key in np.unique(g):
        m = g == key
        out[m] = rank_normalize(v[m])
    return out


def rank_within_fold(recs: list[dict], key: str,
                     id_key: str = "patient_id") -> dict[str, float]:
    """`{patient_id: within-fold normalised rank}` for a list of OOF records.

    The dict form five analysis modules independently reimplemented, each with
    the same tie defect. They now share this one.
    """
    by_fold: dict[object, list[dict]] = {}
    for r in recs:
        if key in r:
            by_fold.setdefault(r.get("fold", 0), []).append(r)
    out: dict[str, float] = {}
    for group in by_fold.values():
        ranks = rank_normalize([r[key] for r in group])
        for r, v in zip(group, ranks):
            out[r[id_key]] = float(v)
    return out
