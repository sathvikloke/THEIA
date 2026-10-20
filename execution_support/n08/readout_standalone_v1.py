"""Versioned native fit-consumption/readout entry.

Statistical, OOF, alignment and map-consistency methods are retained byte-for-byte
from the pinned historical readout. Native unit authority comes only from the
reviewed standalone verifier. Native map consumption is not implemented, so the
final/spatial endpoint refuses explicitly and never produces a scientific result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = None  # Scientific entry is unavailable; no legacy root/default is inferred.
READOUT_VERSION = "theia-native-readout-entry/1 (fit consumption; native maps unavailable)"
FIT_SCHEMA, MAP_SCHEMA, CV_SCHEMA = "n08.fit_receipt.v2", "n08.map_receipt.v1", "n08.condition_v_evidence.v1"

DECLARED = {
    "seeds_new": [42, 7],
    "seed_discovery": 1337,
    "partition_seed": 1337,
    "cells": {"A": ["mean", 0.0, "union"], "B": ["mean", 0.5, "union"], "C": ["query0", 0.0, "union"],
              "D": ["query0", 0.5, "union"], "E": ["query0", 0.5, "query0"]},
    "readout_map": {"A": "mean", "B": "mean", "C": "query0", "D": "query0", "E": "query0"},
    "contrasts": {"E-C": {"E": 1, "C": -1}, "D-C": {"D": 1, "C": -1}, "B-A": {"B": 1, "A": -1},
                  "(D-C)-(B-A)": {"D": 1, "C": -1, "B": -1, "A": 1}, "E-D": {"E": 1, "D": -1}},
    "primary": {"contrast": "(D-C)-(B-A)", "endpoint": "auc", "combine": "mean of seed-specific statistics"},
    "predictive_endpoints": ["auc", "logloss", "brier"],
    "spatial_maps": ["readout", "union"], "spatial_metrics": ["grounding_pointing", "grounding_mass"],
    "streams": {"PRIMARY": {"generator": "numpy.random.default_rng(PCG64)", "seed": 20261003, "draws": 50000,
                            "population": "eligible labelled patients sorted by patient_id",
                            "algorithm": "for b in range(B): concat over folds k=0..4 of "
                                         "rng.choice(idx_k, size=len(idx_k), replace=True)"},
                "SPATIAL": {"generator": "numpy.random.default_rng(PCG64)", "seed": 20261004, "draws": 50000,
                            "population": "patients with a projected mask in every cell-seed map, sorted",
                            "algorithm": "same as PRIMARY"}},
    "interval": "percentile, numpy.percentile linear interpolation, [2.5, 97.5] over valid draws",
    "p_value": "two-sided: min(1, 2*min(1+#{T_b<=0}, 1+#{T_b>=0}) / (B_valid+1)); never zero",
    "invalid_draw": "a PRIMARY draw whose resampled labels are single-class, or any predictive statistic "
                    "non-finite, is invalid for every predictive statistic jointly; counted with its reason",
    "primary_test": "reject H0 (mean interaction = 0) iff p <= 0.05",
    "recurrence": "upper 97.5% percentile bound of the mean interaction < 0 AND I_42 < 0 AND I_7 < 0",
    "secondary_family": "5 contrasts x {auc, logloss, brier} minus the primary (14) + 5 contrasts x "
                        "{readout, union} map x {pointing, mass} (20) = 34; each a mean of seed-specific statistics",
    "multiplicity": "serial gate: Holm at 0.05 over the 34 only if the primary is rejected AND procedure "
                    "validation condition V passed; otherwise all 34 are exploratory (estimates, intervals, "
                    "unadjusted and Holm-adjusted p shown, no familywise claim)",
    "condition_V": "validation null fixtures (>= 1000 replicates) give empirical two-sided 0.05 rejection whose "
                   "95% Clopper-Pearson interval contains 0.05 and point <= 0.075 for auc, logloss, brier and "
                   "spatial-mean statistics, and the alignment/seed-identity mutation fixtures are refused",
    "discovery_display": "seed 1337 shown separately with the same draws; never pooled",
    "sensitivity_known_input_defect": {
        "patients": ["AMC-006", "AMC-008", "AMC-016", "AMC-023", "AMC-032", "AMC-046", "AMC-048"],
        "evidence": "revamp/campaign/parallel_expansion_2026-10-02/reporting/r05_denominators/qc/"
                    "AMC_AIM_SERIES_MISMATCH_QC.json",
        "statistic": "the primary (mean over seeds 42 and 7 of the seed-specific AUC interaction) recomputed "
                     "with these patients removed from EVALUATION only (fits unchanged; they stay in training)",
        "stream": {"generator": "numpy.random.default_rng(PCG64)", "seed": 20261005, "draws": 50000,
                   "algorithm": "same as PRIMARY, over the reduced eligible set"},
        "status": "descriptive; outside the 34-member family; post hoc with respect to discovery, declared "
                  "before any new fit outcome"},
    "descriptive_only": ["calibration intercept/slope", "A-C probability agreement", "per-fold contrasts",
                         "best/last sensitivity"],
}
DECLARED_SHA256 = hashlib.sha256(json.dumps(DECLARED, sort_keys=True).encode()).hexdigest()


class AlignmentError(ValueError):
    """Patients, folds, labels or seed identities do not line up across cell-seed fits."""


def contrast_of(v: dict, name: str):
    """A contrast evaluated in its own nested form, e.g. (D - C) - (B - A), not as a running
    weighted sum: equal cells then cancel exactly (validation fixture F3)."""
    if name == "(D-C)-(B-A)":
        return (v["D"] - v["C"]) - (v["B"] - v["A"])
    x, y = name.split("-")
    return v[x] - v[y]


# ------------------------------------------------------------------ statistics
def auc_rows(Y: np.ndarray, Pm: np.ndarray) -> np.ndarray:
    """Row-wise Mann-Whitney AUC with ties averaged (identical to n08_readout.auc)."""
    R = rankdata(Pm, axis=1)
    n1 = Y.sum(1)
    n0 = Y.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((R * Y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def endpoints_rows(Y: np.ndarray, Pm: np.ndarray) -> dict:
    pc = np.clip(Pm, 1e-12, 1 - 1e-12)
    return {"auc": auc_rows(Y, Pm),
            "logloss": (-(Y * np.log(pc) + (1 - Y) * np.log(1 - pc))).mean(1),
            "brier": ((Pm - Y) ** 2).mean(1)}


def fold_stratified_draws(fold: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(fold == k) for k in sorted(set(fold.tolist()))]
    return np.stack([np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx])
                     for _ in range(n_draws)])


def p_two_sided(d: np.ndarray) -> float:
    b = d.size
    return float(min(1.0, 2 * min(1 + int((d <= 0).sum()), 1 + int((d >= 0).sum())) / (b + 1)))


# ------------------------------------------------------------------ T11: p-value Monte Carlo reporting
def tail_counts(d: np.ndarray) -> dict:
    """Counts behind the declared p-value. Ties (T_b == 0) belong to BOTH tails, as in p_two_sided."""
    d = np.asarray(d, dtype=float)
    fin = np.isfinite(d)
    x = d[fin]
    n_lt, n_eq, n_gt = int((x < 0).sum()), int((x == 0).sum()), int((x > 0).sum())
    return {"n_finite": int(x.size), "n_nonfinite": int((~fin).sum()), "n_lt0": n_lt, "n_eq0": n_eq, "n_gt0": n_gt,
            "k_lo": n_lt + n_eq, "k_hi": n_gt + n_eq}


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion."""
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


P_MC_METHOD = ("target p_inf = min(1, 2*min(P*(T<=0), P*(T>=0))) of the bootstrap distribution (B -> infinity; the "
               "+1 terms vanish). Interval: exact Clopper-Pearson for each tail count k_lo, k_hi at 97.5% "
               "(Bonferroni over the two tails -> simultaneous >= 95%); min over tails is monotone, so "
               "min(q_lo,q_hi) lies in [min(L_lo,L_hi), min(U_lo,U_hi)]; doubled and capped at 1. Delta SE of the "
               "estimator 2*(1+k_min)/(B+1): 2*sqrt(B*q*(1-q))/(B+1), q = k_min/B, reported only when the min tail "
               "is unambiguous (the two tail intervals do not overlap), the cap is unreachable (2*U_min < 1) and "
               "k_min >= 10.")


def p_mc(tc: dict, conf: float = 0.95) -> dict:
    """Monte Carlo uncertainty of the declared doubled one-tail p-value from its tail counts (T11)."""
    B, klo, khi = tc["n_finite"], tc["k_lo"], tc["k_hi"]
    if B == 0:
        return {"status": "undefined: no valid draws", "B_valid": 0}
    kmin = min(klo, khi)
    p_hat = float(min(1.0, 2 * (1 + kmin) / (B + 1)))
    each = 1 - (1 - conf) / 2
    llo, ulo = clopper_pearson(klo, B, each)
    lhi, uhi = clopper_pearson(khi, B, each)
    qlo_min, qhi_min = min(llo, lhi), min(ulo, uhi)
    unamb = ulo < lhi or uhi < llo
    why = []
    if not unamb:
        why.append("min-tail switch possible: the two tail intervals overlap")
    if 2 * qhi_min >= 1:
        why.append("cap at 1 reachable inside the interval")
    if kmin < 10:
        why.append(f"k_min = {kmin} < 10: normal approximation unsuitable")
    q = kmin / B
    naive = math.sqrt(max(p_hat * (1 - p_hat), 0) / B)
    return {"definition": "min(1, 2*min(1+k_lo, 1+k_hi)/(B_valid+1)); k_lo = #{T<=0}, k_hi = #{T>=0}",
            "B_valid": B, "k_lo": klo, "k_hi": khi, "k_tie": tc["n_eq0"],
            "min_tail": "lo" if klo < khi else ("hi" if khi < klo else "equal"),
            "p_hat": p_hat, "mc_interval_95": [float(min(1.0, 2 * qlo_min)), float(min(1.0, 2 * qhi_min))],
            "tail_intervals_97_5": {"q_lo": [llo, ulo], "q_hi": [lhi, uhi]},
            "delta_se": float(2 * math.sqrt(B * q * (1 - q)) / (B + 1)) if not why else None,
            "delta_applicable": not why, "delta_not_applicable_because": why,
            "superseded_naive_se_on_doubled_p": naive,
            "superseded_note": "sqrt(p*(1-p)/B) on the doubled, capped p is not the MC SE of this estimator "
                               "(it understates it by ~29% at small p and is 0 at the cap); kept for comparison only",
            "method": P_MC_METHOD}


def mc_decision(pm: dict, alpha: float) -> str:
    """Descriptive: is the decision 'p <= alpha' stable to Monte Carlo error at this B? Never changes the decision."""
    if pm.get("B_valid", 0) == 0:
        return "undefined"
    lo, hi = pm["mc_interval_95"]
    return "stable_reject" if hi <= alpha else ("stable_not_reject" if lo > alpha else "mc_ambiguous")


def summarize(point: float, d: np.ndarray, requested: int | None = None, prior_invalid: dict | None = None) -> dict:
    """Estimate, percentile interval, declared p-value, its tail counts, MC uncertainty and draw accounting.
    `d` holds the draws that passed the joint validity rule; any non-finite value left in it is excluded here and
    counted (it is never silently passed to percentile/p)."""
    d = np.asarray(d, dtype=float)
    tc = tail_counts(d)
    x = d[np.isfinite(d)]
    req = int(requested if requested is not None else d.size)
    inv = dict(prior_invalid or {})
    if tc["n_nonfinite"]:
        inv["nonfinite_statistic_member"] = inv.get("nonfinite_statistic_member", 0) + tc["n_nonfinite"]
    if x.size == 0:
        return {"estimate": float(point), "ci95": None, "p_two_sided": None, "status": "undefined: no valid draws",
                "tail_counts": tc, "draws": {"requested": req, "valid": 0, "invalid": req, "invalid_reasons": inv},
                "n_valid_draws": 0}
    p = p_two_sided(x)
    pm = p_mc(tc)
    assert abs(pm["p_hat"] - p) < 1e-15
    return {"estimate": float(point), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
            "p_two_sided": p, "p_mc": pm, "p_decision_0.05_mc": mc_decision(pm, 0.05),
            "tail_counts": tc, "draws": {"requested": req, "valid": int(x.size), "invalid": req - int(x.size),
                                         "invalid_reasons": inv},
            "n_valid_draws": int(x.size)}


def holm(ps: dict, alpha: float = 0.05) -> dict:
    items = sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adj, run, out, stop = {}, 0.0, {}, False
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
        rej = (not stop) and p <= alpha / (m - i)
        if not rej:
            stop = True
        out[k] = {"p": p, "holm_adjusted_p": adj[k], "rejected_at_0.05": bool(rej)}
    return out


# ------------------------------------------------------------------ alignment
def align(preds: dict) -> tuple[list, np.ndarray, np.ndarray, dict]:
    """preds[(seed, cell)] = {"meta": {...}, "rows": {pid: (fold, prob, label)}}."""
    keys = sorted(preds)
    seen: dict = {}
    for (seed, cell), v in preds.items():           # campaign-level checkpoint uniqueness (fixture M4b)
        for k, sha_ in (v.get("meta", {}).get("best_pt_sha256") or {}).items():
            if sha_ is None:
                continue
            if sha_ in seen:
                raise AlignmentError(f"best.pt {sha_[:12]} reported by both {seen[sha_]} and {(seed, cell, k)}")
            seen[sha_] = (seed, cell, k)
    for (seed, cell), v in preds.items():
        m = v.get("meta", {})
        if m.get("seed") != seed or m.get("cell") != cell:
            raise AlignmentError(f"table under ({seed}, {cell}) carries identity {m.get('seed')}, {m.get('cell')}")
        if m.get("partition_seed") not in (None, DECLARED["partition_seed"]):
            raise AlignmentError(f"({seed}, {cell}) partition seed {m.get('partition_seed')}")
    ref = preds[keys[0]]["rows"]
    sig = {p: (f, y) for p, (f, _, y) in ref.items()}
    for k in keys[1:]:
        other = {p: (f, y) for p, (f, _, y) in preds[k]["rows"].items()}
        if other != sig:
            diff = sorted(set(other.items()) ^ set(sig.items()))[:5]
            raise AlignmentError(f"{k} differs from {keys[0]} in patients/folds/labels, e.g. {diff}")
    pids = sorted(p for p, (f, y) in sig.items() if y in (0, 1))
    y = np.array([sig[p][1] for p in pids], dtype=float)
    fold = np.array([sig[p][0] for p in pids])
    P = {k: np.array([preds[k]["rows"][p][1] for p in pids], dtype=float) for k in keys}
    for k, v in P.items():
        if not np.all(np.isfinite(v)) or v.min() < 0 or v.max() > 1:
            raise AlignmentError(f"{k}: probabilities outside [0, 1] or non-finite")
    return pids, y, fold, P


def align_maps(maps: dict) -> tuple[list, np.ndarray, dict]:
    """maps[(seed, cell)] = {pid: record with fold and union/mean/query0 sub-records}."""
    keys = sorted(maps)
    common = None
    for k in keys:
        s = {p for p, r in maps[k].items() if "union" in r}
        common = s if common is None else common & s
    allm = [{p for p, r in maps[k].items() if "union" in r} for k in keys]
    if any(s != common for s in allm):
        raise AlignmentError("spatially eligible patients differ across cell-seed maps")
    pids = sorted(common)
    fold = np.array([maps[keys[0]][p]["fold"] for p in pids])
    for k in keys[1:]:
        if [maps[k][p]["fold"] for p in pids] != fold.tolist():
            raise AlignmentError(f"{k}: spatial fold assignment differs")
    V = {}
    for (seed, cell) in keys:
        for mapdef in DECLARED["spatial_maps"]:
            key = DECLARED["readout_map"][cell] if mapdef == "readout" else "union"
            for met in DECLARED["spatial_metrics"]:
                V[(seed, cell, mapdef, met)] = np.array([maps[(seed, cell)][p][key][met] for p in pids], float)
                if not np.all(np.isfinite(V[(seed, cell, mapdef, met)])):          # T11: refuse, never average NaN
                    raise AlignmentError(f"({seed}, {cell}) {mapdef} {met}: non-finite per-patient value")
    return pids, fold, V


# ------------------------------------------------------------------ T12: condition V from bound evidence
PRED_STATS = [f"{n}|{m}" for n in DECLARED["contrasts"] for m in DECLARED["predictive_endpoints"]]
SPATIAL_STATS = [f"{n}|{md}_map|{mt.replace('grounding_', '')}" for n in DECLARED["contrasts"]
                 for md in DECLARED["spatial_maps"] for mt in DECLARED["spatial_metrics"]]
V_MIN_REPLICATES, V_ALPHA, V_MAX_SHARE = 1000, 0.05, 0.075


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bound_file(rec: dict | None, root: Path, what: str, problems: list) -> Path | None:
    if not isinstance(rec, dict) or not rec.get("path") or not rec.get("sha256"):
        problems.append(f"{what}: no path/sha256 bound")
        return None
    p = Path(rec["path"]) if Path(rec["path"]).is_absolute() else root / rec["path"]
    if not p.is_file():
        problems.append(f"{what}: {rec['path']} missing")
        return None
    if _file_sha(p) != rec["sha256"]:
        problems.append(f"{what}: {rec['path']} sha256 differs from the evidence record")
        return None
    return p


def verify_condition_v(ev: dict | None, root: Path = ROOT) -> dict:
    """Condition V (PROTOCOL §7) re-derived from a bound evidence record; never from a boolean.
    Returns {"status": passed | not_passed | not_evaluable | not_supplied | unverifiable, "problems", "per_regime"}."""
    if not ev:
        return {"status": "not_supplied", "problems": ["no condition-V evidence supplied"], "per_regime": {}}
    problems: list[str] = []
    if ev.get("schema") != CV_SCHEMA:
        return {"status": "unverifiable", "per_regime": {},
                "problems": [f"evidence schema {ev.get('schema')!r} != {CV_SCHEMA} (a bare boolean is not evidence)"]}
    body = {k: v for k, v in ev.items() if k != "record_body_sha256"}
    if ev.get("record_body_sha256") != canon_sha(body):
        problems.append("evidence record_body_sha256 does not match its body")
    m = ev.get("method") or {}
    if m.get("declared_sha256") != DECLARED_SHA256:
        problems.append("evidence method.declared_sha256 is not this readout's DECLARED")
    if m.get("p_value_definition") != DECLARED["p_value"]:
        problems.append("evidence p-value definition differs from DECLARED")
    if m.get("binding") != "recorded":
        problems.append(f"method binding {m.get('binding')!r}: the readout version that produced the replicates was "
                        "not recorded by the run itself")
    _bound_file(m.get("readout"), root, "method.readout", problems)
    _bound_file((ev.get("generator") or {}).get("file"), root, "generator", problems)
    per_regime, not_eval = {}, []
    regimes = ev.get("regimes") or {}
    if not regimes:
        problems.append("no regimes in the evidence")
    for name, rg in regimes.items():
        rp = _bound_file(rg.get("replicates_file"), root, f"regime {name} replicates", problems)
        rr = {"n_replicates": None, "per_statistic": {}, "failures": len(rg.get("failures") or [])}
        if rg.get("draws") != DECLARED["streams"]["PRIMARY"]["draws"]:
            problems.append(f"regime {name}: draws {rg.get('draws')} != declared {DECLARED['streams']['PRIMARY']['draws']}")
        if rg.get("failures"):
            problems.append(f"regime {name}: {len(rg['failures'])} failed replicate(s) recorded")
        if rp is not None:
            rows = {}
            for ln in rp.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    if r.get("draws") == rg.get("draws"):
                        rows[r["replicate"]] = r
            n = len(rows)
            rr["n_replicates"] = n
            if n != rg.get("n_replicates"):
                problems.append(f"regime {name}: {n} replicates in the file, record says {rg.get('n_replicates')}")
            if n < V_MIN_REPLICATES:
                problems.append(f"regime {name}: {n} < {V_MIN_REPLICATES} replicates")
            for st in PRED_STATS + SPATIAL_STATS:
                vals = [r["p"].get(st) for r in rows.values()]
                if any(v is None for v in vals) or not vals:
                    if st in SPATIAL_STATS:
                        not_eval.append(f"{name}:{st}")
                    else:
                        problems.append(f"regime {name}: {st} missing in some replicate")
                    continue
                k = sum(1 for v in vals if v <= V_ALPHA)
                ci = clopper_pearson(k, n)
                ok = bool(ci[0] <= V_ALPHA <= ci[1] and k / n <= V_MAX_SHARE)
                rr["per_statistic"][st] = {"rejections": k, "n": n, "share": k / n, "clopper_pearson_95": list(ci),
                                           "meets_V": ok}
                claimed = ((rg.get("per_statistic") or {}).get(st) or {}).get("rejections")
                if claimed is not None and claimed != k:
                    problems.append(f"regime {name}: {st} rejections recorded {claimed}, recomputed {k}")
                if not ok:
                    problems.append(f"regime {name}: {st} size {k}/{n} = {k / n:.3f}, CP {ci[0]:.3f}-{ci[1]:.3f}: "
                                    "fails V")
        per_regime[name] = rr
    hard = [p for p in problems if "fails V" not in p]
    if not_eval:
        status = "not_evaluable"
        problems.append(f"{len(not_eval)} spatial statistic-regime pairs have no null replicates (no spatial null)")
    elif hard:
        status = "unverifiable" if any("sha256" in p or "missing" in p or "schema" in p for p in hard) else "not_passed"
    else:
        status = "passed" if not problems else "not_passed"
    return {"status": status, "problems": problems, "per_regime": per_regime, "rule": DECLARED["condition_V"],
            "verified_by": READOUT_VERSION}


def inference_adequacy(ev: dict | None, cv: dict) -> dict:
    """T12: whether p-values are adequate at Holm's thresholds. Stays `unresolved` unless the evidence carries a
    joint-null familywise and tail-calibration tier that this function can recompute; none exists yet."""
    reasons = ["condition V verifies size at 0.05 only; Holm's smallest threshold 0.05/34 ~ 0.00147 relies on bootstrap "
               "tail accuracy that V does not verify",
               "null regimes are generator assumptions, not the realised data-generating process"]
    tier = (ev or {}).get("joint_null_tier")
    if not tier:
        reasons.append("no joint-null familywise (Holm over 34) evidence supplied")
    else:
        reasons.append(f"joint-null tier supplied ({tier.get('id')}): adequacy rule "
                       f"{tier.get('adequacy_rule_id')!r} -> {tier.get('adequacy_verdict')!r}; this readout does not "
                       "upgrade a pre-declared inadequate tier")
    if cv.get("status") != "passed":
        reasons.append(f"condition V status {cv.get('status')}")
    return {"status": "unresolved", "reasons": reasons}


# ------------------------------------------------------------------ readout
def readout(preds: dict, maps: dict | None, seeds=None, n_draws=None, procedure_validation=None,
            include_discovery=True) -> dict:
    seeds = list(seeds or DECLARED["seeds_new"])
    disc = DECLARED["seed_discovery"]
    st = DECLARED["streams"]
    B = n_draws or st["PRIMARY"]["draws"]
    pids, y, fold, P = align(preds)
    have_disc = include_discovery and all((disc, c) in preds for c in DECLARED["cells"])
    groups = seeds + ([disc] if have_disc else [])
    for s in seeds:
        for c in DECLARED["cells"]:
            if (s, c) not in preds:
                raise AlignmentError(f"missing cell-seed fit ({s}, {c})")
    idx = fold_stratified_draws(fold, B, st["PRIMARY"]["seed"])
    Yb = y[idx]
    valid = Yb.min(1) != Yb.max(1)
    reasons = {"single_class_resample": int((~valid).sum())}
    E = {}
    for s in groups:
        for c in DECLARED["cells"]:
            E[(s, c)] = endpoints_rows(Yb, P[(s, c)][idx])
    pt = {(s, c): {m: float(v[0]) for m, v in endpoints_rows(y[None], P[(s, c)][None]).items()}
          for s in groups for c in DECLARED["cells"]}
    for v in E.values():
        for m in v:
            nonfinite = ~np.isfinite(v[m])
            reasons["nonfinite_statistic"] = reasons.get("nonfinite_statistic", 0) + int((nonfinite & valid).sum())
            valid &= ~nonfinite
    out = {"declared_sha256": DECLARED_SHA256, "n_eligible_labelled": len(pids), "n_positive": int(y.sum()),
           "per_fold_denominators": {int(k): {"n": int((fold == k).sum()), "n_pos": int(y[fold == k].sum())}
                                     for k in sorted(set(fold.tolist()))},
           "draws": {"PRIMARY": {"requested": B, "valid": int(valid.sum()), "invalid": int((~valid).sum()),
                                 "invalid_reasons": reasons}},
           "per_seed": {}, "discovery_1337": None, "secondary": {}, "primary": None}

    def contrast(s, name, m, sel=None):
        d = contrast_of({c: E[(s, c)][m] for c in DECLARED["contrasts"][name]}, name)
        return d[valid] if sel is None else d

    def contrast_pt(s, name, m):
        return contrast_of({c: pt[(s, c)][m] for c in DECLARED["contrasts"][name]}, name)

    for s in groups:
        tgt = out["per_seed"] if s != disc else {}
        tab = {}
        for name in DECLARED["contrasts"]:
            for m in DECLARED["predictive_endpoints"]:
                tab[f"{name}|{m}"] = summarize(contrast_pt(s, name, m), contrast(s, name, m), B, reasons)
        tab["per_cell"] = {c: pt[(s, c)] for c in DECLARED["cells"]}
        if s == disc:
            out["discovery_1337"] = {**tab, "note": "displayed separately with the same draws; not pooled"}
        else:
            tgt[str(s)] = tab
    pn, pm = DECLARED["primary"]["contrast"], DECLARED["primary"]["endpoint"]
    mean_d = sum(contrast(s, pn, pm) for s in seeds) / len(seeds)
    mean_pt = sum(contrast_pt(s, pn, pm) for s in seeds) / len(seeds)
    prim = summarize(mean_pt, mean_d, B, reasons)
    prim["seed_estimates"] = {str(s): float(contrast_pt(s, pn, pm)) for s in seeds}
    prim["rejected_at_0.05"] = bool(prim["p_two_sided"] is not None and prim["p_two_sided"] <= 0.05)
    prim["recurrence"] = bool(prim["ci95"] is not None and prim["ci95"][1] < 0
                              and all(v < 0 for v in prim["seed_estimates"].values()))
    out["primary"] = prim

    # per-fold contributions (descriptive): seed-specific interaction inside each outer fold
    pf = {}
    for s in groups:
        pf[str(s)] = {}
        for k in sorted(set(fold.tolist())):
            sel = fold == k
            a = {c: float(auc_rows(y[sel][None], P[(s, c)][sel][None])[0]) for c in DECLARED["cells"]}
            pf[str(s)][int(k)] = {name: float(contrast_of(a, name)) for name in DECLARED["contrasts"]}
    out["per_fold_auc_contrasts"] = pf

    # declared descriptive sensitivity: known input defect excluded from evaluation
    sk = DECLARED["sensitivity_known_input_defect"]
    keep = np.array([p not in set(sk["patients"]) for p in pids])
    if (~keep).any():
        y2, f2 = y[keep], fold[keep]
        B2 = n_draws or sk["stream"]["draws"]
        idx2 = fold_stratified_draws(f2, B2, sk["stream"]["seed"])
        Y2 = y2[idx2]
        ok2 = Y2.min(1) != Y2.max(1)
        a2 = {(s, c): auc_rows(Y2, P[(s, c)][keep][idx2]) for s in groups for c in DECLARED["cells"]}
        p2 = {(s, c): float(auc_rows(y2[None], P[(s, c)][keep][None])[0]) for s in groups for c in DECLARED["cells"]}
        seed_pt = {s: contrast_of({c: p2[(s, c)] for c in DECLARED["cells"]}, pn) for s in groups}
        seed_d = {s: contrast_of({c: a2[(s, c)] for c in DECLARED["cells"]}, pn) for s in groups}
        ok2 &= np.all([np.isfinite(v) for v in seed_d.values()], axis=0)
        r = summarize(sum(seed_pt[s] for s in seeds) / len(seeds),
                      (sum(seed_d[s] for s in seeds) / len(seeds))[ok2], B2,
                      {"single_class_resample_or_nonfinite": int((~ok2).sum())})
        r.update({"seed_estimates": {str(s): float(seed_pt[s]) for s in seeds},
                  "discovery_1337": float(seed_pt[disc]) if disc in seed_pt else None,
                  "n_excluded_present": int((~keep).sum()), "n_eligible": int(keep.sum()),
                  "n_positive": int(y2.sum()), "invalid_draws": int((~ok2).sum()),
                  "status": sk["status"], "patients_excluded": sorted(p for p in pids if p in set(sk["patients"]))})
        out["sensitivity_known_input_defect"] = r
    else:
        out["sensitivity_known_input_defect"] = {"status": "not applicable: none of the patients is eligible"}

    # secondary family: 14 predictive
    sec_p = {}
    for name in DECLARED["contrasts"]:
        for m in DECLARED["predictive_endpoints"]:
            if (name, m) == (pn, pm):
                continue
            d = sum(contrast(s, name, m) for s in seeds) / len(seeds)
            e = sum(contrast_pt(s, name, m) for s in seeds) / len(seeds)
            r = summarize(e, d, B, reasons)
            r["seed_estimates"] = {str(s): float(contrast_pt(s, name, m)) for s in seeds}
            out["secondary"][f"{name}|{m}"] = r
            sec_p[f"{name}|{m}"] = r["p_two_sided"]

    # secondary family: 20 spatial
    if maps is not None:
        spids, sfold, V = align_maps({k: v for k, v in maps.items() if k[0] in groups})
        Bs = n_draws or st["SPATIAL"]["draws"]
        sidx = fold_stratified_draws(sfold, Bs, st["SPATIAL"]["seed"])
        members, sp_valid = {}, np.ones(Bs, dtype=bool)
        for name, w in DECLARED["contrasts"].items():
            for mapdef in DECLARED["spatial_maps"]:
                for met in DECLARED["spatial_metrics"]:
                    for s in groups:
                        if all((s, c, mapdef, met) in V for c in w):
                            v = contrast_of({c: V[(s, c, mapdef, met)] for c in w}, name)
                            members[(name, mapdef, met, s)] = (float(v.mean()), v[sidx].mean(1))
                            sp_valid &= np.isfinite(members[(name, mapdef, met, s)][1])
        # T11: computed, not asserted (align_maps already refuses non-finite per-patient values)
        out["draws"]["SPATIAL"] = {"requested": Bs, "valid": int(sp_valid.sum()), "invalid": int((~sp_valid).sum()),
                                   "invalid_reasons": {"nonfinite_statistic": int((~sp_valid).sum())},
                                   "n_eligible": len(spids)}
        sp_reasons = out["draws"]["SPATIAL"]["invalid_reasons"]
        disc_sp = {}
        for name, w in DECLARED["contrasts"].items():
            for mapdef in DECLARED["spatial_maps"]:
                for met in DECLARED["spatial_metrics"]:
                    per = {s: (members[(name, mapdef, met, s)][0], members[(name, mapdef, met, s)][1][sp_valid])
                           for s in groups if (name, mapdef, met, s) in members}
                    key = f"{name}|{mapdef}_map|{met.replace('grounding_', '')}"
                    if all(s in per for s in seeds):
                        e = sum(per[s][0] for s in seeds) / len(seeds)
                        d = sum(per[s][1] for s in seeds) / len(seeds)
                        r = summarize(e, d, Bs, sp_reasons)
                        r["seed_estimates"] = {str(s): per[s][0] for s in seeds}
                        r["n"] = len(spids)
                        out["secondary"][key] = r
                        sec_p[key] = r["p_two_sided"]
                    else:
                        out["secondary"][key] = {"status": "not estimable: maps missing", "n": len(spids)}
                    if disc in per:
                        disc_sp[key] = summarize(per[disc][0], per[disc][1], Bs, sp_reasons)
        if out["discovery_1337"] is not None:
            out["discovery_1337"]["spatial"] = disc_sp
    else:
        for name in DECLARED["contrasts"]:
            for mapdef in DECLARED["spatial_maps"]:
                for met in DECLARED["spatial_metrics"]:
                    out["secondary"][f"{name}|{mapdef}_map|{met.replace('grounding_', '')}"] = {
                        "status": "not estimable: maps not supplied"}
    sec_p = {k: v for k, v in sec_p.items() if v is not None}           # undefined p keeps its slot (not estimable)
    fam = sorted(out["secondary"])
    out["family"] = {"size_declared": 34, "size_present": len(fam), "estimable": len(sec_p),
                     "not_estimable": sorted(k for k in fam if k not in sec_p)}
    cv = verify_condition_v(procedure_validation)
    v_ok = cv["status"] == "passed"
    h = holm(sec_p) if sec_p else {}
    for k, v in h.items():                                             # descriptive MC stability at its Holm step
        pm_k = out["secondary"][k].get("p_mc") or {}
        v["mc_interval_95"] = pm_k.get("mc_interval_95")
    # A not-estimable member keeps its slot: Holm is not run on a shrunken family.
    confirm = prim["rejected_at_0.05"] and v_ok and len(sec_p) == 34
    adequacy = inference_adequacy(procedure_validation, cv)
    protocol_label = "confirmatory (approximate; Holm over 34 after primary rejection)" if confirm else "exploratory"
    out["multiplicity"] = {
        "primary_rejected": prim["rejected_at_0.05"], "condition_V_passed": v_ok,
        "condition_V": cv, "procedure_validation_source": (procedure_validation or {}).get("source"),
        "family_complete": len(sec_p) == 34,
        "protocol_label_frozen_s7": protocol_label,
        "inference_adequacy": adequacy,
        "status": protocol_label if (confirm and adequacy["status"] == "resolved") else "exploratory",
        "status_rule": "the frozen §7 label is reported as `protocol_label_frozen_s7`; `status` is exploratory whenever "
                       "inference adequacy at the Holm thresholds is unresolved (ANALYTICAL_DECISIONS.md, N08 review "
                       "item 8). Size at 0.05 does not establish calibration at 0.05/34 or familywise control.",
        "holm": h,
        "why_exploratory": None if (confirm and adequacy["status"] == "resolved") else [w for w, c in (
            ("primary not rejected", not prim["rejected_at_0.05"]),
            ("condition V not passed or not supplied", not v_ok),
            ("secondary family incomplete", len(sec_p) != 34),
            ("inference adequacy unresolved", adequacy["status"] != "resolved")) if c]}
    # descriptive: A-C agreement and calibration-free summaries
    out["descriptive"] = {"a_vs_c_abs_dprob": {
        str(s): {"median": float(np.median(np.abs(P[(s, "A")] - P[(s, "C")]))),
                 "max": float(np.max(np.abs(P[(s, "A")] - P[(s, "C")])))} for s in groups},
        "mean_prob": {f"{s}|{c}": float(P[(s, c)].mean()) for s in groups for c in DECLARED["cells"]}}
    return out


# ------------------------------------------------------------------ loading real runs
def _outer_ids(fold_entry: dict) -> list:
    ot = fold_entry["outer_test"]
    return sorted(ot["patient_ids"] if isinstance(ot, dict) else ot)


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run_predictions(ckpt_root: Path, run_id: str, seed: int, cell: str, splits: dict,
                         receipts: dict | None = None, verify_files: str = "full") -> dict:
    """OOF predictions of one cell-seed run, bound to the frozen split and (when given) to the
    accepted receipts: receipts[f"{run_id}-fold{k}"] = {"accepted", "best_pt_sha256"
    [, "fold_summary_sha256"][, "files": {"best.pt": {"sha256", "bytes"}, "last.pt": {...}}]}.
    Files named in a receipt must exist with the receipt's size and (verify_files="full") sha256."""
    rows, meta = {}, {"seed": seed, "cell": cell, "run_id": run_id, "partition_seed": None, "best_pt_sha256": {}}
    for k in range(5):
        p = ckpt_root / run_id / f"fold{k}" / "fold_summary.json"
        raw = p.read_bytes()
        fs = json.loads(raw)
        comps = fs.get("fingerprint_components") or {}
        if comps:                                     # replication runs (R07+): bind identity
            if comps.get("seed") != seed or comps.get("split") != splits["folds"][k]["trainer_split_component"]:
                raise AlignmentError(f"{run_id} fold {k}: fingerprint seed/split do not match ({seed}, frozen split)")
            meta["partition_seed"] = comps.get("partition_seed", meta["partition_seed"])
        oof_ids = sorted(r["patient_id"] for r in fs["summary"]["oof"])
        if oof_ids != _outer_ids(splits["folds"][k]):
            raise AlignmentError(f"{run_id} fold {k}: OOF patients differ from the frozen outer test "
                                 f"(extra {sorted(set(oof_ids) - set(_outer_ids(splits['folds'][k])))[:3]}, "
                                 f"missing {sorted(set(_outer_ids(splits['folds'][k])) - set(oof_ids))[:3]})")
        scored = (fs["summary"].get("best_pt_scored") or {}).get("sha256")
        meta["best_pt_sha256"][k] = scored
        if receipts is not None:
            rec = receipts.get(f"{run_id}-fold{k}")
            if not rec or rec.get("accepted") is not True:
                raise AlignmentError(f"{run_id} fold {k}: no accepted receipt")
            if scored is None or scored != rec.get("best_pt_sha256"):
                raise AlignmentError(f"{run_id} fold {k}: scored best.pt {scored} != accepted receipt "
                                     f"{rec.get('best_pt_sha256')}")
            if rec.get("fold_summary_sha256") and hashlib.sha256(raw).hexdigest() != rec["fold_summary_sha256"]:
                raise AlignmentError(f"{run_id} fold {k}: fold_summary.json (predictions) changed since acceptance")
            for name, want in (rec.get("files") or {}).items():
                fp = p.parent / name
                if want is None:
                    continue
                if not fp.exists():
                    raise AlignmentError(f"{run_id} fold {k}: {name} named in the accepted receipt is missing")
                if fp.stat().st_size != want["bytes"]:
                    raise AlignmentError(f"{run_id} fold {k}: {name} size {fp.stat().st_size} != receipt {want['bytes']}")
                if verify_files == "full" and _file_sha(fp) != want["sha256"]:
                    raise AlignmentError(f"{run_id} fold {k}: {name} sha256 differs from the accepted receipt")
        for r in fs["summary"]["oof"]:
            if r["patient_id"] in rows:
                raise AlignmentError(f"{run_id}: {r['patient_id']} in two outer folds")
            rows[r["patient_id"]] = (int(r["fold"]), float(r["egfr_prob"]), int(r["egfr_true"]))
    return {"meta": meta, "rows": rows}


# ------------------------------------------------------------------ T10/T09: design, data, fit receipts, maps
CELLS_ALL = tuple(DECLARED["cells"])
FOLDS_ALL = tuple(range(5))
SCORER = "theia.analysis.spatial_comparators.score_production"


def _abs(p: str, base: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


def matrix_identity_problems(rows: list[dict]) -> list[str]:
    """FIX-R1: every matrix row must name a distinct fit_id AND a distinct (cell, fold, seed) unit, in final and
    partial modes alike. A repeat is an invalid matrix, refused by name; never deduplicated (partial mode counted a
    repeated row twice)."""
    probs, seen_id, seen_unit = [], {}, {}
    for i, r in enumerate(rows):
        unit = (r.get("cell"), str(r.get("fold")).strip(), str(r.get("init_seed")).strip())
        try:
            unit = (unit[0], int(unit[1]), int(unit[2]))
        except ValueError:
            probs.append(f"matrix identity: row {i} fold/init_seed not integers {unit[1:]}")
        if r.get("fit_id") in seen_id:
            probs.append(f"matrix identity: row {i} repeats fit_id {r.get('fit_id')} (row {seen_id[r.get('fit_id')]})")
        if unit in seen_unit:
            probs.append(f"matrix identity: row {i} repeats (cell, fold, seed) {unit} (row {seen_unit[unit]})")
        seen_id.setdefault(r.get("fit_id"), i)
        seen_unit.setdefault(unit, i)
    return probs


def check_design(rows: list[dict], version: str) -> list[str]:
    """T10: exactly the intended 50 units (A-E x folds 0-4 x seeds 42/7), unique, one run per cell-seed, each row's
    objective equal to the declared cell."""
    probs = []
    want = {(c, k, s) for c in CELLS_ALL for k in FOLDS_ALL for s in DECLARED["seeds_new"]}
    got = [(r["cell"], int(r["fold"]), int(r["init_seed"])) for r in rows]
    if len(rows) != 50:
        probs.append(f"design: matrix has {len(rows)} rows, the design has 50")
    if len(set(got)) != len(got):
        probs.append("design: matrix repeats a (cell, fold, seed) unit")
    if set(got) != want:
        probs.append(f"design: units differ from A-E x 0-4 x 42/7: missing {sorted(want - set(got))[:3]}, "
                     f"extra {sorted(set(got) - want)[:3]}")
    ids = [r["fit_id"] for r in rows]
    if len(set(ids)) != len(ids):
        probs.append("design: matrix repeats a fit_id")
    runs: dict = {}
    for r in rows:
        runs.setdefault((r["cell"], int(r["init_seed"])), set()).add(r["run_id"])
        pool, gw, gt = DECLARED["cells"].get(r["cell"], (None, None, None))
        if (r.get("region_pooling"), float(r.get("ground_weight", "nan")), r.get("ground_target"),
                r.get("readout_map")) != (pool, gw, gt, DECLARED["readout_map"].get(r["cell"])):
            probs.append(f"design: {r['fit_id']} objective columns are not cell {r['cell']}")
        if int(r.get("partition_seed", DECLARED["partition_seed"])) != DECLARED["partition_seed"]:
            probs.append(f"design: {r['fit_id']} partition seed {r.get('partition_seed')}")
    if any(len(v) != 1 for v in runs.values()):
        probs.append("design: a cell-seed has more than one run_id")
    if len({min(v) for v in runs.values()}) != len(runs):
        probs.append("design: two cell-seeds share a run_id")
    if version == "M1" and any(not i.startswith("n08m1-") for i in ids):
        probs.append("design: an M1 readout needs M1 fit ids (n08m1-*); the matrix names other units")
    return probs


def verify_data_manifest(manifest_path: Path, acceptance_path: Path | None, version: str, root: Path) -> tuple[dict, list]:
    """M1_SCHEMA consumer checks: body hash, version, rows bytes, rows<->entries 1:1 (order, id, npz path), every
    npz re-hashed, no blocking m1_status, acceptance decision for this exact body hash."""
    probs: list[str] = []
    if not manifest_path.is_file():
        return {}, [f"data: manifest {manifest_path} missing"]
    man = json.loads(manifest_path.read_text())
    miss = [k for k in ("dataset_version", "parent_version", "rows_path", "rows_sha256", "n_rows", "entries",
                        "preprocessing", "manifest_body_sha256") if k not in man]          # M1_SCHEMA keys
    if miss:
        return {}, [f"data: manifest lacks M1_SCHEMA keys {miss}"]
    info = {"path": str(manifest_path), "file_sha256": _file_sha(manifest_path),
            "dataset_version": man.get("dataset_version"), "manifest_body_sha256": man.get("manifest_body_sha256")}
    if canon_sha({k: v for k, v in man.items() if k != "manifest_body_sha256"}) != man.get("manifest_body_sha256"):
        probs.append("data: manifest_body_sha256 does not match the manifest body")
    if man.get("dataset_version") != version:
        probs.append(f"data: dataset_version {man.get('dataset_version')!r} != configured {version!r}")
    rows_p = _abs(man.get("rows_path", ""), root)
    if not rows_p.is_file():
        return info, probs + [f"data: rows file {man.get('rows_path')} missing"]
    info["rows_sha256"] = _file_sha(rows_p)
    if info["rows_sha256"] != man.get("rows_sha256"):
        probs.append("data: rows file sha256 differs from the manifest")
    rows = [json.loads(ln) for ln in rows_p.read_text().splitlines() if ln.strip()]
    ents = man.get("entries") or []
    if not (len(rows) == len(ents) == man.get("n_rows")):
        probs.append(f"data: {len(rows)} rows, {len(ents)} entries, n_rows {man.get('n_rows')}")
    norm = lambda s: str(Path(str(s)))  # noqa: E731  ('./data/x.npz' and 'data/x.npz' are the same file)
    for r, e in zip(rows, ents):
        if r.get("patient_id") != e.get("patient_id") or norm(r.get("npz")) != norm(e.get("npz_path")):
            probs.append(f"data: rows and entries differ at {r.get('patient_id')} / {e.get('patient_id')}")
            break
    npz_all = []
    for e in ents:
        p = _abs(e.get("npz_path", ""), root)
        if not p.is_file():
            probs.append(f"data: npz missing for {e.get('patient_id')}")
        elif p.stat().st_size != e.get("npz_bytes") or _file_sha(p) != e.get("npz_sha256"):
            probs.append(f"data: npz of {e.get('patient_id')} differs from the manifest (bytes/sha256)")
        npz_all.append([e.get("patient_id"), os.path.normpath(str(e.get("npz_path"))), e.get("npz_sha256")])
        if str(e.get("m1_status", "")).startswith("unresolved"):
            probs.append(f"data: {e.get('patient_id')} has blocking m1_status {e.get('m1_status')}")
    info.update({"n_entries": len(ents), "npz_all_sha256": canon_sha(npz_all),
                 "labels": {r["patient_id"]: (r.get("labels") or {}).get("EGFR") for r in rows}})
    if acceptance_path is None:
        probs.append("data: no acceptance record supplied (M1_ACCEPTANCE.json decision must be 'accepted')")
    elif not acceptance_path.is_file():
        probs.append(f"data: acceptance record {acceptance_path} missing")
    else:
        acc = json.loads(acceptance_path.read_text())
        info["acceptance"] = {"path": str(acceptance_path), "sha256": _file_sha(acceptance_path),
                              "decision": acc.get("decision")}
        if acc.get("decision") != "accepted" or acc.get("manifest_body_sha256") != man.get("manifest_body_sha256"):
            probs.append(f"data: acceptance decision {acc.get('decision')!r} / body hash not this manifest's")
    return info, probs








def verify_map_receipt_consumer(maps_dir: Path, run_id: str, cell: str, which: str, fits: dict, data: dict,
                                split_fold: dict, oof_ids: set) -> tuple[dict | None, dict, list]:
    """T09: a map output is consumed only through an accepted MAP_RECEIPT bound to the accepted fits' checkpoints,
    the configured data, the patient set and the map definition. Returns (maps doc, provenance, problems)."""
    tag = f"maps {run_id} {which}"
    rp = maps_dir / f"MAP_RECEIPT_{run_id}_{which}.json"
    prov: dict = {"receipt_path": str(rp)}
    failed_name = maps_dir / f"N08_MAPS_{run_id}_{which}.control_failed.json"
    if failed_name.exists():
        prov["control_failed_output_present"] = str(failed_name)
    if not rp.is_file():
        return None, prov, [f"{tag}: no MAP_RECEIPT (a maps JSON without an accepted receipt is never consumed)"]
    try:
        mr = json.loads(rp.read_text())
    except ValueError as exc:
        return None, prov, [f"{tag}: MAP_RECEIPT does not parse ({exc})"]
    prov.update({"receipt_sha256": _file_sha(rp), "control": mr.get("control"), "exit_code": mr.get("exit_code"),
                 "status": mr.get("status")})
    probs: list[str] = []
    for f, v in (("schema", MAP_SCHEMA), ("status", "accepted"), ("failed_reasons", []), ("exit_code", 0),
                 ("cell", cell), ("run_id", run_id), ("which", which)):
        if mr.get(f) != v:
            probs.append(f"{tag}: {f} {mr.get(f)!r} != {v!r}")
    if (mr.get("control") or {}).get("passed") is not True:
        probs.append(f"{tag}: control.passed is {(mr.get('control') or {}).get('passed')!r}")
    md = mr.get("map_definition") or {}
    if md.get("readout_map") != DECLARED["readout_map"][cell] or md.get("scorer") != SCORER:
        probs.append(f"{tag}: map_definition {md.get('readout_map')!r}/{md.get('scorer')!r} is not the declared one")
    cks = mr.get("checkpoints") or []
    if sorted(c.get("fold") for c in cks if isinstance(c, dict)) != list(FOLDS_ALL):
        probs.append(f"{tag}: checkpoints cover folds {sorted(c.get('fold') for c in cks if isinstance(c, dict))}")
    for c in cks:
        f = fits.get(c.get("fold"))
        if f is None:
            probs.append(f"{tag}: fold {c.get('fold')} has no accepted fit")
            continue
        ident = f["best_pt"] if which == "best" else f["last_pt"]
        if (c.get("fit_id"), c.get("fit_receipt_sha256"), c.get("fit_identity_sha256")) != \
                (f["fit_id"], f["receipt_sha256_now"], f["fit_identity_sha256"]):
            probs.append(f"{tag}: fold {c.get('fold')} not bound to the current accepted fit receipt")
        if not ident or c.get("sha256") != ident.get("sha256") or c.get("bytes") != ident.get("bytes"):
            probs.append(f"{tag}: fold {c.get('fold')} checkpoint {str(c.get('sha256'))[:12]} is not the accepted "
                         f"{which}.pt")
        if not c.get("path") or Path(c["path"]).resolve() != (Path(f["fold_dir"]) / f"{which}.pt").resolve():
            probs.append(f"{tag}: fold {c.get('fold')} checkpoint path {c.get('path')} is not the accepted fold's")
    dd = mr.get("data") or {}
    for f in ("dataset_version", "manifest_body_sha256", "rows_sha256", "npz_all_sha256"):
        if data.get(f) is None or dd.get(f) != data.get(f):
            probs.append(f"{tag}: data.{f} {str(dd.get(f))[:16]} != configured {str(data.get(f))[:16]}")
    lg = mr.get("log") or {}
    lgp = _abs(lg.get("path", ""), maps_dir) if lg.get("path") else None
    if lgp is None or not lgp.is_file() or lgp.stat().st_size != lg.get("bytes") or _file_sha(lgp) != lg.get("sha256"):
        probs.append(f"{tag}: maps job log missing or not the one bound in the receipt (v1.1 field `log`)")
    o = mr.get("output") or {}
    want_name = f"N08_MAPS_{run_id}_{which}.json"
    op = _abs(o.get("path", ""), maps_dir)
    if op.name != want_name:
        return None, prov, probs + [f"{tag}: output {op.name} is not {want_name} (control-failed outputs are never read)"]
    if not op.is_file() or op.stat().st_size != o.get("bytes") or _file_sha(op) != o.get("sha256"):
        return None, prov, probs + [f"{tag}: output missing or bytes/sha256 differ from the receipt"]
    doc = json.loads(op.read_text())
    prov.update({"output_path": str(op), "output_sha256": o.get("sha256"),
                 "metadata": {k: v for k, v in doc.items() if k != "per_patient"}})   # never discarded
    if doc.get("control_passed") is not True:
        probs.append(f"{tag}: maps JSON control_passed {doc.get('control_passed')!r}")
    if doc.get("cell") != cell or doc.get("run") != run_id:
        probs.append(f"{tag}: maps JSON names {doc.get('cell')}/{doc.get('run')}")
    pp = doc.get("per_patient") or {}
    ps = mr.get("patient_set") or {}
    if set(pp) != oof_ids:
        probs.append(f"{tag}: per_patient covers {len(pp)} patients, the accepted OOF set has {len(oof_ids)}")
    if ps.get("patient_ids_sha256") != canon_sha(sorted(pp)) or ps.get("n") != len(pp):
        probs.append(f"{tag}: patient_set does not bind the per_patient ids")
    pf: dict = {}
    for pid, r in pp.items():
        pf[str(r.get("fold"))] = pf.get(str(r.get("fold")), 0) + 1
        if split_fold.get(pid) != r.get("fold"):
            probs.append(f"{tag}: {pid} fold {r.get('fold')} != frozen outer-test fold {split_fold.get(pid)}")
            break
        if which == "last" and not (isinstance(r.get("p"), (int, float)) and 0 <= r["p"] <= 1 and math.isfinite(r["p"])):
            probs.append(f"{tag}: {pid} has no valid last.pt probability")
            break
    if {str(k): v for k, v in (ps.get("per_fold_n") or {}).items()} != pf:
        probs.append(f"{tag}: patient_set.per_fold_n {ps.get('per_fold_n')} != per_patient {pf}")
    if ps.get("n_spatial") != sum(1 for r in pp.values() if "union" in r):
        probs.append(f"{tag}: patient_set.n_spatial does not match the per_patient records")
    return (doc if not probs else None), prov, probs


def partial_summary(rows: list[dict], recs: dict, unit_probs: dict, preds: dict) -> dict:
    """T10: an explicitly partial ENGINEERING summary. No primary, no contrast across missing cells, no p-values,
    no intervals, no family. Missing or unverified units are listed, never imputed."""
    runs: dict = {}
    for r in rows:
        runs.setdefault((int(r["init_seed"]), r["cell"]), []).append(r["fit_id"])
    complete = {k: v for k, v in runs.items() if all(f in recs and not unit_probs.get(f) for f in v) and len(v) == 5}
    per_run = {}
    for (s, c) in sorted(complete):
        t = preds.get((s, c))
        if t is None:
            continue
        lab = [(p, y) for p, (f, p, y) in t["rows"].items() if y in (0, 1)]
        yv = np.array([y for _, y in lab], float)
        pv = np.array([p for p, _ in lab], float)
        per_run[f"{s}|{c}"] = {m: float(v[0]) for m, v in endpoints_rows(yv[None], pv[None]).items()}
    seeds_complete = [s for s in DECLARED["seeds_new"] if all((s, c) in complete for c in CELLS_ALL)]
    return {"label": "PARTIAL ENGINEERING SUMMARY - NOT the N08 replication result. No primary, recurrence, p-value, "
                     "interval or secondary family is computed; missing or unverified units are listed and never "
                     "imputed; a one-seed table is not a replication.",
            "units_total": len(rows), "units_verified": sum(1 for r in rows if r["fit_id"] in recs and not unit_probs.get(r["fit_id"])),
            "units": {r["fit_id"]: ("verified" if r["fit_id"] in recs and not unit_probs.get(r["fit_id"]) else
                                    ("missing" if r["fit_id"] not in recs and any("no ACCEPTED" in p for p in unit_probs.get(r["fit_id"], []))
                                     else "unverified")) for r in rows},
            "unit_problems": {k: v[:5] for k, v in unit_probs.items() if v},
            "complete_cell_seed_runs": sorted(f"{s}|{c}" for s, c in complete),
            "seeds_with_all_25_units_verified": seeds_complete,
            "per_run_point_metrics_descriptive": per_run}


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    tmp.replace(p)


# ------------------------------------------------------------------ T13: input recorder and run receipt
_INPUT_REC: dict | None = None          # active only inside main(); audit hooks cannot be removed, so the hook checks this
_HOOK_INSTALLED = False
_SYS_PREFIXES = tuple(sorted({os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                                    sys.base_exec_prefix) if p}))


def _audit(event: str, args) -> None:
    rec = _INPUT_REC
    if rec is None:
        return
    try:
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if path is None or isinstance(path, int):
                return
            if mode is not None and any(ch in str(mode) for ch in "wax+"):
                return                                                  # an output, not an input
            if mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                return
            rec["files"].add(os.path.abspath(os.fsdecode(path)))
        elif event in ("os.scandir", "os.listdir"):
            p = args[0] if args else None
            rec["dirs"].add(os.path.abspath(os.fsdecode(p) if p is not None else "."))
    except Exception:                                                   # noqa: BLE001 - recording must never alter the run
        rec["errors"] = rec.get("errors", 0) + 1


def _start_recording() -> dict:
    global _INPUT_REC, _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    _INPUT_REC = {"files": set(), "dirs": set(), "errors": 0}
    return _INPUT_REC


def _stop_recording() -> dict | None:
    global _INPUT_REC
    rec, _INPUT_REC = _INPUT_REC, None
    return rec


def _is_project_input(p: str) -> bool:
    rp = os.path.realpath(p)
    return not rp.startswith(_SYS_PREFIXES) and not rp.startswith("/dev/") and "__pycache__" not in rp


def _dir_digest(d: str) -> dict:
    try:
        names = sorted(os.listdir(d))
    except OSError as exc:
        return {"path": d, "error": type(exc).__name__}
    return {"path": d, "n_entries": len(names),
            "entries_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}








# Native integration block. The retained method bodies are inserted unchanged.
NATIVE_VERIFIER_SHA256 = "013e165ee000bd7876f89c25eabc6911c98359fed148d12fb729034144319b42"
RETAINED_READOUT_SHA256 = "b5e1ef4ef64f415cda4895477377ce7e1614e41a9526f5c2f588d6e289830e70"
NATIVE_REFUSAL_SCHEMA = "theia.corrected_cpu.readout_refusal.v1"


def _require_native(ok, message):
    if not ok:
        raise AlignmentError(message)


def load_native_verifier():
    """Execute only the exact reviewed native verifier, with no legacy loader."""
    import types
    path = HERE / "standalone_receipts.py"
    content = path.read_bytes()
    _require_native(hashlib.sha256(content).hexdigest() == NATIVE_VERIFIER_SHA256,
                    "native verifier source changed; review the new component first")
    module = types.ModuleType("_native_readout_receipts_v1")
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


def check_native_campaign(recs: dict, rows: list[dict]) -> list[str]:
    """Retained uniqueness/initialization checks with explicitly native identity names."""
    probs = []
    for key in ("fold_dir", "best_pt_sha256", "fit_identity_sha256", "native_bundle_sha256"):
        seen: dict = {}
        for fid, r in recs.items():
            v = r.get(key)
            if v is not None and v in seen:
                probs.append(f"native execution: {key} shared by {seen[v]} and {fid}")
            seen.setdefault(v, fid)
    fps = {r.get("execution_set_sha256") for r in recs.values()}
    if recs and (len(fps) != 1 or None in fps):
        probs.append(f"native execution: {len(fps)} different execution-set identities across accepted units")
    by_seed: dict = {}
    for r in rows:
        if r["fit_id"] in recs:
            by_seed.setdefault(int(r["init_seed"]), set()).add(recs[r["fit_id"]].get("initial_state_sha256"))
    for s, v in by_seed.items():
        if len(v) != 1:
            probs.append(f"native execution: seed {s} fits carry {len(v)} different initial-state digests")
    if len(by_seed) == 2 and len(set.union(*by_seed.values())) != 2:
        probs.append("native execution: seeds 42 and 7 share an initial-state digest")
    return probs


def verify_native_fits(root: Path, output_root: Path) -> dict:
    """Obtain exactly 50 currently FULL-verified native units; no readout is computed.

    The returned context is produced afresh in this process. A saved consumption
    report, structural inspection, partial roster or caller-supplied acceptance
    flag is never loaded as authority.
    """
    root, output_root = Path(root).resolve(strict=True), Path(output_root).resolve(strict=True)
    native = load_native_verifier()
    evidence = native.Evidence()
    driver = evidence.module("driver")
    matrix, rows = native._matrix(driver, evidence)
    problems = matrix_identity_problems(rows) + check_design(rows, "M1")
    _require_native(not problems, "; ".join(problems))
    paths = [output_root / "units" / row["fit_id"] / "RESULT.json" for row in rows]
    missing = [path.parent.name for path in paths if not path.is_file()]
    _require_native(not missing, f"native readout requires all 50 matrix results; missing {missing[:3]}")
    wanted = {path.parent.name for path in paths}
    smoke_id = driver.select_row(matrix, "smoke", 0)["fit_id"]
    extras = [path.parent.name for path in (output_root / "units").glob("*/RESULT.json")
              if path.parent.name not in wanted | {smoke_id}]
    _require_native(not extras, f"native results outside the frozen matrix: {sorted(extras)}")
    try:
        report = native.verify_results(paths, root=root, output_root=output_root, mode="full")
    except (Exception, SystemExit) as exc:
        raise AlignmentError(f"native full verification refused: {type(exc).__name__}: {exc}") from exc
    _require_native(
        report.get("schema") == native.SCHEMA
        and report.get("mode") == "full" and report.get("status") == "verified_current"
        and report.get("authority") == "e05_native_execution"
        and type(report.get("scientific_fit_count")) is int and report["scientific_fit_count"] == 50
        and report.get("full_matrix_verified") is True,
        "native readout requires current full verification of all 50 scientific fits",
    )
    units = report.get("units", [])
    _require_native(len(units) == 50 and {u.get("fit_id") for u in units} == wanted,
                    "native full report does not contain exactly the 50 frozen units")
    by_id = {unit["fit_id"]: unit for unit in units}
    all_inputs = {}
    for unit in units:
        _require_native(unit.get("kind") == "fit" and unit.get("status") == "verified_current"
                        and type(unit.get("scientific_fit_count")) is int and unit["scientific_fit_count"] == 1,
                        "structural/bootstrap unit cannot become a native readout fit")
        inputs = unit.get("inputs", [])
        _require_native(inputs and len({p["path"] for p in inputs}) == len(inputs),
                        "native bundle has missing/duplicate input identities")
        _require_native(unit.get("native_bundle_sha256") == native.digest(sorted(inputs, key=lambda p: p["path"])),
                        "native bundle digest differs from its bound inputs")
        for item in inputs:
            previous = all_inputs.setdefault(item["path"], item)
            _require_native(previous == item, "native inputs disagree across units")
    for name, item in all_inputs.items():
        evidence.read(Path(name), binary=True)
        _require_native(evidence.files[name] == item, "native input identity changed after full verification: " + name)
    recs = {}
    for row in rows:
        fid = row["fit_id"]
        unit = output_root / "units" / fid
        bound = by_id[fid]
        for name in ("EXECUTION_BINDING.json", "WORKER_RESULT.json", "RESULT.json"):
            _require_native(str(unit / name) in {p["path"] for p in bound["inputs"]},
                            fid + ": native bundle does not bind its own " + name)
        binding = evidence.read(unit / "EXECUTION_BINDING.json")
        worker = evidence.read(unit / "WORKER_RESULT.json")
        _require_native(binding["row"] == row, fid + ": bound frozen matrix row changed")
        expected_fold = output_root / "checkpoints" / row["run_id"] / ("fold" + row["fold"])
        _require_native(binding["fold_dir"] == str(expected_fold), fid + ": native fold path changed")
        fi = worker["fit_identity"]
        fit_sha = native.digest(fi)
        _require_native(fit_sha == worker["fit_identity_sha256"] == bound["fit_identity_sha256"]
                        == bound["current_fit_identity_sha256"], fid + ": native fit identity changed")
        ids = worker["acceptance"]["identities"]
        recs[fid] = {
            "fit_id": fid, "native_result_path": str(unit / "RESULT.json"),
            "native_bundle_sha256": bound["native_bundle_sha256"],
            "execution_set_sha256": bound["execution_set_sha256"],
            "fit_identity_sha256": fit_sha, "fold_dir": str(expected_fold),
            "best_pt_sha256": ids["best_pt"]["sha256"],
            "best_pt": ids["best_pt"], "last_pt": ids["last_pt"],
            "fold_summary_sha256": ids["fold_summary"]["sha256"],
            "initial_state_sha256": fi["initial_state_sha256"],
        }
    problems = check_native_campaign(recs, rows)
    _require_native(not problems, "; ".join(problems))
    evidence.unchanged()
    return {
        "verification": report, "rows": rows, "fits": recs,
        "execution_set_sha256": next(iter(recs.values()))["execution_set_sha256"],
        "native_input_bundle_sha256": native.digest(sorted(all_inputs.values(), key=lambda p: p["path"])),
        "native_verifier_sha256": NATIVE_VERIFIER_SHA256,
        "readout_complete": False, "canonical_adoption": False,
    }


def require_native_maps(verified: dict | None, maps_dir: Path | None, which: str):
    """No native map-receipt interface has been reviewed; never adapt legacy hashes."""
    raise AlignmentError(
        "native_map_integration_unavailable: the versioned native map producer/consumer "
        "must bind native bundles and checkpoints and retain map-control, data, patient, "
        "fold and definition checks; no spatial or final scientific readout was computed"
    )


def main(argv=None) -> int:
    """Refuse the unavailable endpoint before any expensive fit verification."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Versioned native readout input integration; maps unavailable.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--maps-dir", type=Path)
    parser.add_argument("--which", choices=("best", "last"), default="best")
    args = parser.parse_args(argv)
    try:
        # Capability preflight comes first: no full array/checkpoint verification
        # can make this endpoint available. Explicit prerequisite verification
        # remains separately callable through verify_native_fits().
        require_native_maps(None, args.maps_dir, args.which)
    except (Exception, SystemExit) as exc:
        reason = str(exc)
        refusal = {
            "schema": NATIVE_REFUSAL_SCHEMA, "decision": "refused",
            "refusal": "native_map_integration_unavailable" if reason.startswith("native_map_integration_unavailable:")
                       else "native_input_verification_failed",
            "reason": reason, "readout_version": READOUT_VERSION,
            "which_checkpoint": args.which, "readout_complete": False,
            "scientific_readout_count": 0, "canonical_adoption": False,
            "native_full_fits_verified": 0, "fit_verification_performed": False,
            "phase": "capability_preflight",
            "lineage": "e05_native_execution; no legacy release, witness, controller or accepted-alias authority",
            "native_verifier_sha256": NATIVE_VERIFIER_SHA256,
            "retained_method_source_sha256": RETAINED_READOUT_SHA256,
            "readout_source_sha256": _file_sha(Path(__file__)),
            "declared_sha256": DECLARED_SHA256,
            "invocation": {"argv": argv, "cwd": os.getcwd()},
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"N08_NATIVE_READOUT_REFUSAL_{args.which}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(refusal, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"decision": "refused", "refusal": refusal["refusal"],
                          "native_full_fits_verified": refusal["native_full_fits_verified"],
                          "scientific_readout_count": 0, "receipt": str(path)}))
        return 2
    raise AlignmentError("No native map consumer is implemented; no final-output path is available")


if __name__ == "__main__":
    raise SystemExit(main())
