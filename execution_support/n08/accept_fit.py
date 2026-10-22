"""N06: decide whether one finished N08 replication fit is an accepted unit.

A zero trainer exit is necessary and never sufficient. Acceptance binds the
receipt to the artifacts on disk and to the frozen protocol:

  identity     fold_summary run_id/fold; R07 fingerprint components: split ==
               SPLITS.json trainer_split_component of the fold, seed == init seed,
               partition_seed == 1337, max_retries == 0, monitor, data_rows ==
               the frozen rows.jsonl sha256
  checkpoints  best.pt and last.pt exist; sha256/bytes equal the identities the
               patched trainer recorded (best.pt as scored on the outer fold,
               last.pt as written); size within the declared band
  selection    log.jsonl inner-validation epochs contiguous from 0; replaying the
               selection rule (max of -egfr_logloss, earliest on ties) gives
               best_epoch; best_inner == -logloss[best_epoch]; the patience rule
               (min epochs, patience) reproduces the final epoch; the outer_test
               row carries best_epoch
  outcomes     OOF patients == the fold's frozen outer-test list, fold field == k,
               labels == rows.jsonl, probabilities finite in [0, 1]; not stalled;
               no retry (init_seed_offset 0)
  supervision  grounding_microbatch.jsonl present; header design == the matrix
               row; zero anomalies; microbatch count == epochs x microbatches per
               epoch; weight 0 -> no supervised microbatch; weight > 0 -> some;
               query0 target -> gradient on query 0 only; union -> >= 1 query
  environment  device cpu; torch threads == the declared count
  (2026-10-03, BW/n08 T02/T03/T06/T08 additions)
  success      the trainer's true exit status is 0 (a forged or copied verdict is not
               enough: verify_accepted_receipt re-runs accept() on the files on disk)
  data         fold_summary dataset_version / data_manifest / data_npz_all / data_rows
               components == the configured manifest (re-verified by the caller, full
               npz re-hash, at this acceptance boundary); summary.data_identity agrees
  supervision  counts are REBUILT from the raw microbatch fields (finiteness, expected
               supervision, valid targets, loss weight); NaN/Inf is an anomaly and never
               supervision; every microbatch's patients lie in the fold's inner-train set
               and its has_mask count matches the configured rows
  load identity the encoder digest taken before training (patched trainer) == the
               campaign reference (frozen at launch from the parity run); the
               initial-state digest is recorded for the cross-fit check in the
               runner (same seed -> equal, other seed -> different). NOTE: the
               historical summary.backbone.weight_sha256 is computed after
               training (on the best.pt weights), so it is NOT a load identity.

  (2026-10-04, HX gate-release G07) lineage: verify_accepted_receipt refuses a bootstrap (parity/smoke) identity as a
               scientific fit and, for a non-M0 dataset version, requires the receipt's training_release (file and
               body sha256) to equal the campaign's verified release (expected["training_release"]).

Usage as a library: accept(row, fold_dir, expected) -> verdict dict.
CLI (re-verify an accepted receipt against the files now on disk):
  .venv/bin/python accept_fit.py verify RECEIPT.json [--full]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

SELECTION_MONITOR = '[["egfr_logloss", -1.0]]'
RECEIPT_SCHEMA = "n08.fit_receipt.v2"
COUNT_KEYS = ("microbatches", "supervised", "no_target_by_design", "empty_target", "anomaly_disconnected",
              "anomaly_unexpected_gradient", "anomaly_nonfinite")


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_bootstrap_id(x) -> bool:
    """G07 (HX gate-release): parity (n08parity*) and smoke (*-smoke-*) identities are bootstrap units; they are
    counted in the budgets but can never be accepted as one of the 50 scientific fits."""
    x = str(x or "")
    return x.startswith("n08parity") or "-smoke-" in x or x.endswith("-smoke")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _finite_scalar(x) -> bool:
    """A finite numeric scalar; booleans are not numbers here (NX-D01)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def finite_objective_problems(mbs: list[dict], expected_epochs: int, microbatches_per_epoch: int) -> list[str]:
    """NX-D01 / F02: every EXPECTED training microbatch (all cells, including zero-grounding-weight cells) carries
    finite numeric loss_cls and loss_total, and its recorded loss_total_finite flag equals the raw value's finiteness.
    Completeness is checked on (epoch, step) identities, not just on the rows supplied. Shared by scientific-fit
    acceptance and the smoke consumer (each passes its own expected epoch count)."""
    probs = []
    want = {(e, s) for e in range(int(expected_epochs)) for s in range(int(microbatches_per_epoch))}
    got = [(r.get("epoch"), r.get("step")) for r in mbs]
    if len(got) != len(set(got)):
        probs.append("duplicate microbatch (epoch, step) identities")
    missing, extra = sorted(want - set(got)), sorted(set(got) - want, key=str)
    if missing:
        probs.append(f"missing expected microbatches {missing[:5]} ({len(missing)} total)")
    if extra:
        probs.append(f"unexpected microbatch identities {extra[:5]} ({len(extra)} total)")
    for r in mbs:
        es = (r.get("epoch"), r.get("step"))
        for k in ("loss_cls", "loss_total"):
            if not _finite_scalar(r.get(k)):
                probs.append(f"microbatch {es}: {k} {r.get(k)!r} is not a finite numeric scalar")
        fl = r.get("loss_total_finite")
        if not isinstance(fl, bool) or fl != _finite_scalar(r.get("loss_total")):
            probs.append(f"microbatch {es}: loss_total_finite {fl!r} disagrees with raw loss_total {r.get('loss_total')!r}")
    return probs


def selection_history_problems(inner: list[dict], expected_epochs: int | None = None) -> list[str]:
    """NX-D01 / F03: every expected inner-validation epoch has a finite numeric selection log loss (egfr_logloss).
    A NaN/missing/null/bool/infinite value at ANY epoch (best or not) fails the whole fit; replay never skips it.
    Optional descriptive metrics (e.g. single-class AUC) are not judged here."""
    probs = []
    eps = [m.get("epoch") for m in inner]
    if expected_epochs is not None and eps != list(range(int(expected_epochs))):
        probs.append(f"inner-validation epochs {eps[:3]}..{eps[-3:]} != expected 0..{int(expected_epochs) - 1}")
    for m in inner:
        if not _finite_scalar(m.get("egfr_logloss")):
            probs.append(f"inner-validation epoch {m.get('epoch')}: egfr_logloss {m.get('egfr_logloss')!r} is not finite")
    return probs


# NX F04 / NX-D02: outer-test metrics are recomputed from the bound OOF vector with theia/engine/evaluate.py's
# conventions (labelled rows only, egfr_true in {0,1}; roc_auc_score on raw probabilities; log loss and Brier on
# probabilities clipped to [1e-12, 1-1e-12]). Tolerances are FROZEN before any scientific output (2026-10-05):
# counts and AUC exact; log loss and Brier |diff| <= 1e-12 (float64 summation order only). Positive control: the real
# HX G09 parity fold reproduces all five fields with difference 0.0.
METRIC_TOL = {"auc": 0.0, "n": 0.0, "n_pos": 0.0, "logloss": 1e-12, "brier": 1e-12}


def recompute_outer_metrics(oof: list[dict], gene: str = "egfr") -> dict:
    import numpy as np
    lab = [r for r in oof if r.get(f"{gene}_true") in (0, 1)]
    y = np.asarray([r[f"{gene}_true"] for r in lab], dtype=float)
    p = np.asarray([r[f"{gene}_prob"] for r in lab], dtype=float)
    out = {"n": float(len(y)), "n_pos": float(y.sum())}
    if len(set(y.tolist())) < 2:
        out["auc"] = float("nan")                     # structurally undefined (single class), as evaluate.py records
    else:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p))
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    out["logloss"] = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()) if len(y) else float("nan")
    out["brier"] = float(((pc - y) ** 2).mean()) if len(y) else float("nan")
    return out


def metric_consistency_problems(recomputed: dict, reported: dict, gene: str = "egfr", where: str = "") -> list[str]:
    probs = []
    for k, tol in METRIC_TOL.items():
        a, b = recomputed.get(k), reported.get(f"{gene}_{k}")
        if not isinstance(b, (int, float)) or isinstance(b, bool):
            probs.append(f"{where}{gene}_{k} missing or non-numeric ({b!r})"); continue
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                probs.append(f"{where}{gene}_{k} reported {b!r} but recomputed {a!r}")
        elif abs(a - b) > tol:
            probs.append(f"{where}{gene}_{k} reported {b!r} != recomputed {a!r} (tol {tol})")
    return probs


def classify_microbatch(r: dict, weight: float) -> str:
    """The trainer's status rule, re-derived from raw fields (T06). A record without the
    raw finiteness fields is judged on its gradient norms and losses alone."""
    norms = r.get("grad_norm_by_query")
    fired = norms is not None
    grad_finite = fired and all(_finite(x) for x in norms)
    lg = r.get("loss_ground")
    lg_finite = (lg is None) or _finite(lg)
    if r.get("loss_ground_finite") is False:
        lg_finite = False
    n_valid = int(r.get("n_valid_targets") or 0) if r.get("n_valid_targets") is not None else None
    expected = weight != 0.0 and (n_valid is None or n_valid > 0)
    observed = bool(fired and grad_finite and any(x != 0.0 for x in norms))
    if (fired and not grad_finite) or (weight != 0.0 and not lg_finite):
        return "anomaly_nonfinite"
    if expected and observed:
        return "supervised"
    if not expected and not observed:
        return "no_target_by_design" if weight == 0.0 else "empty_target"
    return "anomaly_disconnected" if expected else "anomaly_unexpected_gradient"


def rebuild_diagnostics(mbs: list[dict], weight: float, target: str) -> tuple[dict, list[str]]:
    counts = {k: 0 for k in COUNT_KEYS}
    qwg: set[int] = set()
    q0_only = True
    problems = []
    for r in mbs:
        st = classify_microbatch(r, weight)
        if r.get("status") != st:
            problems.append(f"epoch {r.get('epoch')} step {r.get('step')}: recorded status {r.get('status')!r} "
                            f"!= rebuilt {st!r}")
        counts["microbatches"] += 1
        counts[st] += 1
        if st == "supervised":
            nz = [i for i, x in enumerate(r["grad_norm_by_query"]) if x != 0.0]
            qwg.update(nz)
            if nz != [0]:
                q0_only = False
    counts["anomalies"] = counts["anomaly_disconnected"] + counts["anomaly_unexpected_gradient"] + \
        counts["anomaly_nonfinite"]
    counts["queries_with_gradient"] = sorted(qwg)
    counts["supervised_gradient_on_query0_only"] = q0_only if counts["supervised"] else None
    return counts, problems


def file_identity(p: Path) -> dict | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def replay_selection(inner: list[dict], min_epochs: int, patience_n: int) -> dict:
    """theia.engine.train's Selector (no gate) + patience loop, replayed on logged metrics."""
    best, best_epoch, sel_best, patience, stop = -math.inf, -1, -math.inf, 0, None
    for m in inner:
        e = int(m["epoch"])
        v = -float(m["egfr_logloss"]) if m.get("egfr_logloss") is not None else float("nan")
        if not math.isnan(v) and v > sel_best:
            sel_best, best_epoch = v, e
        if not math.isnan(v) and v > best:
            best, patience = v, 0
        else:
            if e + 1 < min_epochs:
                continue
            patience += 1
            if patience >= patience_n:
                stop = e
                break
    return {"best_epoch": best_epoch, "best_inner": sel_best, "stopped_at": stop}


def accept(row: dict, fold_dir: Path, expected: dict, trainer_exit: int | None = None) -> dict:
    """row: FIT_MATRIX row (dict); expected: splits, rows_labels, rows_sha256, threads,
    encoder_sha256, ckpt_bytes, ckpt_bytes_tol, min_epochs, patience, max_epochs,
    microbatches_per_epoch, partition_seed; with a data manifest also data (the identity
    the caller re-verified at this boundary: dataset_version, manifest_body_sha256,
    rows_sha256, npz_all_sha256) and rows_has_mask. trainer_exit: the true exit status."""
    checks: list[dict] = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    fold = int(row["fold"])
    init_seed = int(row["init_seed"])
    weight = float(row["ground_weight"])
    target = row["ground_target"]
    split_fold = expected["splits"]["folds"][fold]
    ident: dict = {"best_pt": file_identity(fold_dir / "best.pt"), "last_pt": file_identity(fold_dir / "last.pt"),
                   "fold_summary": file_identity(fold_dir / "fold_summary.json"),
                   "log_jsonl": file_identity(fold_dir / "log.jsonl"),
                   "grounding_microbatch_jsonl": file_identity(fold_dir / "grounding_microbatch.jsonl"),
                   "attempts_jsonl": file_identity(fold_dir / "attempts.jsonl")}
    out = {"fit_id": row["fit_id"], "fold_dir": str(fold_dir), "identities": ident, "checks": checks}

    if expected.get("require_trainer_exit", True) or trainer_exit is not None:
        check("trainer exit status == 0", trainer_exit == 0, trainer_exit)
    if not check("fold_summary.json exists", ident["fold_summary"] is not None):
        out["accepted"] = False
        out["failed_checks"] = [c["check"] for c in checks if not c["ok"]]
        return out
    fs = json.loads((fold_dir / "fold_summary.json").read_text())
    s = fs.get("summary", {})
    comps = fs.get("fingerprint_components", {})
    check("run_id matches", fs.get("run_id") == row["run_id"], fs.get("run_id"))
    check("fold matches", fs.get("fold") == fold and s.get("fold") == fold)
    check("fingerprint present", bool(fs.get("fingerprint")))
    check("split component == frozen fold split", comps.get("split") == split_fold["trainer_split_component"],
          comps.get("split"))
    check("seed component == init seed", comps.get("seed") == init_seed, comps.get("seed"))
    check("partition_seed component == 1337", comps.get("partition_seed") == expected["partition_seed"],
          comps.get("partition_seed"))
    check("max_retries component == 0", comps.get("max_retries") == 0, comps.get("max_retries"))
    check("monitor component == egfr_logloss selection", comps.get("monitor") == SELECTION_MONITOR,
          comps.get("monitor"))
    check("data_rows component == configured rows sha256", comps.get("data_rows") == expected["rows_sha256"],
          comps.get("data_rows"))
    data = expected.get("data")
    if data is not None:
        check("dataset_version component == configured", comps.get("dataset_version") == data["dataset_version"],
              comps.get("dataset_version"))
        check("data_manifest component == configured manifest_body_sha256",
              comps.get("data_manifest") == data["manifest_body_sha256"], comps.get("data_manifest"))
        check("data_npz_all component == configured npz identities", comps.get("data_npz_all") == data["npz_all_sha256"],
              comps.get("data_npz_all"))
        sdi = s.get("data_identity") or {}
        check("summary.data_identity == configured (verified before and after the fold)",
              all(sdi.get(k) == data[k] for k in ("dataset_version", "manifest_body_sha256", "rows_sha256",
                                                  "npz_all_sha256")) and bool(sdi.get("reverified_utc")),
              {k: sdi.get(k) for k in ("dataset_version", "manifest_body_sha256", "reverified_utc")})
        check("data re-verified at this acceptance boundary (full npz re-hash)",
              bool(data.get("npz_rehashed")) and bool(data.get("verified_utc")), data.get("verified_utc"))
    check("not stalled", not s.get("stalled"), s.get("stalled"))
    led = _jsonl(fold_dir / "attempts.jsonl") if ident["attempts_jsonl"] is not None else []
    att_n = fs.get("attempt")
    att_rows = [r for r in led if r.get("attempt") == att_n]
    check("trainer ledger: the summary's attempt completed and was not failed/stopped",
          att_n is not None and any(r.get("status") == "complete" for r in att_rows)
          and not any(r.get("status") in ("failed", "stopped_cooperative", "interrupted") for r in att_rows),
          {"attempt": att_n, "statuses": [r.get("status") for r in att_rows]})
    check("no retry (init_seed_offset 0)", int(s.get("init_seed_offset", 0)) == 0, s.get("init_seed_offset"))
    check("outer test metrics present", bool(s.get("test")))
    check("not truncated (no train.stop_after_epochs)",
          expected.get("allow_truncation", False) or "stop_after_epochs" not in s, s.get("stop_after_epochs"))

    # ---- checkpoints
    for name, key in (("best.pt", "best_pt_scored"), ("last.pt", "last_pt")):
        idn = ident["best_pt" if name == "best.pt" else "last_pt"]
        if check(f"{name} exists", idn is not None):
            check(f"{name} identity == trainer-recorded identity", s.get(key) == idn,
                  {"recorded": s.get(key), "on_disk": idn})
            lo, hi = expected["ckpt_bytes"] - expected["ckpt_bytes_tol"], expected["ckpt_bytes"] + expected["ckpt_bytes_tol"]
            check(f"{name} size within declared band", lo <= idn["bytes"] <= hi, idn["bytes"])

    # ---- selection and stopping
    sel = {}
    if check("log.jsonl exists", ident["log_jsonl"] is not None):
        log = _jsonl(fold_dir / "log.jsonl")
        inner = [m for m in log if m.get("split") == "inner_val"]
        ot = [m for m in log if m.get("split") == "outer_test"]
        epochs = [int(m["epoch"]) for m in inner]
        check("inner-validation epochs contiguous from 0", epochs == list(range(len(epochs))) and epochs, epochs[-1:] or None)
        sh = selection_history_problems(inner)
        check("NX-D01 every inner-validation selection log loss finite (any invalid epoch fails the fit)", not sh, sh[:5])
        sel = replay_selection(inner, expected["min_epochs"], expected["patience"])
        check("replayed selection == best_epoch", sel["best_epoch"] == s.get("best_epoch"),
              {"replayed": sel["best_epoch"], "recorded": s.get("best_epoch")})
        check("best_inner == -logloss[best_epoch]",
              s.get("best_inner") is not None and abs(sel["best_inner"] - float(s["best_inner"])) <= 1e-12,
              {"replayed": sel["best_inner"], "recorded": s.get("best_inner")})
        final = epochs[-1] if epochs else None
        stop_ok = (sel["stopped_at"] == final) if sel["stopped_at"] is not None else (final == expected["max_epochs"] - 1)
        check("patience rule reproduces the final epoch", stop_ok, {"replayed_stop": sel["stopped_at"], "final": final})
        check("final_epoch recorded == logged", s.get("final_epoch") == final, s.get("final_epoch"))
        check("epochs_run recorded == logged", s.get("epochs_run") == len(epochs), s.get("epochs_run"))
        check("epochs run <= max_epochs", len(epochs) <= expected["max_epochs"], len(epochs))
        if expected.get("allow_truncation"):                 # NX VA-02: the smoke contract lives in acceptance too
            check("truncated run is exactly the declared budget (stop_after_epochs == epochs_run == max_epochs)",
                  s.get("stop_after_epochs") == expected["max_epochs"] == len(epochs)
                  and not isinstance(s.get("stop_after_epochs"), bool),
                  {"stop_after_epochs": s.get("stop_after_epochs"), "epochs_run": len(epochs),
                   "max_epochs": expected["max_epochs"]})
        check("one outer_test row at best_epoch", len(ot) == 1 and ot[0].get("epoch") == s.get("best_epoch"))
        sel.update({"final_epoch": final, "epochs_run": len(epochs),
                    "inner_logloss_at_best": (-float(s["best_inner"]) if s.get("best_inner") is not None else None),
                    "inner_auc_at_best": next((m.get("egfr_auc") for m in inner if m["epoch"] == s.get("best_epoch")), None)})

    # ---- outcomes
    oof = s.get("oof") or []
    ids = sorted(r["patient_id"] for r in oof)
    check("OOF patients == frozen outer test", ids == split_fold["outer_test"]["patient_ids"], len(ids))
    check("OOF fold field == fold", all(int(r["fold"]) == fold for r in oof))
    lab = expected["rows_labels"]
    check("OOF labels == rows.jsonl", all(int(r["egfr_true"]) == lab[r["patient_id"]] for r in oof if r["patient_id"] in lab))
    check("OOF probabilities finite in [0,1]",
          all(isinstance(r.get("egfr_prob"), (int, float)) and math.isfinite(r["egfr_prob"]) and 0 <= r["egfr_prob"] <= 1
              for r in oof) and bool(oof))

    # ---- NX F04: recomputed outer-test metrics must equal every reported copy
    rcm = recompute_outer_metrics(oof)
    mp = metric_consistency_problems(rcm, s.get("test") or {}, where="summary.test: ")
    if sel and ident.get("log_jsonl") is not None:
        ot_rows = [m for m in _jsonl(fold_dir / "log.jsonl") if m.get("split") == "outer_test"]
        if len(ot_rows) == 1:
            mp += metric_consistency_problems(rcm, ot_rows[0], where="log outer_test: ")
    out["recomputed_outer_metrics"] = rcm
    check("NX-D02 outer-test AUC/logloss/Brier/counts recomputed from bound OOF == reported", not mp, mp[:5])

    # ---- supervision path
    gd = s.get("grounding_diagnostics") or {}
    diag_head = {}
    rebuilt: dict = {}
    if check("grounding_microbatch.jsonl exists", ident["grounding_microbatch_jsonl"] is not None):
        recs = _jsonl(fold_dir / "grounding_microbatch.jsonl")
        diag_head = recs[0] if recs and recs[0].get("kind") == "header" else {}
        mbs = [r for r in recs if r.get("kind") == "microbatch"]
        rebuilt, status_probs = rebuild_diagnostics(mbs, weight, target)
        out["grounding_diagnostics_rebuilt"] = rebuilt
        if expected.get("data") is not None:
            check("diagnostics schema 2 (raw finiteness fields present)", diag_head.get("diagnostics_schema") == 2,
                  diag_head.get("diagnostics_schema"))
        check("every recorded status == the status rebuilt from raw fields", not status_probs, status_probs[:5])
        check("rebuilt counts == trainer summary counts",
              all(rebuilt.get(k) == gd.get(k) for k in COUNT_KEYS + ("anomalies",)
                  if k != "anomaly_nonfinite" or k in gd)
              and rebuilt.get("queries_with_gradient") == gd.get("queries_with_gradient"),
              {k: (rebuilt.get(k), gd.get(k)) for k in COUNT_KEYS + ("anomalies",)})
        inner = set(split_fold["inner_train"]["patient_ids"])
        stray = sorted({p for r in mbs for p in (r.get("patient_ids") or []) if p not in inner})
        check("microbatch patients within the fold's inner-train set",
              all(r.get("patient_ids") is not None for r in mbs) and not stray, stray[:5])
        hm = expected.get("rows_has_mask")
        if hm is not None:
            bad = [(r.get("epoch"), r.get("step")) for r in mbs
                   if r.get("n_has_mask") != sum(1 for p in r.get("patient_ids") or [] if hm.get(p))
                   or int(r.get("n_valid_targets") or 0) > int(r.get("n_has_mask") or 0)]
            check("microbatch has_mask / valid-target counts consistent with the configured rows", not bad, bad[:5])
        check("diagnostics header design == matrix row",
              diag_head.get("ground_weight") == weight and diag_head.get("ground_target") == target
              and diag_head.get("region_pooling") == row["region_pooling"]
              and diag_head.get("init_seed") == init_seed
              and diag_head.get("partition_seed") == expected["partition_seed"], diag_head)
        check("diagnostics summary == fold result", recs[-1].get("kind") == "summary"
              and {k: v for k, v in recs[-1].items() if k != "kind"} == gd)
        check("zero grounding anomalies (rebuilt; NaN/Inf included)", rebuilt.get("anomalies") == 0
              and gd.get("anomalies") == 0, {"rebuilt": rebuilt.get("anomalies"), "summary": gd.get("anomalies")})
        fo = finite_objective_problems(mbs, sel.get("epochs_run", 0), expected["microbatches_per_epoch"])
        check("NX-D01 every expected microbatch loss_cls/loss_total finite and flag-consistent", not fo, fo[:5])
        n_exp = sel.get("epochs_run", 0) * expected["microbatches_per_epoch"]
        check("microbatch count == epochs x microbatches/epoch", len(mbs) == rebuilt.get("microbatches") == n_exp,
              {"logged": len(mbs), "expected": n_exp})
        if weight == 0.0:
            check("weight 0: no supervised microbatch", rebuilt.get("supervised") == 0 and
                  rebuilt.get("no_target_by_design") == rebuilt.get("microbatches"))
        else:
            check("weight > 0: supervision reached the maps", (rebuilt.get("supervised") or 0) > 0)
            if target == "query0":
                check("query0 target: gradient on query 0 only",
                      rebuilt.get("supervised_gradient_on_query0_only") is True)
            else:
                check("union target: gradient on >= 1 query", len(rebuilt.get("queries_with_gradient") or []) >= 1)

    # ---- environment and parameter-load identity
    check("device cpu", diag_head.get("device") == "cpu", diag_head.get("device"))
    check("torch threads == declared", diag_head.get("torch_num_threads") == expected["threads"],
          diag_head.get("torch_num_threads"))
    pli = s.get("parameter_load_identity") or {}
    enc = pli.get("encoder_load_sha256")
    check("parameter-load identity recorded", bool(enc) and bool(pli.get("initial_state_sha256")), pli)
    if expected.get("establish_encoder_reference"):
        check("encoder load digest recorded (this run establishes the reference)", bool(enc), enc)
    else:
        check("encoder load digest == campaign reference",
              bool(expected.get("encoder_sha256")) and enc == expected["encoder_sha256"],
              {"recorded": enc, "reference": expected.get("encoder_sha256")})

    out["selection"] = sel
    out["environment"] = {"device": diag_head.get("device"), "torch": diag_head.get("torch"),
                          "torch_num_threads": diag_head.get("torch_num_threads"),
                          "torch_num_interop_threads": diag_head.get("torch_num_interop_threads"),
                          "env_threads": diag_head.get("env_threads"), "encoder_load_sha256": enc,
                          "initial_state_sha256": pli.get("initial_state_sha256"),
                          "post_training_backbone_digest": (s.get("backbone") or {}).get("weight_sha256"),
                          "trainer_env_component": comps.get("env")}
    out["fingerprint"] = {"trainer_fingerprint": fs.get("fingerprint"), "components": comps}
    out["grounding_diagnostics"] = gd
    out["oof_n"] = len(oof)
    out["accepted"] = all(c["ok"] for c in checks)
    out["failed_checks"] = [c["check"] for c in checks if not c["ok"]]
    return out


def fit_identity(row: dict, verdict: dict, expected: dict) -> dict:
    """T03: everything one fit's result is a function of, in one hashable record."""
    fp = verdict.get("fingerprint") or {}
    comps = fp.get("components") or {}
    env = verdict.get("environment") or {}
    sf = expected["splits"]["folds"][int(row["fold"])]
    data = expected.get("data") or {}
    return {"dataset_version": data.get("dataset_version"), "manifest_body_sha256": data.get("manifest_body_sha256"),
            "rows_sha256": expected.get("rows_sha256"), "npz_all_sha256": data.get("npz_all_sha256"),
            "annotation_preprocessing_sha256": data.get("annotation_preprocessing_sha256"),
            "split_component": comps.get("split"),
            "membership_sha256": {k: sf[k]["sha256_patient_ids"] for k in ("inner_train", "inner_val", "outer_test")},
            "initial_state_sha256": env.get("initial_state_sha256"), "encoder_load_sha256": env.get("encoder_load_sha256"),
            "selected_backbone_sha256": env.get("post_training_backbone_digest"),
            "code_component": comps.get("code"), "config_component": comps.get("config"),
            "trainer_fingerprint": fp.get("trainer_fingerprint"),
            "init_seed": int(row["init_seed"]), "partition_seed": int(expected["partition_seed"]),
            "objective": {"region_pooling": row.get("region_pooling"), "ground_weight": float(row["ground_weight"]),
                          "ground_target": row.get("ground_target"), "readout_map": row.get("readout_map")},
            "selection_rule": {"monitor": SELECTION_MONITOR, "min_epochs": expected["min_epochs"],
                               "patience": expected["patience"], "max_epochs": expected["max_epochs"],
                               "ties": "earliest"},
            "best_pt_sha256": (verdict.get("identities", {}).get("best_pt") or {}).get("sha256"),
            "last_pt_sha256": (verdict.get("identities", {}).get("last_pt") or {}).get("sha256")}


def verify_accepted_receipt(acc_path, expected: dict, row: dict) -> list[str]:
    """T08: is this ACCEPTED.json a real accepted unit NOW? Re-runs accept() on the files on
    disk (full sha256) with the caller's freshly re-verified data identity, and requires
    the stored verdict, identities and fit identity to be reproduced exactly."""
    acc_path = Path(acc_path)
    if not acc_path.exists():
        return ["ACCEPTED.json missing"]
    try:
        acc = json.loads(acc_path.read_text())
    except ValueError as exc:
        return [f"ACCEPTED.json unreadable: {exc}"]
    probs = []
    if acc.get("kind") != expected.get("unit_kind", "fit"):
        probs.append(f"kind {acc.get('kind')!r} is not a {expected.get('unit_kind', 'fit')} unit")
    if acc.get("schema") != RECEIPT_SCHEMA:
        probs.append(f"schema {acc.get('schema')!r} != {RECEIPT_SCHEMA}")
    if acc.get("status") != "accepted" or not (acc.get("acceptance") or {}).get("accepted"):
        probs.append(f"status {acc.get('status')!r} is not an accepted verdict")
    if acc.get("trainer_exit") != 0:
        probs.append(f"trainer exit {acc.get('trainer_exit')!r} != 0")
    if acc_path.parent.name != row["fit_id"] or acc.get("fit_id") != row["fit_id"]:
        probs.append(f"receipt fit_id {acc.get('fit_id')!r} in {acc_path.parent.name!r} != matrix {row['fit_id']!r}")
    for k in ("run_id", "cell"):
        if acc.get(k) != row.get(k):
            probs.append(f"{k} {acc.get(k)!r} != matrix {row.get(k)!r}")
    for k in ("fold", "init_seed"):
        if acc.get(k) is None or int(acc.get(k)) != int(row[k]):
            probs.append(f"{k} {acc.get(k)!r} != matrix {row[k]!r}")
    rp = Path(acc.get("receipt_path") or "")
    if not rp.exists():
        probs.append("attempt receipt missing")
    elif file_identity(rp)["sha256"] != acc.get("receipt_sha256"):
        probs.append("attempt receipt changed since acceptance")
    else:
        att = json.loads(rp.read_text())
        if {k: v for k, v in acc.items() if k not in ("receipt_path", "receipt_sha256")} != att:
            probs.append("ACCEPTED.json body differs from its attempt receipt")
    if is_bootstrap_id(row["fit_id"]) or is_bootstrap_id(row.get("run_id", "")) or is_bootstrap_id(acc.get("fit_id", "")):
        probs.append("a bootstrap (parity/smoke) identity is never a scientific matrix fit")       # G07
    data = expected.get("data") or {}
    if "training_release" in expected and data.get("dataset_version") not in (None, "M0"):          # G07 lineage
        want, got = expected.get("training_release") or {}, acc.get("training_release") or {}
        if not want:
            probs.append("the campaign has no verified training release for a non-M0 dataset version")
        elif got.get("sha256") != want.get("sha256") or got.get("release_body_sha256") != want.get("release_body_sha256"):
            probs.append("the fit was accepted under a different (or no) training release than the campaign's")
    if (acc.get("data") or {}).get("manifest_body_sha256") != data.get("manifest_body_sha256") or \
            (acc.get("data") or {}).get("dataset_version") != data.get("dataset_version"):
        probs.append(f"receipt data {((acc.get('data') or {}).get('dataset_version'))} / "
                     f"{str((acc.get('data') or {}).get('manifest_body_sha256'))[:12]} != configured "
                     f"{data.get('dataset_version')} / {str(data.get('manifest_body_sha256'))[:12]}")
    fold_dir = Path((acc.get("acceptance") or {}).get("fold_dir") or "")
    v = accept(row, fold_dir, expected, trainer_exit=acc.get("trainer_exit"))
    if not v["accepted"]:
        probs.append(f"re-acceptance fails now: {v['failed_checks']}")
    if v.get("identities") != (acc.get("acceptance") or {}).get("identities"):
        probs.append("artifact identities on disk differ from the accepted receipt")
    fi = fit_identity(row, v, expected)
    stored = acc.get("fit_identity") or {}
    if canon_sha(fi) != acc.get("fit_identity_sha256") or canon_sha(stored) != acc.get("fit_identity_sha256"):
        diff = sorted(k for k in set(fi) | set(stored) if fi.get(k) != stored.get(k))
        probs.append(f"fit identity differs ({diff})")
    return probs


def campaign_duplicates(accepted: dict) -> list[str]:
    """accepted: {fit_id: receipt}. Two accepted records may not share a fold dir, a
    best.pt or a fit identity."""
    probs, seen = [], {}
    for fid, acc in accepted.items():
        a = acc.get("acceptance") or {}
        keys = {"fold_dir": a.get("fold_dir"), "best_pt": ((a.get("identities") or {}).get("best_pt") or {}).get("sha256"),
                "fit_identity": acc.get("fit_identity_sha256")}
        for k, v in keys.items():
            if v is None:
                continue
            if (k, v) in seen:
                probs.append(f"duplicate fit record: {fid} and {seen[(k, v)]} share {k} {str(v)[:16]}")
            else:
                seen[(k, v)] = fid
    return probs


def verify_receipt(receipt: dict, full: bool = False) -> list[str]:
    """Problems with an accepted receipt's artifacts as they are on disk now."""
    probs = []
    acc = receipt.get("acceptance", {})
    fold_dir = Path(acc.get("fold_dir", ""))
    for name, key in (("best.pt", "best_pt"), ("last.pt", "last_pt"), ("fold_summary.json", "fold_summary"),
                      ("log.jsonl", "log_jsonl"), ("grounding_microbatch.jsonl", "grounding_microbatch_jsonl")):
        want = acc.get("identities", {}).get(key)
        p = fold_dir / name
        if want is None:
            continue
        if not p.exists():
            probs.append(f"{name} missing")
            continue
        if p.stat().st_size != want["bytes"]:
            probs.append(f"{name} size {p.stat().st_size} != {want['bytes']}")
        elif full and file_identity(p)["sha256"] != want["sha256"]:
            probs.append(f"{name} sha256 changed")
    return probs


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        rc = json.loads(Path(sys.argv[2]).read_text())
        pr = verify_receipt(rc, full="--full" in sys.argv)
        print(json.dumps({"receipt": sys.argv[2], "problems": pr}, indent=1))
        raise SystemExit(1 if pr else 0)
    raise SystemExit(__doc__)
