"""Training loop — k-fold aware, LoRA fine-tune, AMP, early stopping.

Run: python -m theia.engine.train --config configs/default.yaml
Trains one model per fold and writes checkpoints + a metrics log per fold.

Fixed here:
  * Device selection was `cuda if available else cpu`, which ignored Apple
    Silicon entirely. Now goes through theia.runtime (cuda > mps > cpu).
  * `torch.cuda.amp` was requested unconditionally; off CUDA it disables itself
    with a warning, so `amp: true` was quietly a lie. AMP is now device-aware and
    the resolved state is printed.
  * A NaN monitor (single-class validation fold) made `nan > best` False for every
    epoch, so `best.pt` was NEVER written for that fold, `best` stayed -1.0 and
    polluted the CV mean, and run_eval.sh died on a missing checkpoint. NaN is now
    handled explicitly and a checkpoint is always written.
  * The scheduler stepped even when the GradScaler skipped the optimizer step.
  * The CV summary printed (max-min)/2 as "±", which is a half-range, not a
    standard deviation or a confidence interval.
  * Model selection ran on the same split that was reported. Nested splits are
    now the default: early stopping uses an inner validation set, and the outer
    fold is scored once, after the model is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from theia.config import load_config
from theia.data.dataset import (RadiogenomicsDataset, kfold_indices, make_collate,
                                nested_kfold_indices)
from theia.engine.evaluate import collect_predictions, evaluate, pooled_metrics
from theia.engine.losses import total_loss
from theia.models.theia_model import Theia
from theia.runtime import amp_settings, autocast, describe, make_grad_scaler, resolve_device


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def partition_seed(cfg) -> int:
    """Seed of the outer and inner patient partition.

    `seed` used to drive both the partition (nested_kfold_indices) and the
    initialisation (set_seed), so "a new seed" also re-partitioned the patients
    and a seed replication could not hold the folds fixed. `split.seed` now sets
    the partition alone. Absent -> cfg.seed, so every existing configuration
    partitions exactly as before; tests pin that byte-for-byte.
    """
    split = cfg.get("split") or {}
    v = split.get("seed") if isinstance(split, dict) else None
    if v is None:
        return int(cfg.seed)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"split.seed must be an integer, got {v!r}")
    return int(v)


def _loader(cfg, rows, genes, idx, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = RadiogenomicsDataset(rows, genes, idx)
    workers = int(getattr(cfg.train, "num_workers", 2))
    return DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
        collate_fn=make_collate(genes), num_workers=workers,
        drop_last=drop_last and len(ds) > cfg.train.batch_size,
        persistent_workers=workers > 0,
    )


def make_loaders(cfg, train_idx, val_idx, test_idx=None):
    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    genes = cfg.data.target_genes
    tl = _loader(cfg, rows, genes, train_idx, shuffle=True, drop_last=True)
    vl = _loader(cfg, rows, genes, val_idx, shuffle=False, drop_last=False)
    testl = _loader(cfg, rows, genes, test_idx, shuffle=False, drop_last=False) if test_idx else None
    return tl, vl, testl


def _is_better(candidate: float, best: float) -> bool:
    """NaN is 'no signal', never an improvement — and never a permanent block."""
    return not math.isnan(candidate) and candidate > best


def monitor_value(metrics: dict, monitor) -> float:
    """Resolve train.monitor, which may be one key or several to be summed.

    Selecting on `egfr_auc` alone and then *reporting* grounding means the
    grounding number comes from whichever epoch happened to win on AUC.
    Measured: fold 1 selected epoch 4 (lift +0.007) while its best grounding sat
    at epoch 12 (+0.069). A composite monitor keeps an epoch that does both.

    Accepts "egfr_auc", ["egfr_auc", "grounding_mass_lift"], or
    [["egfr_auc", 1.0], ["grounding_mass_lift", 0.5]].
    """
    if isinstance(monitor, str):
        return float(metrics.get(monitor, float("nan")))
    total = 0.0
    for item in monitor:
        key, weight = (item, 1.0) if isinstance(item, str) else (item[0], float(item[1]))
        v = metrics.get(key, float("nan"))
        if v != v:                      # NaN in any component invalidates the sum
            return float("nan")
        total += weight * float(v)
    return total


def monitor_gate_ok(metrics: dict, gate) -> bool:
    """Whether an epoch is eligible to be *deployed* at all.

    A weighted sum cannot express "must localise". Inner validation is 24-31
    patients, so egfr_auc carries SE ~0.11, and a lucky draw at an epoch where
    the grounding head has learned nothing outscores a trained epoch that both
    classifies and points. Measured, on the 20-seed stability set: fold s101/4
    selected epoch 0 (auc 0.889, lift +0.000, pointing 0.000) over its own
    epoch 13 (auc 0.639, lift +0.374, pointing 1.000), because
    1.0*0.889 > 1.0*0.639 + 0.5*0.374. The deployed checkpoint then scored
    0.005 pointing on 420 external patients. s108/1 failed the same way.

    The warmup carve-out in the patience branch below already names this exact
    draw ("letting that draw set the bar is selection on noise") but only kept
    it from resetting *patience* -- such an epoch could still be saved as
    best.pt and shipped.

    Raising the grounding weight is the wrong instrument: across the 60
    archived folds, w=1.0 rescues both folds but perturbs 15 selections and
    costs 0.026 mean inner AUC, while this gate rescues both and perturbs 3 at
    a cost of 0.010.

    `gate` is None (unconstrained, the pre-fix behaviour) or [key, threshold].

    The key is grounding_POINTING, not grounding_mass_lift, and that choice was
    forced by data rather than taste. Mass lift is continuous and noisy, so the
    rescue survives only 0.01..0.08 -- stab-s116/fold4 first localises at its
    final epoch with lift +0.088, and a gate at 0.10 excludes it and re-ships
    the pointing-0.000 checkpoint. Pointing is bimodal (0.000, or 0.84 and up),
    so the same rescue survives 0.01..0.50, a fiftyfold band. Pointing is also
    the quantity the constraint is actually about: did the attention peak land
    inside the tumour.

    0.10 sits mid-band. With 24-31 inner-validation patients one patient is
    0.042 and two are 0.083, so 0.10 requires at least three and cannot be
    cleared by a single lucky peak.
    """
    if gate is None:
        return True
    key, thr = (gate, 0.0) if isinstance(gate, str) else (gate[0], float(gate[1]))
    v = metrics.get(key, float("nan"))
    return v == v and float(v) > float(thr)


class Selector:
    """Which epoch ships.

    Extracted from the training loop so the rule can be replayed over archived
    log.jsonl files -- both to prove `gate=None` reproduces every historical
    selection bit-for-bit, and to measure a candidate gate without retraining.
    A test that reimplemented the rule would drift from it; this one cannot.
    """

    def __init__(self, gate=None):
        self.gate = gate
        self.best = -float("inf")
        self.best_epoch = -1
        self.seen_eligible = False

    def offer(self, metrics: dict, monitor: float, epoch: int) -> bool:
        """Return True if this epoch becomes the deployed checkpoint."""
        eligible = monitor_gate_ok(metrics, self.gate)
        if eligible and not self.seen_eligible:
            # The first localising epoch supersedes every non-localising one,
            # however well it scored. Without this reset a lucky early AUC keeps
            # the crown for the rest of the fold, which is the bug itself.
            self.seen_eligible, self.best = True, -float("inf")
        if (eligible or not self.seen_eligible) and _is_better(monitor, self.best):
            self.best, self.best_epoch = monitor, epoch
            return True
        return False


def _param_groups(model, cfg) -> list[dict]:
    """Discriminative learning rates: pretrained vision blocks train slower.

    With `unfreeze_last_n`, the re-enabled ViT blocks carry pretrained weights but
    were getting the same LR as randomly-initialised heads. That is far too high
    for them and is the likely cause of grounding's fold-to-fold instability —
    measured mass lift ranged 0.000 to 0.217 across five folds on identical
    settings, and the same fold swung 0.302 -> 0.004 between runs.

    Vision parameters get lr * train.backbone_lr_mult; heads keep the full rate.
    """
    mult = float(getattr(cfg.train, "backbone_lr_mult", 1.0))
    vision, heads = [], []
    vision_ids = {id(p) for p in model.vision.parameters()}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (vision if id(p) in vision_ids else heads).append(p)
    if not vision or mult == 1.0:
        return [{"params": vision + heads, "lr": cfg.train.lr}]
    return [
        {"params": vision, "lr": cfg.train.lr * mult},
        {"params": heads, "lr": cfg.train.lr},
    ]


def _file_identity(path: Path) -> dict:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def _atomic_torch_save(obj, path: Path, max_bytes: int | None = None) -> None:
    """T13: a checkpoint on disk is either the previous complete file or the new one.

    torch.save names the zip's top-level record after the file's stem, so the temporary
    file keeps the final NAME inside a private sibling directory (same filesystem) and is
    then renamed over the target: the bytes equal a direct torch.save(path), and a crash
    mid-write leaves only .partial-<pid>/<name>, which nothing reads as a checkpoint.

    Rolling admission (EC/resource): the transient peak of a rewrite is the old file plus
    this temporary. With `max_bytes` (the admitted write forecast), a temporary larger than
    the forecast is NOT renamed over the target: the previous checkpoint stays, the
    temporary stays in .partial-<pid>/ (preserved by begin_attempt) and a cooperative stop
    is raised. The admission already reserved stop headroom beyond the forecast."""
    path = Path(path)
    tmp_dir = path.parent / f".partial-{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / path.name
    torch.save(obj, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    if max_bytes is not None and tmp.stat().st_size > int(max_bytes):
        raise _CooperativeStop({"reason": "checkpoint_exceeds_forecast",
                                "reasons": [{"reason": "checkpoint_exceeds_forecast",
                                             "detail": {"target": path.name, "bytes": tmp.stat().st_size,
                                                        "forecast_bytes": int(max_bytes), "temporary": str(tmp)}}]})
    os.replace(tmp, path)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: Path, obj) -> None:
    _write_text_atomic(path, json.dumps(obj, indent=1, default=str) + "\n")


EX_DATA_REFUSED = 78     # the configured data manifest does not match the bytes this process would read
EX_COOP_STOP = 86        # cooperative within-fit stop at a safe unit (T13): not a failure, not a result


def _canon_sha(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _DataRefused(RuntimeError):
    pass


def _verify_content_manifest(manifest_path: str, rows_path: str, expected_version) -> dict:
    """T02/T03: bind the bytes this process will read to the configured data manifest
    (BW/input_repair/M1_SCHEMA.md). Paths resolve against the working directory, which is
    where RadiogenomicsDataset opens each row's npz. Every npz is re-hashed. Raises
    _DataRefused naming the problems; matching patient ids never suffice."""
    import hashlib

    def fsha(p: Path):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest(), p.stat().st_size

    probs = []
    mp, rp = Path(manifest_path), Path(rows_path)
    if not mp.exists() or not rp.exists():
        raise _DataRefused(f"missing {mp if not mp.exists() else rp}")
    m = json.loads(mp.read_text())
    if _canon_sha({k: v for k, v in m.items() if k != "manifest_body_sha256"}) != m.get("manifest_body_sha256"):
        probs.append("manifest_body_sha256 does not match the manifest body")
    if not expected_version or m.get("dataset_version") != expected_version:
        probs.append(f"dataset_version {m.get('dataset_version')!r} != configured data.dataset_version "
                     f"{expected_version!r}")
    rows_bytes = rp.read_bytes()
    rows_sha = hashlib.sha256(rows_bytes).hexdigest()
    if rows_sha != m.get("rows_sha256"):
        probs.append(f"rows file {rp} is not the manifest's rows (sha256 {rows_sha[:12]})")
    if Path(str(m.get("rows_path", ""))).resolve() != rp.resolve():
        probs.append(f"rows file read ({rp}) is not the manifest rows_path ({m.get('rows_path')})")
    rows = [json.loads(x) for x in rows_bytes.decode().splitlines() if x.strip()]
    ents = {e.get("patient_id"): e for e in m.get("entries", [])}
    if not (len(rows) == len(ents) == len(m.get("entries", [])) == m.get("n_rows")):
        probs.append("rows and manifest entries are not 1:1")
    triples = []
    for r in rows:
        e = ents.get(r.get("patient_id"))
        if e is None:
            probs.append(f"row {r.get('patient_id')} has no manifest entry")
            continue
        if str(e.get("m1_status", "")).startswith("unresolved_"):
            probs.append(f"{r['patient_id']}: blocking m1_status {e['m1_status']}")
        if os.path.normpath(r["npz"]) != os.path.normpath(e.get("npz_path", "")):
            probs.append(f"{r['patient_id']}: row npz {r['npz']} != manifest npz_path {e.get('npz_path')}")
            continue
        if not Path(r["npz"]).exists():
            probs.append(f"{r['patient_id']}: npz missing")
            continue
        sh, nb = fsha(Path(r["npz"]))
        if sh != e.get("npz_sha256") or nb != e.get("npz_bytes"):
            probs.append(f"{r['patient_id']}: npz bytes differ from the manifest")
        triples.append([r["patient_id"], os.path.normpath(e["npz_path"]), e["npz_sha256"]])
    if probs:
        raise _DataRefused(f"{len(probs)} problem(s): " + "; ".join(probs[:8]))
    return {"dataset_version": m["dataset_version"], "manifest_body_sha256": m["manifest_body_sha256"],
            "rows_sha256": rows_sha, "npz_all_sha256": _canon_sha(triples), "n_entries": len(rows),
            "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


class _CooperativeStop(Exception):
    """Raised at a safe unit when a resource condition says the fit must not continue."""

    def __init__(self, record: dict):
        super().__init__(record.get("reason"))
        self.record = record


class _CoopControl:
    """T13: cooperative within-fit resource stops (opt-in: --coop-control FILE).

    The runner writes FILE before the fit (deadline_utc, active_budget_s, min_free_bytes,
    ckpt_forecast_bytes, fit_bytes_budget, epoch_forecast_s, check_every_steps,
    max_battery_pause_s) and may update runner_paused_s / stop_requested during it.
    Checks run (a) after every `check_every_steps` OPTIMIZER steps, (b) at the start of
    every epoch, with a forecast of one more epoch, and (c) before every checkpoint
    write, with a forecast of its bytes. A stop in (a) abandons the partial epoch and
    writes no checkpoint (an epoch without inner validation cannot be selected); (b)
    and (c) stop between epochs with best.pt of the last completed epoch intact, and (b)
    adds last.pt (end of the last completed epoch) when save_last is on and room allows.
    Maximum overrun past a deadline or budget: one check interval (check_every_steps
    optimizer steps) plus one inner-validation pass plus one checkpoint write; the
    forecasts in (b)/(c) normally stop before the boundary. Battery: the trainer waits
    at the safe unit (paused, not active time) and stops if still on battery after
    max_battery_pause_s; "unknown" power never stops a fit (the runner gates on it).
    No probe touches an RNG, so a fit that is never stopped is numerically unchanged.
    Test hooks (test_clock_file, test_power_file, test_free_file) are honoured only when
    the control file sets allow_test_hooks, and the stop record says so."""

    def __init__(self, path):
        self.path = Path(path)
        self.cfg = json.loads(self.path.read_text())
        self.start = self.now()
        self.battery_wait_s = 0.0
        self.max_step_s = 0.0
        self.max_epoch_s = 0.0
        self.opt_steps = 0
        self.checks = 0

    def _hook(self, key):
        return self.cfg.get(key) if self.cfg.get("allow_test_hooks") else None

    def now(self) -> float:
        f = self._hook("test_clock_file")
        return float(Path(f).read_text()) if f else time.time()

    def power(self) -> str:
        f = self._hook("test_power_file")
        if f:
            return Path(f).read_text().strip()
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            return "unknown"
        return "AC" if "'AC Power'" in out else ("battery" if "'Battery Power'" in out else "unknown")

    def free(self, p: Path) -> int:
        f = self._hook("test_free_file")
        if f:
            return int(float(Path(f).read_text()))
        p = Path(p)
        while not p.exists():
            p = p.parent
        return shutil.disk_usage(p).free

    def active_s(self, now: float) -> float:
        return (now - self.start) - self.battery_wait_s - float(self.cfg.get("runner_paused_s") or 0.0)

    # ---- rolling admission (EC/resource): the same campaign scopes and caps as the runner's registry
    def write_forecast(self) -> int:
        acc = self.cfg.get("accounting")
        return int((acc or {}).get("ckpt_write_peak_bytes") or self.cfg.get("ckpt_forecast_bytes") or 0)

    def write_peak_limit(self):
        """The admitted size of one checkpoint temporary (None without campaign accounting: unchanged)."""
        acc = self.cfg.get("accounting")
        return int(acc["ckpt_write_peak_bytes"]) if acc and acc.get("ckpt_write_peak_bytes") else None

    def accounted_bytes(self, acc: dict) -> int:
        """Campaign-accounted occupancy now: the registry scopes re-walked (regular files, inode dedup,
        nested roots skipped) plus the runner's charged-minus-walk offset (receipt floors, high-water)."""
        import stat as _stat
        seen: set = set()
        total = 0
        for sc in acc.get("scopes") or []:
            rp, kind = sc["path"], sc["scope"]["kind"]
            if not os.path.isdir(rp):
                continue
            if kind == "tree":
                tops = [rp]
            else:
                ex = set(sc.get("excluded_names") or [])
                pick = [n for n in sorted(os.listdir(rp)) if n not in ex and (
                    n in set(sc["scope"].get("names") or []) if kind == "names"
                    else any(n.startswith(q) for q in sc["scope"].get("prefixes") or []))]
                tops = [os.path.join(rp, n) for n in pick]
            skip = set(sc.get("skip") or [])
            for top in tops:
                walker = os.walk(top, followlinks=False) if os.path.isdir(top) else \
                    [(os.path.dirname(top), [], [os.path.basename(top)])]
                for d, dirs, files in walker:
                    dirs[:] = [x for x in dirs if os.path.join(d, x) not in skip]
                    for f in files:
                        try:
                            st = os.lstat(os.path.join(d, f))
                        except FileNotFoundError:
                            continue
                        if _stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) not in seen:
                            seen.add((st.st_dev, st.st_ino))
                            total += st.st_size
        return total + int(acc.get("charged_offset_bytes") or 0)

    def write_admission(self, fold_dir: Path, nbytes: int) -> list:
        """Reasons a write of `nbytes` (a checkpoint temporary; 0 = none) may NOT be made now.
        With campaign accounting: accounted + campaign reservations + nbytes + stop headroom <= bytes_cap,
        and on every affected volume free - nbytes (checkpoint volume) - other reservations - stop headroom
        >= reserve. Without it (v1 control): the pre-amendment rule, plus the fit byte budget, which the
        epoch-start last.pt writer used to skip."""
        c = self.cfg
        acc = c.get("accounting")
        reasons = []
        if not acc:
            need = int(c.get("min_free_bytes") or 0) + int(nbytes)
            if c.get("fit_bytes_budget") is not None and nbytes:
                used = sum(f.stat().st_size for f in Path(fold_dir).rglob("*") if f.is_file())
                if used + int(nbytes) > int(c["fit_bytes_budget"]):
                    reasons.append({"reason": "bytes_budget_forecast",
                                    "detail": {"fold_dir_bytes": used, "budget": c["fit_bytes_budget"]}})
            free = self.free(Path(fold_dir))
            if free < need:
                reasons.append({"reason": "free_space_forecast" if nbytes else "free_space",
                                "detail": {"free": free, "need": need}})
            return reasons
        head = int(acc.get("stop_headroom_bytes") or 0)
        if not acc.get("resolved", False):
            reasons.append({"reason": "accounting_unresolved", "detail": "the runner's accounting was unresolved"})
        try:
            used = self.accounted_bytes(acc)
        except OSError as exc:
            used = None
            reasons.append({"reason": "accounting_unresolved", "detail": repr(exc)[:200]})
        res = int(acc.get("campaign_reserved_bytes") or 0)
        if used is not None and used + res + int(nbytes) + head > int(acc["bytes_cap"]):
            reasons.append({"reason": "bytes_cap_forecast",
                            "detail": {"accounted": used, "campaign_reserved": res, "write": int(nbytes),
                                       "stop_headroom": head, "cap": int(acc["bytes_cap"])}})
        for v in acc.get("volumes") or []:
            free = self.free(Path(v["path"]))
            g = int(nbytes) if v.get("hosts_checkpoint") else 0
            other = int(v.get("other_reserved_bytes") or 0)
            if free - g - other - head < int(acc["reserve_bytes"]):
                reasons.append({"reason": "free_space_forecast" if nbytes else "free_space",
                                "detail": {"volume": v["path"], "free": free, "write": g, "other_reserved": other,
                                           "stop_headroom": head, "reserve": int(acc["reserve_bytes"])}})
        return reasons

    def check(self, phase: str, epoch, step, fold_dir: Path) -> None:
        self.checks += 1
        try:
            self.cfg = json.loads(self.path.read_text())          # the runner may have updated it
        except (OSError, ValueError):
            pass                                                    # keep the last good copy
        c = self.cfg
        reasons = []
        if phase in ("step", "epoch_start") and self.power() == "battery":
            t0 = time.monotonic()
            limit = float(c.get("max_battery_pause_s", 1800))
            while self.power() == "battery":
                if time.monotonic() - t0 >= limit:
                    reasons.append({"reason": "power", "detail": f"on battery for >= {limit} s at a safe unit"})
                    break
                time.sleep(float(c.get("battery_poll_s", 5.0)))
            self.battery_wait_s += time.monotonic() - t0
        now = self.now()
        dl = c.get("deadline_utc")
        dl_ts = datetime.fromisoformat(dl.replace("Z", "+00:00")).timestamp() if dl else None
        budget = c.get("active_budget_s")
        act = self.active_s(now)
        if c.get("stop_requested"):
            reasons.append({"reason": "stop_requested", "detail": c.get("stop_reason")})
        if dl_ts is not None and now >= dl_ts:
            reasons.append({"reason": "deadline", "detail": {"deadline_utc": dl, "overrun_s": round(now - dl_ts, 3)}})
        if budget is not None and act >= float(budget):
            reasons.append({"reason": "active_budget", "detail": {"active_s": round(act, 3), "budget_s": budget}})
        if phase == "epoch_start":
            fc = max(self.max_epoch_s, float(c.get("epoch_forecast_s") or 0.0))
            if dl_ts is not None and now + fc > dl_ts:
                reasons.append({"reason": "deadline_forecast", "detail": {"forecast_epoch_s": fc}})
            if budget is not None and act + fc > float(budget):
                reasons.append({"reason": "active_budget_forecast", "detail": {"forecast_epoch_s": fc}})
        if c.get("accounting"):
            # before a save, and at an epoch start (the epoch may need one checkpoint write): room for one
            # checkpoint temporary; at a step: growth elsewhere must still leave the safe-stop headroom.
            reasons += self.write_admission(fold_dir, self.write_forecast() if phase in ("before_save", "epoch_start")
                                            else 0)
        else:
            need = int(c.get("min_free_bytes") or 0)
            if phase == "before_save":
                need += int(c.get("ckpt_forecast_bytes") or 0)
                if c.get("fit_bytes_budget") is not None:
                    used = sum(f.stat().st_size for f in Path(fold_dir).rglob("*") if f.is_file())
                    if used + int(c.get("ckpt_forecast_bytes") or 0) > int(c["fit_bytes_budget"]):
                        reasons.append({"reason": "bytes_budget_forecast",
                                        "detail": {"fold_dir_bytes": used, "budget": c["fit_bytes_budget"]}})
            free = self.free(Path(fold_dir))
            if free < need:
                reasons.append({"reason": "free_space_forecast" if phase == "before_save" else "free_space",
                                "detail": {"free": free, "need": need}})
        if reasons:
            raise _CooperativeStop({"reason": reasons[0]["reason"], "reasons": reasons, "phase": phase,
                                    "epoch": epoch, "step": step, "clock_s": now,
                                    "overrun_s": (round(now - dl_ts, 3) if dl_ts is not None else None),
                                    "active_s": round(act, 3), "battery_wait_s": round(self.battery_wait_s, 3),
                                    "max_step_s": round(self.max_step_s, 3),
                                    "max_epoch_s": round(self.max_epoch_s, 3),
                                    "check_every_steps": int(c.get("check_every_steps") or 1),
                                    "checks": self.checks, "test_hooks_active": bool(c.get("allow_test_hooks"))})


_COOP_CONTROL = None      # a _CoopControl, set by main() from --coop-control; None = no checks

def _state_digest(state: dict) -> str:
    """sha256 over (name, shape, float32 bytes) in name order -- VisionEncoder.provenance()'s
    algorithm, so an encoder digest taken here is comparable with a fresh provenance()."""
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        arr = t.detach().to("cpu")
        h.update(name.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.contiguous().float().numpy().tobytes())
    return h.hexdigest()


class _GroundingDiagnostics:
    """Per-microbatch record of whether grounding supervision reaches the maps.

    Opt-in (train.grounding_diagnostics). Reads only: the grid target is the
    same roi_to_grid(batch["roi"], h, w) that grounding_loss computes, and a
    tensor hook on out["attn_maps"] observes -- never modifies -- the gradient
    arriving at the head-averaged maps. In training those maps feed nothing but
    the grounding loss (the classifier reads region embeddings built from the
    per-head weights), so the hook sees exactly the grounding term's gradient.

    Expected gradient: ground weight != 0 AND at least one sample in the
    microbatch has a non-empty grid target. A zero weight drops the term
    (total_loss); an empty target makes grounding_loss return a constant. Both
    are "no gradient by design" and are recorded as such, never forced.
    Anomalies: expected but absent (a disconnected graph), present but not
    expected, or NON-FINITE. A NaN/Inf gradient at the maps, or a non-finite
    grounding loss under a non-zero weight, is `anomaly_nonfinite` and never
    counts as supervision (T06, 2026-10-03: before, any NaN norm made
    `observed` true and the microbatch counted as "supervised"). Each record
    carries the raw fields (finiteness, expectation, valid target count, loss
    weight) so acceptance can rebuild every count instead of trusting the
    summary. Per-query gradient norms show the target semantics: query0 ->
    query 0 only; union -> whichever queries hold the max.
    """

    def __init__(self, cfg, ckpt_dir: Path, fold: int, n_train: int, n_val: int, n_test: int, device):
        lw = cfg.train.loss_weights
        self.weight = float(lw.get("ground", 0.0)) if isinstance(lw, dict) else float(lw.ground)
        self.target = str(getattr(cfg.train, "ground_target", "union"))
        self.path = ckpt_dir / "grounding_microbatch.jsonl"
        self.fh = open(self.path, "w")
        self.counts = {"microbatches": 0, "supervised": 0, "no_target_by_design": 0,
                       "empty_target": 0, "anomaly_disconnected": 0, "anomaly_unexpected_gradient": 0,
                       "anomaly_nonfinite": 0}
        self.queries_with_grad: set[int] = set()
        self.q0_only = True
        env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                               "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS")}
        self._write({"kind": "header", "diagnostics_schema": 2, "fold": fold, "init_seed": int(cfg.seed),
                     "partition_seed": partition_seed(cfg),
                     "region_pooling": str(cfg.model.get("region_pooling", "mean")),
                     "ground_weight": self.weight, "ground_target": self.target,
                     "grounding_gradient_expected_by_design": self.weight != 0.0,
                     "device": str(device), "torch": torch.__version__,
                     "torch_num_threads": torch.get_num_threads(),
                     "torch_num_interop_threads": torch.get_num_interop_threads(),
                     "env_threads": env, "n_inner_train": n_train, "n_inner_val": n_val,
                     "n_outer_test": n_test})

    def _write(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec) + "\n")

    def before_backward(self, out: dict, batch) -> tuple:
        from theia.engine.losses import roi_to_grid
        a = out["attn_maps"]
        holder: dict = {}
        handle = None
        if a.requires_grad:
            def _hook(g):
                holder["g"] = g.detach().float().flatten(2).norm(dim=2).sum(0)   # [Q]
                return None
            handle = a.register_hook(_hook)
        with torch.no_grad():
            h, w = a.shape[-2:]
            tgt = roi_to_grid(batch["roi"], h, w, a.device)
            n_valid = int((tgt.flatten(1).sum(1) > 0).sum())
        return bool(a.requires_grad), holder, handle, n_valid

    def after_backward(self, probe: tuple, epoch: int, step: int, parts: dict, batch) -> None:
        req, holder, handle, n_valid = probe
        if handle is not None:
            handle.remove()
        g = holder.get("g")
        fired = g is not None
        norms = [float(x) for x in g] if fired else None
        grad_finite = all(math.isfinite(x) for x in norms) if fired else None
        loss_ground = float(parts["ground"])
        loss_ground_finite = math.isfinite(loss_ground)
        # Only a finite, non-zero gradient is evidence that supervision reached the maps.
        observed = bool(fired and grad_finite and any(x != 0.0 for x in norms))
        expected = self.weight != 0.0 and n_valid > 0
        if (fired and not grad_finite) or (self.weight != 0.0 and not loss_ground_finite):
            status = "anomaly_nonfinite"
        elif expected and observed:
            status = "supervised"
            nz = [i for i, x in enumerate(norms) if x != 0.0]
            self.queries_with_grad.update(nz)
            if nz != [0]:
                self.q0_only = False
        elif not expected and not observed:
            status = "no_target_by_design" if self.weight == 0.0 else "empty_target"
        elif expected:
            status = "anomaly_disconnected"
        else:
            status = "anomaly_unexpected_gradient"
        self.counts["microbatches"] += 1
        self.counts[status] += 1
        self._write({"kind": "microbatch", "epoch": epoch, "step": step,
                     "patient_ids": list(batch["patient_id"]),
                     "n": len(batch["patient_id"]),
                     "n_has_mask": int(batch["has_mask"].sum()) if "has_mask" in batch else None,
                     "n_valid_targets": n_valid, "attn_requires_grad": req,
                     "grounding_gradient_expected": expected, "grounding_gradient_observed": observed,
                     "hook_fired": fired, "grad_norm_by_query": norms, "grad_finite": grad_finite,
                     "loss_weight_ground": self.weight, "loss_ground_finite": loss_ground_finite,
                     "loss_total_finite": math.isfinite(float(parts["total"])), "status": status,
                     "loss_cls": float(parts["cls"]), "loss_ground": loss_ground,
                     "loss_total": float(parts["total"])})

    def stopped(self, rec: dict) -> None:
        """Cooperative stop (T13): mark the file as belonging to an unfinished fit."""
        self._write({"kind": "stopped", **rec, "cumulative": dict(self.counts)})
        self.fh.close()

    def end_epoch(self, epoch: int) -> None:
        self._write({"kind": "epoch_end", "epoch": epoch, "cumulative": dict(self.counts)})
        self.fh.flush()

    def close(self) -> dict:
        summary = {**self.counts, "ground_weight": self.weight, "ground_target": self.target,
                   "queries_with_gradient": sorted(self.queries_with_grad),
                   "supervised_gradient_on_query0_only": (self.q0_only if self.counts["supervised"] else None),
                   "anomalies": (self.counts["anomaly_disconnected"] + self.counts["anomaly_unexpected_gradient"]
                                 + self.counts["anomaly_nonfinite"])}
        self._write({"kind": "summary", **summary})
        self.fh.close()
        return summary


class _FoldStalled(RuntimeError):
    """Every optimizer step in an epoch was skipped, so the fold is not training."""


def train_fold_with_retry(cfg, fold: int, tr, va, te, device, max_retries: int = 1,
                          on_retry=None):
    """Run a fold, retrying from a fresh initialisation if it stalls numerically.

    There is an unresolved numerical instability in the backward pass on MPS: the
    forward is finite everywhere, the backward is NaN everywhere, and registering
    backward hooks on every module makes it disappear — so it is a marginal
    overflow sensitive to autograd graph structure, not a divide-by-zero. An
    affected fold skips every optimizer step from then on, so its weights never
    change and it would report an untrained model as a result.

    Measured, so nobody re-derives it. Every loss term has been dropped from the
    graph individually and NONE of them is the cause: at ground=0.0 the NaN
    reappears unchanged, and at gen=0.0 it still occurs with the affected set
    reading vision 28 / grounding 7 / classifier 8 / generation 0 — the
    generation head is not even in the graph. What is common to every failing
    configuration is the backward through the unfrozen ViT blocks. Retrying from
    a different initialisation does NOT reliably help either: run 15's fold 0
    stalled on all three seeds.

    So the retry is kept only because it is cheap and occasionally works. The
    stall DETECTION is the part that earns its place — without it the fold is
    silently reported as a result.

    The retry criterion is deliberately NUMERICAL FAILURE, never performance. A
    fold is re-run only when it skipped every step of an epoch; a fold that
    trains and scores badly is kept exactly as it is. Retrying on a bad AUC would
    be seed-shopping and would bias every reported number upward. The seed offset
    used is recorded in the fold's result so the archive shows what happened.
    """
    for attempt in range(max_retries + 1):
        if attempt:
            # Perturb only the initialisation. The data split is untouched, so
            # the patient held out by this fold is the same on every attempt.
            set_seed(cfg.seed + 1000 * attempt)
            print(f"[train] fold{fold}: retry {attempt} with init seed "
                  f"{cfg.seed + 1000 * attempt}")
            if on_retry is not None:
                # A changed initialization is a new experimental attempt, never
                # an invisible continuation (R07): the runner ledgers it.
                on_retry({"retry": attempt, "init_seed": cfg.seed + 1000 * attempt})
        try:
            result = train_fold(cfg, fold, tr, va, te, device)
            if attempt:
                result["init_seed_offset"] = 1000 * attempt
            return result
        except _FoldStalled as exc:
            print(f"[train] {exc}")
            if attempt == max_retries:
                print(f"[train] fold{fold}: still stalling after {max_retries} "
                      "retries; reporting it as stalled rather than as a result")
                return {"fold": fold, "stalled": True, "test": {}}
    return {"fold": fold, "stalled": True, "test": {}}


def _warn_if_monitor_is_frozen(log: list[dict], fold: int, monitor, n: int = 4) -> None:
    """Flag a monitor that is bit-identical across several epochs.

    Training that is genuinely stuck and training that is merely plateauing look
    the same in the summary, but they are not the same: an AUC repeating to full
    float precision for four epochs means the model's ranking of the validation
    set has not changed AT ALL, which is a degenerate state, not slow progress.

    Run 13's fold 0 sat at exactly 0.556 with lift exactly 0.000 for all nine of
    its epochs and was then reported as a normal early stop. Nothing in the
    output distinguished it from a fold that trained and plateaued.
    """
    vals = [monitor_value(m, monitor) for m in log if m.get("split") == "inner_val"]
    if len(vals) < n or any(v != v for v in vals[-n:]):
        return
    if len(set(vals[-n:])) == 1:
        print(f"[train] WARNING: fold{fold} monitor has been exactly {vals[-1]:.6f} "
              f"for {n} epochs — the model's validation ranking is not changing. "
              "This is a stuck fold, not a plateau.")


def train_fold(cfg, fold: int, train_idx, val_idx, test_idx, device) -> dict:
    tl, vl, testl = make_loaders(cfg, train_idx, val_idx, test_idx)
    amp_on, amp_dtype = amp_settings(device, cfg.train.amp)
    model = Theia(cfg).to(device)
    opt = torch.optim.AdamW(_param_groups(model, cfg), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    # scheduler steps once per OPTIMIZER step (every grad_accum batches), so
    # total_steps must match that count or OneCycleLR raises past the end.
    opt_steps_per_epoch = max(len(tl) // cfg.train.grad_accum, 1)
    steps = cfg.train.epochs * opt_steps_per_epoch
    # max_lr must be per-group, or OneCycleLR overwrites the discriminative rates
    # set in _param_groups and drives the pretrained blocks at the head rate.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps,
        pct_start=cfg.train.warmup_ratio)
    scaler = make_grad_scaler(device, amp_on)

    best, best_epoch, patience = -float("inf"), -1, 0
    # `best` drives patience; the selector decides which checkpoint ships. They
    # agree exactly unless train.monitor_gate is set. See monitor_gate_ok.
    selector = Selector(getattr(cfg.train, "monitor_gate", None))
    reported_nan = False
    ckpt_dir = Path(cfg.paths.ckpt_dir) / f"fold{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = []
    # N08 replication additions, all opt-in; absent keys reproduce history.
    diag = (_GroundingDiagnostics(cfg, ckpt_dir, fold, len(train_idx), len(val_idx),
                                  len(test_idx or []), device)
            if bool(getattr(cfg.train, "grounding_diagnostics", False)) else None)
    record_identity = bool(getattr(cfg.train, "record_checkpoint_identity", False))
    # Parameter-load identity, taken BEFORE any training step. result["backbone"]'s
    # weight_sha256 is computed at the end of the fold (after best.pt is loaded back), so
    # it digests the fine-tuned weights, not what was loaded; it is left unchanged.
    init_identity = None
    if record_identity:
        vis = getattr(model.vision, "model", model.vision)
        init_identity = {"encoder_load_sha256": _state_digest(vis.state_dict()),
                         "initial_state_sha256": _state_digest(model.state_dict())}
    stop_after = getattr(cfg.train, "stop_after_epochs", None)
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, int)
                                   or stop_after < 1):
        raise ValueError(f"train.stop_after_epochs must be a positive integer, got {stop_after!r}")

    coop = _COOP_CONTROL
    every = max(1, int((coop.cfg.get("check_every_steps") if coop is not None else 1) or 1))

    def payload(metrics: dict) -> dict:
        return {"model": model.state_dict(), "cfg": dict(cfg), "metrics": metrics,
                "fold": fold, "lora_applied": model.lora_applied}

    def finish_stop(stop: "_CooperativeStop", phase: str, epoch, step) -> None:
        """Keep best.pt (atomic, last completed epoch); at an epoch boundary add last.pt ONLY when the
        same write admission as every other checkpoint allows it; write cooperative_stop.json."""
        rec = stop.record
        for k, v in (("phase", phase), ("epoch", epoch), ("step", step)):
            rec.setdefault(k, v)
        rec["last_completed_epoch"] = log[-1]["epoch"] if log else None
        rec["best_epoch"] = selector.best_epoch
        if phase == "epoch_start" and log and bool(getattr(cfg.train, "save_last", False)):
            refused = coop.write_admission(ckpt_dir, coop.write_forecast())
            if refused:
                rec["last_pt_refused"] = refused          # the safe best.pt stays; no write crosses a cap
            else:
                try:
                    _atomic_torch_save(payload(log[-1]), ckpt_dir / "last.pt", max_bytes=coop.write_peak_limit())
                    rec["last_pt_epoch"] = log[-1]["epoch"]
                except _CooperativeStop as over:
                    rec["last_pt_refused"] = over.record.get("reasons")
        rec["checkpoints"] = {n: (_file_identity(ckpt_dir / n) if (ckpt_dir / n).exists() else None)
                              for n in ("best.pt", "last.pt")}
        rec["inner_val_log"] = list(log)
        rec["outer_fold_scored"] = False
        if diag is not None:
            diag.stopped({k: rec.get(k) for k in ("reason", "phase", "epoch", "step")})
        _write_json_atomic(ckpt_dir / "cooperative_stop.json", rec)
        raise stop

    def coop_check(phase: str, epoch, step) -> None:
        """T13. No-op without --coop-control. On a stop: see finish_stop."""
        if coop is None:
            return
        try:
            coop.check(phase, epoch, step, ckpt_dir)
        except _CooperativeStop as stop:
            finish_stop(stop, phase, epoch, step)

    def save(path: Path, metrics: dict) -> None:
        coop_check("before_save", metrics.get("epoch"), None)
        lim = coop.write_peak_limit() if coop is not None else None
        if lim is None:
            _atomic_torch_save(payload(metrics), path)
            return
        try:
            _atomic_torch_save(payload(metrics), path, max_bytes=lim)
        except _CooperativeStop as stop:
            finish_stop(stop, "save", metrics.get("epoch"), None)

    for epoch in range(cfg.train.epochs):
        t_epoch = time.monotonic()
        coop_check("epoch_start", epoch, None)
        t_step = time.monotonic()
        model.train()
        opt.zero_grad()
        pbar = tqdm(tl, desc=f"fold{fold} ep{epoch}")
        running, seen, skipped, overflowed = None, 0, 0, 0
        recent_ids: list = []
        for step, batch in enumerate(pbar):
            with autocast(device, amp_on, amp_dtype):
                out = model(batch, device)
                loss, parts = total_loss(out, batch, cfg.train.loss_weights, device,
                                         str(getattr(cfg.train, "ground_target", "union")))
                loss = loss / cfg.train.grad_accum
            recent_ids = (recent_ids + list(batch["patient_id"]))[-2 * cfg.train.batch_size:]
            probe = diag.before_backward(out, batch) if diag is not None else None
            scaler.scale(loss).backward()
            if diag is not None:
                diag.after_backward(probe, epoch, step, parts, batch)
            if (step + 1) % cfg.train.grad_accum == 0:
                prev_scale = scaler.get_scale()
                # Clip, and skip the step outright if the gradient is not finite.
                #
                # On CUDA, GradScaler already skips non-finite steps. Off CUDA
                # (MPS, CPU) AMP is disabled, the scaler is a no-op, and NOTHING
                # skipped them: one bad batch wrote NaN into the weights and the
                # fold was dead from that point on, still burning epochs until
                # early stopping. Fold 0 of run 10 went NaN at epoch 8 and kept
                # its epoch-0 checkpoint, i.e. contributed an untrained model to
                # the pooled estimate. The old code detected this and printed a
                # warning, but had no way to act on it.
                #
                # unscale_ is a no-op when the scaler is disabled, so the clip
                # sees true gradients on every backend.
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None],
                    float(getattr(cfg.train, "grad_clip", 1.0)))
                if torch.isfinite(gnorm):
                    scaler.step(opt)
                    scaler.update()
                else:
                    # Distinguish the two ways this happens, because they need
                    # opposite responses and look identical from the norm alone:
                    #   every grad finite -> the SUM OF SQUARES overflowed fp32
                    #     while each element was representable. The gradients are
                    #     usable and merely enormous.
                    #   some grad non-finite -> a true NaN/inf. Skipping is right.
                    # Without this the log says "non-finite" for both and there is
                    # no way to tell which you are looking at.
                    grads_ok = all(torch.isfinite(p.grad).all()
                                   for p in model.parameters() if p.grad is not None)
                    skipped += 1
                    if grads_ok:
                        overflowed += 1
                    elif not reported_nan:
                        # One-shot per fold: name the parameters and the patients.
                        # This instability has recurred across runs 11 and 14 and
                        # is not yet explained; isolated harnesses would not
                        # reproduce it, and guessing at causes (grounding loss,
                        # empty rationales, fp32 norm overflow) has cost more
                        # than instrumenting it once. Bounded output, no spam.
                        reported_nan = True
                        bad = [n for n, p in model.named_parameters()
                               if p.grad is not None and not torch.isfinite(p.grad).all()]
                        total = sum(1 for p in model.parameters() if p.grad is not None)
                        # Group by top-level module. "first 6" is misleading here:
                        # named_parameters() starts at the vision encoder, so a NaN
                        # that originates at the loss and propagates everywhere looks
                        # identical to one that starts in the backbone. The share of
                        # each module is what distinguishes them.
                        by_mod: dict[str, int] = {}
                        for n in bad:
                            by_mod[n.split(".")[0]] = by_mod.get(n.split(".")[0], 0) + 1
                        print(f"[train] fold{fold} ep{epoch} step{step}: FIRST "
                              f"non-finite gradient in {len(bad)}/{total} params "
                              f"with grads; by module: {by_mod}")
                        print(f"[train]   loss parts: "
                              f"{ {k: round(float(v), 4) for k, v in parts.items()} }")
                        print(f"[train]   patients in the accumulated batches: "
                              f"{recent_ids}")
                    scaler.update()
                opt.zero_grad()
                # A skipped step (inf/nan gradients) must not advance the LR
                # schedule, or OneCycleLR drifts out of sync with real progress.
                if torch.isfinite(gnorm) and scaler.get_scale() >= prev_scale:
                    sched.step()
                if coop is not None:
                    coop.opt_steps += 1
                    coop.max_step_s = max(coop.max_step_s, time.monotonic() - t_step)
                    if coop.opt_steps % every == 0:
                        coop_check("step", epoch, step)
                    t_step = time.monotonic()
            # parts are detached tensors; converting every step would force a
            # host sync per iteration. Accumulate and format occasionally.
            if not torch.isfinite(parts["total"]):
                print(f"[train] fold{fold} ep{epoch} step{step}: non-finite loss "
                      f"({ {k: float(v) for k, v in parts.items()} }); "
                      "the fold has diverged")
            running = parts if running is None else {k: running[k] + parts[k] for k in parts}
            seen += 1
            if step % 20 == 0:
                pbar.set_postfix({k: f"{float(v) / seen:.3f}" for k, v in running.items()})

        metrics = evaluate(model, vl, cfg, device)
        metrics["epoch"] = epoch
        metrics["split"] = "inner_val"
        if skipped:
            # Recorded, not just printed: a fold that skipped many steps trained
            # on less than it appears to, and that belongs in the archive.
            metrics["skipped_steps"] = skipped
            metrics["overflow_steps"] = overflowed
            how = (f"{overflowed} from fp32 norm overflow (gradients themselves "
                   f"finite), {skipped - overflowed} from true NaN/inf")
            print(f"[train] fold{fold} ep{epoch}: skipped {skipped} optimizer "
                  f"step(s): {how}")
        # A fold that skipped EVERY step this epoch is not training at all: the
        # weights are exactly what they were, and it will report an untrained
        # model as if it were a result. Give up on this initialisation and say so.
        if skipped and skipped >= opt_steps_per_epoch and epoch > 0:
            raise _FoldStalled(
                f"fold{fold} skipped all {skipped} optimizer steps at epoch {epoch}")
        log.append(metrics)
        monitor = monitor_value(metrics, cfg.train.monitor)

        # Selection and stopping are deliberately decoupled. The gate decides
        # which checkpoint is deployed; patience still counts on the ungated
        # monitor, so enabling the gate cannot change when a fold stops -- only
        # which of that fold's epochs ships. That is what makes the change
        # verifiable against the archived logs, which record every epoch but
        # only one set of stopping decisions.
        if selector.offer(metrics, monitor, epoch):
            best_epoch = selector.best_epoch
            save(ckpt_dir / "best.pt", metrics)
        if diag is not None:
            diag.end_epoch(epoch)
        if coop is not None:
            coop.max_epoch_s = max(coop.max_epoch_s, time.monotonic() - t_epoch)
        if stop_after is not None and epoch + 1 >= stop_after:
            # Parity runs only: stop early while the OneCycle schedule stays sized
            # by train.epochs, so the first epochs see the production LR curve.
            print(f"[train] fold{fold}: stopping after {epoch + 1} epoch(s) "
                  f"(train.stop_after_epochs; schedule sized for {cfg.train.epochs})")
            break

        if _is_better(monitor, best):
            best, patience = monitor, 0
        else:
            # Patience does not run during warmup. Under OneCycleLR the first
            # epochs sit at a fraction of peak LR (8e-6 against 2e-4 here), so
            # epoch 0 is an essentially untrained model whose validation AUC is
            # a draw from noise on ~24 patients. Letting that draw set the bar
            # is selection on noise, and it is not hypothetical: run 13's fold 2
            # peaked at ep0 (0.620), then climbed 0.436 -> 0.571 over epochs 4-8
            # with grounding lift rising to +0.272 and still going, and was
            # stopped anyway because patience had been counting from ep0.
            if epoch + 1 < int(getattr(cfg.train, "early_stop_min_epochs", 0)):
                continue
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[train] early stop fold{fold} @ ep{epoch} (best ep{best_epoch})")
                break

        _warn_if_monitor_is_frozen(log, fold, cfg.train.monitor)

    # Always leave a usable checkpoint. If the monitor was NaN every epoch
    # (single-class validation split) there is no "best", but downstream tools
    # still need weights to load rather than a FileNotFoundError.
    #
    # last.pt is written only when it is the fallback, or when explicitly asked
    # for. Nothing in this project ever LOADS it -- every tool opens best.pt --
    # and it doubles checkpoint storage: 33 of them had accumulated to 59 GB,
    # which was most of the reason there was no disk left for a second
    # pretraining cohort. Set train.save_last true to keep the final-epoch
    # weights, e.g. to inspect a run that early-stopped.
    if best_epoch < 0:
        save(ckpt_dir / "last.pt", log[-1] if log else {})
        print(f"[train] fold{fold}: monitor '{cfg.train.monitor}' was NaN every epoch "
              f"(validation split likely single-class); using last.pt as best.pt")
        save(ckpt_dir / "best.pt", log[-1] if log else {})
    elif bool(getattr(cfg.train, "save_last", False)):
        save(ckpt_dir / "last.pt", log[-1] if log else {})
    last_identity = (_file_identity(ckpt_dir / "last.pt")
                     if record_identity and (ckpt_dir / "last.pt").exists() else None)

    # selector.best, not `best`: report the monitor value of the epoch that
    # actually shipped. With no gate the two are identical; with a gate, `best`
    # may name an epoch that was never eligible to be deployed.
    result = dict(fold=fold, best_inner=None if best_epoch < 0 else selector.best,
                  best_epoch=best_epoch)
    if record_identity:
        iv = [m["epoch"] for m in log if m.get("split") == "inner_val"]
        result["final_epoch"] = iv[-1] if iv else None
        result["epochs_run"] = len(iv)
        result["last_pt"] = last_identity
        result["best_pt_scored"] = None
        result["parameter_load_identity"] = init_identity
    if stop_after is not None:
        result["stop_after_epochs"] = stop_after

    # The outer fold is scored exactly once, here, after the model is frozen.
    if testl is not None:
        if record_identity:
            result["best_pt_scored"] = _file_identity(ckpt_dir / "best.pt")
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, testl, cfg, device, full=True)
        test_metrics["epoch"] = best_epoch
        test_metrics["split"] = "outer_test"
        log.append(test_metrics)
        result["test"] = test_metrics
        # Kept per-patient so the folds can be pooled into one ranking later.
        result["oof"] = collect_predictions(model, testl, cfg, device, fold=fold)

    # DEF-0012: record the encoder that actually loaded, not the one requested.
    # The weight digest is what lets a later reader answer "which model produced
    # this?" from the artifact rather than from the configuration's intent.
    try:
        result["backbone"] = model.vision.provenance()
    except Exception as exc:  # noqa: BLE001
        result["backbone"] = {"error": f"{type(exc).__name__}: {exc}"}

    _write_text_atomic(ckpt_dir / "log.jsonl", "".join(json.dumps(row) + "\n" for row in log))
    if diag is not None:
        result["grounding_diagnostics"] = diag.close()
    return result


def _monitor_label(monitor) -> str:
    if isinstance(monitor, str):
        return monitor
    return " + ".join(m if isinstance(m, str) else f"{m[1]}*{m[0]}" for m in monitor)


def _summarize(summary: list[dict], monitor) -> str:
    """Per-fold spread of the monitored quantity.

    Goes through monitor_value so a composite monitor (a list) does not get used
    as a dict key — `s["test"][monitor]` raises "unhashable type: list".
    """
    label = _monitor_label(monitor)
    vals = [monitor_value(s["test"], monitor) for s in summary if s.get("test")]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return f"[train] no fold produced a finite {label}"
    spread = f" ± {stdev(vals):.3f} (sd)" if len(vals) > 1 else ""
    return (f"[train] per-fold {label}: {mean(vals):.3f}{spread} "
            f"over {len(vals)}/{len(summary)} folds")


def _report_pooled(summary: list[dict], cfg) -> dict:
    """Pool every fold's out-of-fold predictions into one estimate and print it.

    This is the headline number, not the per-fold mean. Each patient contributes
    exactly one prediction, made by a model that never trained on them.
    """
    rows = [r for s in summary for r in (s.get("oof") or [])]
    if not rows:
        return {}
    pooled = pooled_metrics(rows, cfg.data.target_genes, cfg.eval.bootstrap_n)
    print(f"\n[train] POOLED OUT-OF-FOLD over {int(pooled['n_patients'])} patients:")
    for g in [x.lower() for x in cfg.data.target_genes]:
        auc = pooled.get(f"{g}_auc", float("nan"))
        n, npos = int(pooled.get(f"{g}_n", 0)), int(pooled.get(f"{g}_n_pos", 0))
        if math.isnan(auc):
            print(f"[train]   {g.upper():6s} n={n:4d} pos={npos:3d}  AUC undefined "
                  "(single-class)")
            continue
        lo, hi = pooled.get(f"{g}_auc_lo", float("nan")), pooled.get(f"{g}_auc_hi", float("nan"))
        warn = "  <-- CI includes chance" if lo <= 0.5 else ""
        print(f"[train]   {g.upper():6s} n={n:4d} pos={npos:3d}  "
              f"AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{warn}")
    return pooled


def _archive(run_id: str, cfg, summary: list, pooled: dict) -> str:
    """Write the one artifact that must outlive the checkpoints.

    Everything heavy — best.pt, last.pt — is gitignored and namespaced by run id,
    so it survives only as long as the disk. This file is small, tracked, and
    holds the resolved config plus every per-fold metric AND the out-of-fold
    predictions, which is enough to regenerate the pooled AUC, its CI and every
    ROC curve without the weights.

    Written because run 6 — the best result this project has produced — was
    destroyed by runs 7, 8 and 9 writing to the same `checkpoints/fold{k}/` and
    `runs/cv_summary.json` paths. Its headline AUC survived only in a log file
    under /private/tmp, which the OS is free to delete. A result you cannot
    reproduce a figure from is not a result.
    """
    out = Path(getattr(cfg.paths, "results_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    with open(path, "w") as fh:
        # Config subclasses dict and its nested values stay plain dicts, so it
        # serialises as-is — the archive records what actually ran, not what the
        # YAML on disk says today.
        # One backbone record per run: every fold builds the same encoder from
        # the same config, so disagreement between folds is itself a finding
        # and is kept rather than collapsed.
        seen = {json.dumps(f["backbone"], sort_keys=True)
                for f in summary if isinstance(f, dict) and f.get("backbone")}
        backbone = ([json.loads(x) for x in sorted(seen)] if len(seen) != 1
                    else json.loads(seen.pop()))
        json.dump({"run_id": run_id, "config": dict(cfg), "backbone": backbone,
                   "pooled": pooled, "folds": summary}, fh, indent=2, default=str)
    print(f"[train] archived -> {path}")
    return str(path)


# --------------------------------------------------------------------------- #
# R07: resume only what provably belongs to this experiment
# --------------------------------------------------------------------------- #
# The code a fold's result depends on. Declared, not read from sys.modules: a
# reused fold imports less than a trained one, and a fingerprint that depended on
# import order would refuse legitimate resumes. tests/test_train_resume_r07.py
# fails if the trainer ever imports a theia module missing from this list.
TRAINING_CODE = ("theia/__init__.py", "theia/config.py", "theia/runtime.py", "theia/ranking.py",
                 "theia/data/__init__.py", "theia/data/dataset.py",
                 "theia/engine/__init__.py", "theia/engine/train.py", "theia/engine/evaluate.py",
                 "theia/engine/losses.py", "theia/models/__init__.py", "theia/models/backbone.py",
                 "theia/models/heads.py", "theia/models/theia_model.py",
                 "theia/analysis/__init__.py", "theia/analysis/shards.py")
_RUN_PATH_KEYS = ("ckpt_dir", "ckpt_root", "runs_dir", "results_dir")


def _sha(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def run_fingerprint(cfg, rows_path: str, fold: int, tr, va, te, max_retries: int,
                    data_identity: dict | None = None) -> tuple[str, dict]:
    """What a fold's result depends on: data, split, code, config, seed,
    selection rule, retry policy and environment. With a configured data manifest the
    image bytes are bound too (dataset version, manifest body, every npz sha256)."""
    import platform

    c = json.loads(json.dumps(dict(cfg), default=str))
    for k in _RUN_PATH_KEYS:                   # where outputs go is not what they are
        c.get("paths", {}).pop(k, None)
    root = Path(__file__).resolve().parents[2]
    code = {f: _sha((root / f).read_bytes()) for f in TRAINING_CODE if (root / f).exists()}
    comps = {
        "config": _sha(json.dumps(c, sort_keys=True).encode()),
        "data_rows": _sha(Path(rows_path).read_bytes()),
        "split": _sha(json.dumps([None if x is None else [int(i) for i in x] for x in (tr, va, te)]).encode()),
        "code": _sha(json.dumps(code, sort_keys=True).encode()),
        "seed": int(cfg.seed),
        "monitor": json.dumps(cfg.train.monitor, default=str),
        "max_retries": int(max_retries),
        "fold": int(fold),
        "env": f"torch {torch.__version__}; numpy {np.__version__}; python {platform.python_version()}",
    }
    # Only when set, so configurations without split.seed keep their component set.
    if (cfg.get("split") or {}).get("seed") is not None:
        comps["partition_seed"] = partition_seed(cfg)
    if data_identity is not None:              # only when configured: legacy component sets unchanged
        comps["dataset_version"] = data_identity["dataset_version"]
        comps["data_manifest"] = data_identity["manifest_body_sha256"]
        comps["data_npz_all"] = data_identity["npz_all_sha256"]
    return _sha(json.dumps(comps, sort_keys=True).encode()), comps


def _ledger(fold_dir: Path, rec: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **rec}
    with open(fold_dir / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _ledger_rows(fold_dir: Path) -> list[dict]:
    p = fold_dir / "attempts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


_PRESERVED = ("best.pt", "last.pt", "log.jsonl", "fold_summary.json", "grounding_microbatch.jsonl")


def _preserve(fold_dir: Path, label: str, names=_PRESERVED) -> list:
    moved = []
    for n in names:
        if (fold_dir / n).exists():
            keep = fold_dir / "attempts" / label
            keep.mkdir(parents=True, exist_ok=True)
            (fold_dir / n).replace(keep / n)
            moved.append(n)
    return moved


def begin_attempt(fold_dir: Path, fp: str) -> int:
    """Classify earlier attempts, preserve their artifacts, open a new attempt."""
    rows = _ledger_rows(fold_dir)
    started = {r["attempt"] for r in rows if r.get("status") == "started"}
    ended = {r["attempt"] for r in rows if r.get("status") in ("complete", "failed", "interrupted",
                                                                "stopped_cooperative")}
    for a in sorted(started - ended):
        _ledger(fold_dir, {"attempt": a, "status": "interrupted",
                           "preserved": _preserve(fold_dir, f"attempt{a}_interrupted",
                                                  _PRESERVED + ("cooperative_stop.json",))})
    stopped = {r["attempt"] for r in rows if r.get("status") == "stopped_cooperative"}
    kept = {r["attempt"] for r in rows if r.get("status") == "stopped_preserved"}
    for a in sorted(stopped - kept):
        _ledger(fold_dir, {"attempt": a, "status": "stopped_preserved",
                           "preserved": _preserve(fold_dir, f"attempt{a}_stopped",
                                                  _PRESERVED + ("cooperative_stop.json",))})
    for d in sorted(fold_dir.glob(".partial-*")) if fold_dir.exists() else []:
        keep = fold_dir / "attempts" / "partial_writes"
        keep.mkdir(parents=True, exist_ok=True)
        d.replace(keep / f"{d.name}_{int(time.time())}")
        _ledger(fold_dir, {"status": "partial_write_preserved", "path": str(keep / d.name)})
    if not rows and any((fold_dir / n).exists() for n in ("best.pt", "last.pt")):
        _ledger(fold_dir, {"attempt": 0, "status": "interrupted",
                           "note": "artifacts from an attempt made before the ledger existed",
                           "preserved": _preserve(fold_dir, "attempt0_unrecorded")})
    n = max([int(r.get("attempt", 0)) for r in _ledger_rows(fold_dir)], default=0) + 1
    _ledger(fold_dir, {"attempt": n, "status": "started", "fingerprint": fp, "pid": os.getpid()})
    return n


def reuse_refusal(prev: dict, fp: str, comps: dict, accept_legacy: bool) -> str | None:
    """None if a completed summary may be reused under this fingerprint."""
    if "fingerprint" not in prev:
        return None if accept_legacy else (
            "summary has no fingerprint (written before R07), so it cannot be shown to belong "
            "to this experiment; pass --accept-legacy-summaries to reuse it knowingly")
    if prev["fingerprint"] != fp:
        old = prev.get("fingerprint_components", {})
        diff = sorted(k for k in comps if old.get(k) != comps[k])
        return f"completed under a different fingerprint (differs in: {', '.join(diff) or 'unknown'})"
    return None


def parse_deadline(value: str | None):
    if not value:
        return None
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("--deadline-utc needs an explicit UTC offset, e.g. 2026-09-21T13:32:55Z")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None, help="override cfg.train.device")
    ap.add_argument("--run_id", default=None,
                    help="names checkpoints/<run_id>/ and results/<run_id>.json; "
                         "defaults to a timestamp. Runs no longer overwrite each other.")
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold subset, e.g. 0,1. Default: every fold "
                         "(the historical behaviour).")
    ap.add_argument("--resume", action="store_true",
                    help="reuse a fold whose fold_summary.json is on disk under this run_id "
                         "AND whose recorded fingerprint (data, split, code, config, seed, "
                         "monitor, retry policy, environment) matches this invocation.")
    ap.add_argument("--accept-legacy-summaries", action="store_true",
                    help="with --resume, also reuse summaries written before fingerprints "
                         "existed; each reuse is ledgered as legacy_unverified.")
    ap.add_argument("--deadline-utc", default=None,
                    help="absolute UTC instant (e.g. 2026-09-21T13:32:55Z): no fold STARTS at or "
                         "after it. Absolute, so restarting the run cannot reset it.")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="numerical-stall retries per fold, each with a recorded new init seed.")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, e.g. --set train.loss_weights.gen=0 "
                         "--set seed=7. Repeatable. The resolved config is what "
                         "results/<run_id>.json records, so an override is never "
                         "invisible after the fact.")
    ap.add_argument("--coop-control", default=None, metavar="FILE",
                    help="T13: JSON written by the N08 runner; enables cooperative within-fit stops "
                         "(deadline, active budget, free space/bytes, power) at a safe unit, exit 86.")
    args = ap.parse_args()
    global _COOP_CONTROL
    _COOP_CONTROL = _CoopControl(args.coop_control) if args.coop_control else None
    overrides = {}
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key] = yaml.safe_load(raw)     # ints/floats/bools/lists, not str
    cfg = load_config(args.config, overrides or None)
    if overrides:
        print(f"[train] config overrides: {overrides}")
    set_seed(cfg.seed)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    # Join from a remembered root, not from the current value: mutating cfg in
    # place means a second main() on the same object would nest run-b inside
    # run-a. setdefault makes this idempotent however many times it runs.
    root = cfg["paths"].setdefault("ckpt_root", cfg.paths.ckpt_dir)
    cfg["paths"]["ckpt_dir"] = str(Path(root) / run_id)
    print(f"[train] run_id={run_id}  checkpoints -> {cfg.paths.ckpt_dir}")

    device = resolve_device(args.device or getattr(cfg.train, "device", "auto"))
    amp_on, _ = amp_settings(device, cfg.train.amp)
    print(f"[train] {describe(device, amp_on)}")
    if device.type == "cpu":
        print("[train] WARNING: no GPU backend. This is a scaffold run, not a real train.")

    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    data_identity = None
    manifest = (cfg.get("data") or {}).get("content_manifest")
    if manifest:
        try:
            data_identity = _verify_content_manifest(manifest, rows, (cfg.get("data") or {}).get("dataset_version"))
        except _DataRefused as exc:
            print(f"[train] data manifest refused: {exc}")
            raise SystemExit(EX_DATA_REFUSED)
        print(f"[train] data {data_identity['dataset_version']} manifest "
              f"{data_identity['manifest_body_sha256'][:12]}: {data_identity['n_entries']} npz verified")
    elif (cfg.get("data") or {}).get("dataset_version"):
        print("[train] data.dataset_version is set without data.content_manifest; refusing")
        raise SystemExit(EX_DATA_REFUSED)
    nested = bool(getattr(cfg.split, "nested", True))
    pseed = partition_seed(cfg)
    if (cfg.get("split") or {}).get("seed") is not None:
        print(f"[train] partition seed {pseed} (split.seed); initialisation seed {cfg.seed}")
    if nested:
        folds = nested_kfold_indices(rows, cfg.split.stratify_on, cfg.split.n_folds,
                                     pseed, float(getattr(cfg.split, "inner_val_frac", 0.2)))
    else:
        print("[train] WARNING: split.nested is false — the checkpoint is selected on "
              "the same split that gets reported, which biases the result upward.")
        folds = ((tr, va, None) for tr, va in
                 kfold_indices(rows, cfg.split.stratify_on, cfg.split.n_folds, pseed))

    # N09 resume boundary. Each fold's summary is written atomically as soon as
    # the fold finishes, so an interrupted run keeps every completed fold
    # instead of losing all of them with the end-of-run archive. `--folds`
    # restricts the run to a subset, and `--resume` reuses a fold whose summary
    # is already on disk under this run_id. Both default to the historical
    # behaviour: all folds, nothing reused.
    from theia.analysis.shards import write_atomic

    want = None if args.folds is None else {int(v) for v in str(args.folds).split(",")}
    deadline = parse_deadline(args.deadline_utc)
    summary = []
    for fold, (tr, va, te) in enumerate(folds):
        fold_dir = Path(cfg.paths.ckpt_dir) / f"fold{fold}"
        done = fold_dir / "fold_summary.json"
        if want is not None and fold not in want:
            print(f"[train] fold {fold}: not selected by --folds, skipping")
            continue
        fp, comps = run_fingerprint(cfg, rows, fold, tr, va, te, args.max_retries, data_identity)
        if args.resume and done.exists():
            prev = json.loads(done.read_text())
            refusal = reuse_refusal(prev, fp, comps, args.accept_legacy_summaries)
            if refusal:
                raise SystemExit(f"[train] fold {fold}: {refusal}; refusing to mix it into this run")
            legacy = "fingerprint" not in prev
            _ledger(fold_dir, {"status": "reused_legacy_unverified" if legacy else "reused",
                               "fingerprint": fp})
            print(f"[train] fold {fold}: complete summary on disk "
                  f"({'legacy, unverified' if legacy else 'fingerprint match'}), reusing")
            summary.append(prev["summary"])
            continue
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            print(f"[train] deadline {args.deadline_utc} reached; not starting fold {fold}")
            _ledger(fold_dir, {"status": "not_started_deadline", "deadline_utc": args.deadline_utc})
            break
        if done.exists():              # a completed fold being retrained is preserved first
            _ledger(fold_dir, {"status": "superseded_by_retrain",
                               "preserved": _preserve(fold_dir, f"superseded_{int(time.time())}")})
        att = begin_attempt(fold_dir, fp)
        t0 = time.time()
        try:
            s = train_fold_with_retry(
                cfg, fold, tr, va, te, device, max_retries=args.max_retries,
                on_retry=lambda r, a=att: _ledger(fold_dir, {"attempt": a, "status": "retry", **r}))
        except _CooperativeStop as stop:
            _ledger(fold_dir, {"attempt": att, "status": "stopped_cooperative", "reason": stop.record.get("reason"),
                               "phase": stop.record.get("phase"), "epoch": stop.record.get("epoch"),
                               "wall_s": round(time.time() - t0, 1)})
            print(f"[train] fold{fold}: cooperative stop ({stop.record.get('reason')}) at "
                  f"{stop.record.get('phase')} epoch {stop.record.get('epoch')}; exit {EX_COOP_STOP}")
            raise SystemExit(EX_COOP_STOP)
        except BaseException as exc:
            _ledger(fold_dir, {"attempt": att, "status": "failed", "error": repr(exc)[:500],
                               "wall_s": round(time.time() - t0, 1)})
            raise
        if data_identity is not None:
            # The bytes read during the fold must still be the configured ones.
            try:
                again = _verify_content_manifest(manifest, rows, data_identity["dataset_version"])
            except _DataRefused as exc:
                _ledger(fold_dir, {"attempt": att, "status": "failed", "error": f"data changed during fold: {exc}"[:500],
                                   "wall_s": round(time.time() - t0, 1)})
                print(f"[train] data changed during fold {fold}: {exc}")
                raise SystemExit(EX_DATA_REFUSED)
            s["data_identity"] = {**data_identity, "reverified_utc": again["verified_utc"]}
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(done, {"run_id": run_id, "fold": fold, "summary": s,
                            "written_utc": datetime.now(timezone.utc).isoformat(),
                            "fingerprint": fp, "fingerprint_components": comps, "attempt": att})
        _ledger(fold_dir, {"attempt": att, "status": "complete", "wall_s": round(time.time() - t0, 1),
                           "init_seed_offset": s.get("init_seed_offset", 0),
                           "stalled": bool(s.get("stalled"))})
        summary.append(s)
    if not summary:
        print("[train] no folds ran")
        return

    Path(cfg.paths.runs_dir).mkdir(parents=True, exist_ok=True)
    print(_summarize(summary, cfg.train.monitor))
    pooled = _report_pooled(summary, cfg)
    with open(os.path.join(cfg.paths.runs_dir, "cv_summary.json"), "w") as fh:
        json.dump({"folds": summary, "pooled": pooled, "run_id": run_id}, fh, indent=2)
    _archive(run_id, cfg, summary, pooled)


if __name__ == "__main__":
    main()
