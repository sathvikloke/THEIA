"""Resumable, atomically written result shards (WP33 / next-phase N09).

Why this exists: the WP09 MPS run lost six hours of computation because the
program wrote its only output at the end (ledger defect 21). The per-fold shell
script was a workaround at fold granularity; this module makes the boundary
explicit, finer, and checkable, so an interrupted long run keeps the work it
finished and cannot contribute invalid work to a final report.

Three separate things are kept apart on purpose:

  * a **shard** is a completed unit of work (a fold, or a batch of patients).
    It is written atomically -- to a temporary file in the same directory, then
    os.replace -- so a shard file is either absent or complete. It carries the
    schema version, its patient coverage, the hashes of everything it depended
    on, and the controls that passed while producing it.

  * the **run manifest** is append-only and records every attempt, including
    the ones that failed, were cancelled, or produced nothing. A partial shard
    is not a successful experiment receipt, and the manifest is where that
    distinction lives.

  * the **aggregate** is built from shards only after they have been checked
    for compatibility with each other and with the current inputs.

Resume is deliberately conservative: a shard is reused only when the code, the
inputs, the model checkpoint, the intervention definition and the seeds all
match what the current run would use. Anything else is a miss, and the unit is
recomputed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

_STATES = ("complete", "partial", "failed", "cancelled", "not_started")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_atomic(path: str | Path, payload: dict) -> str:
    """Write JSON so the file is never observed half-written. Returns its sha256.

    The temporary file is created in the destination directory so that
    os.replace is a same-filesystem rename, and it is fsync'd before the
    rename so a crash cannot leave a renamed-but-empty file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    body = json.dumps(payload, indent=1, sort_keys=False, default=str) + "\n"
    try:
        with open(tmp, "w") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class RunKey:
    """Everything a shard must agree on before it may be reused.

    `code` and `inputs` map a path to its sha256. `definition` holds the
    intervention/estimator definition and the seeds -- a change there means the
    saved numbers answer a different question, so they are never reused.
    """
    package: str
    attempt: str
    code: dict[str, str]
    inputs: dict[str, str]
    definition: dict[str, Any]

    def fingerprint(self) -> str:
        return sha256_obj({
            "schema": SCHEMA_VERSION,
            "package": self.package,
            "code": self.code,
            "inputs": self.inputs,
            "definition": self.definition,
        })

    def why_incompatible(self, saved: dict) -> list[str]:
        """Human-readable reasons a saved shard may not be reused. Empty = reusable."""
        out = []
        if saved.get("schema_version") != SCHEMA_VERSION:
            out.append(f"schema {saved.get('schema_version')} != {SCHEMA_VERSION}")
        if saved.get("package") != self.package:
            out.append(f"package {saved.get('package')} != {self.package}")
        for label, mine in (("code", self.code), ("inputs", self.inputs)):
            theirs = saved.get(label) or {}
            for k, v in mine.items():
                if k not in theirs:
                    out.append(f"{label}: {k} missing from the shard")
                elif theirs[k] != v:
                    out.append(f"{label}: {k} changed")
            for k in theirs:
                if k not in mine:
                    out.append(f"{label}: {k} no longer an input")
        if saved.get("definition") != self.definition:
            out.append("definition (intervention/estimator/seeds) changed")
        return out


@dataclass
class ShardStore:
    """Shard directory plus its append-only manifest."""
    root: Path
    key: RunKey
    manifest_path: Path | None = None
    _hits: list[str] = field(default_factory=list)
    _misses: list[tuple[str, list[str]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path is None:
            self.manifest_path = self.root / "run_manifest.jsonl"
        self.manifest_path = Path(self.manifest_path)

    # -- manifest ---------------------------------------------------------- #
    def log(self, unit: str, state: str, **extra: Any) -> None:
        if state not in _STATES:
            raise ValueError(f"unknown state {state!r}; expected one of {_STATES}")
        rec = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "package": self.key.package,
            "attempt": self.key.attempt,
            "unit": unit,
            "state": state,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "fingerprint": self.key.fingerprint(),
            **extra,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # -- shards ------------------------------------------------------------ #
    def path_for(self, unit: str) -> Path:
        return self.root / f"shard__{unit}.json"

    def load(self, unit: str) -> dict | None:
        """Return a reusable shard, or None. A shard that exists but is
        unreadable, incomplete or incompatible is a miss, and the reason is
        recorded rather than silently swallowed."""
        p = self.path_for(unit)
        if not p.exists():
            self._misses.append((unit, ["absent"]))
            return None
        try:
            saved = json.loads(p.read_text())
        except Exception as exc:                       # truncated or corrupt
            self._misses.append((unit, [f"unreadable: {type(exc).__name__}"]))
            self.log(unit, "failed", reason="corrupt shard", detail=str(exc))
            return None
        if saved.get("state") != "complete":
            self._misses.append((unit, [f"state {saved.get('state')!r}"]))
            return None
        why = self.key.why_incompatible(saved)
        if why:
            self._misses.append((unit, why))
            self.log(unit, "not_started", reason="incompatible shard", detail=why)
            return None
        if saved.get("payload_sha256") != sha256_obj(saved.get("payload")):
            self._misses.append((unit, ["payload hash mismatch"]))
            self.log(unit, "failed", reason="payload hash mismatch")
            return None
        self._hits.append(unit)
        return saved

    def save(self, unit: str, payload: dict, *, patients: Iterable[str],
             controls: dict | None = None, wall_s: float | None = None) -> Path:
        pids = sorted(set(patients))
        if len(pids) != len(list(patients)):
            raise ValueError(f"{unit}: duplicate patient ids in the shard")
        rec = {
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "package": self.key.package,
            "attempt": self.key.attempt,
            "unit": unit,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code": self.key.code,
            "inputs": self.key.inputs,
            "definition": self.key.definition,
            "fingerprint": self.key.fingerprint(),
            "patients": pids,
            "n_patients": len(pids),
            "controls": controls or {},
            "wall_s": wall_s,
            "payload": payload,
            "payload_sha256": sha256_obj(payload),
        }
        p = self.path_for(unit)
        h = write_atomic(p, rec)
        self.log(unit, "complete", shard=str(p), shard_sha256=h,
                 n_patients=len(pids), wall_s=wall_s, controls=controls or {})
        return p

    # -- aggregation -------------------------------------------------------- #
    def collect(self, units: list[str], *, require_all: bool = True) -> dict:
        """Gather shards for `units`, refusing an aggregate built from
        incompatible or overlapping work."""
        got, missing, seen = {}, [], {}
        for u in units:
            s = self.load(u)
            if s is None:
                missing.append(u)
                continue
            for pid in s["patients"]:
                if pid in seen:
                    raise ValueError(
                        f"patient {pid} appears in shards {seen[pid]!r} and {u!r}")
                seen[pid] = u
            got[u] = s
        if missing and require_all:
            raise ValueError(f"missing or unusable shards: {missing}")
        return {
            "schema_version": SCHEMA_VERSION,
            "package": self.key.package,
            "attempt": self.key.attempt,
            "fingerprint": self.key.fingerprint(),
            "units_expected": units,
            "units_present": sorted(got),
            "units_missing": missing,
            "n_patients_total": len(seen),
            "shards": got,
        }

    def status(self, units: list[str]) -> dict:
        out = {}
        for u in units:
            p = self.path_for(u)
            if not p.exists():
                out[u] = "not_started"
                continue
            try:
                saved = json.loads(p.read_text())
            except Exception:
                out[u] = "failed"
                continue
            out[u] = "complete" if not self.key.why_incompatible(saved) \
                and saved.get("state") == "complete" else "partial"
        return out

    def reuse_report(self) -> dict:
        return {"hits": sorted(self._hits),
                "misses": {u: why for u, why in self._misses}}
