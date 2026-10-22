#!/usr/bin/env python3
"""Continue only the sealed, selected NLST roster with receipt-bound acquisition."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlencode
import uuid

try:
    from .nlst_resume_core import (
        Journal, ReceiptIndex, Refusal, SEALED_LEGACY_SHA256, SUCCESS, exclusive_json,
        exclusive_lock, file_identity, iso_now, json_digest, parse_utc, same_persistent_identity,
        reconcile_archive, stable_sha256, utcnow,
    )
    from .nlst_resources import ResourceGuard, StopSession
    from .nlst_transfer import ArchiveExpectation, TransferSpec, run_transfers
except ImportError:
    from nlst_resume_core import (
        Journal, ReceiptIndex, Refusal, SEALED_LEGACY_SHA256, SUCCESS, exclusive_json,
        exclusive_lock, file_identity, iso_now, json_digest, parse_utc, same_persistent_identity,
        reconcile_archive, stable_sha256, utcnow,
    )
    from nlst_resources import ResourceGuard, StopSession
    from nlst_transfer import ArchiveExpectation, TransferSpec, run_transfers


SEALED_PLAN_SHA256 = "746c7f24c8fb5cb39ee8a5d29130a3847d181082386562fa53b0b5cffe36971b"
NLST_ENDPOINT = "https://nlst.cancerimagingarchive.net/nbia-api/services/v1/getImage"
SOURCE_DOI = "https://doi.org/10.7937/TCIA.HMQ8-J677"
UID = r"[0-9]+(?:\.[0-9]+)+"
PARTIAL_NAME = re.compile(
    r"^(?P<uid>" + UID + r")(?:\.(?:zip\.)?part(?:\..+)?|\.zip\.attempt[0-9]+\.part)$"
)


def checked_bytes(path: Path, expected: str) -> bytes:
    before = file_identity(path)
    content = path.read_bytes()
    if file_identity(path) != before:
        raise Refusal("Source changed while reading: " + str(path))
    if hashlib.sha256(content).hexdigest() != expected:
        raise Refusal("Sealed source SHA-256 mismatch: " + str(path))
    return content


def source_paths(source: bytes) -> dict[str, Path]:
    """Read the sealed script's literal path expressions without executing it."""
    paths = {}

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in paths:
            return paths[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "Path" and len(node.args) == 1 and not node.keywords:
                return Path(evaluate(node.args[0]))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return Path(evaluate(node.left)) / evaluate(node.right)
        raise Refusal("Unsupported sealed source path expression")

    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in {"ROOT", "PLAN", "DEST", "INTERNAL"}:
                paths[target.id] = Path(evaluate(statement.value))
    if set(paths) != {"ROOT", "PLAN", "DEST", "INTERNAL"}:
        raise Refusal("Sealed source path contract is incomplete")
    return paths


def validate_plan(plan: dict) -> list[dict]:
    """Validate the actual frozen schema and preserve its order and nested flags."""
    if not isinstance(plan, dict) or plan.get("download_only") is not True:
        raise Refusal("Plan must authorize archive-only acquisition")
    if plan.get("pixels_decoded") is not False or plan.get("scientific_acceptance") is not False:
        raise Refusal("Plan role flags disagree with archive-only scope")
    items = plan.get("items")
    if not isinstance(items, list) or not items or plan.get("ordered_persons") != len(items):
        raise Refusal("Plan item count mismatch")
    seen = {key: set() for key in ("series_uid", "person_key", "patient_id")}
    total = 0
    for rank, row in enumerate(items, 1):
        if not isinstance(row, dict) or type(row.get("rank")) is not int or row["rank"] != rank:
            raise Refusal("Frozen roster order/rank mismatch")
        for key in ("series_uid", "study_uid"):
            if not isinstance(row.get(key), str) or not re.fullmatch(UID, row[key]):
                raise Refusal("Invalid source identity: " + key)
        for key in ("person_key", "patient_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise Refusal("Missing source identity: " + key)
        for key, values in seen.items():
            if row[key] in values:
                raise Refusal("Duplicate planned identity: " + key)
            values.add(row[key])
        if row.get("role") != "pretraining" or row.get("license") != "CC BY 4.0":
            raise Refusal("Frozen source role/license mismatch")
        for key, limit in (("in_P1k", 1000), ("in_P3k", 3000)):
            if type(row.get(key)) is not bool or row[key] != (rank <= limit):
                raise Refusal("Frozen nested membership mismatch: " + key)
        for key in ("expected_images", "expected_bytes"):
            if type(row.get(key)) is not int or row[key] <= 0:
                raise Refusal("Invalid planned count: " + key)
        total += row["expected_bytes"]
    if plan.get("uncompressed_bytes") != total:
        raise Refusal("Frozen uncompressed DICOM byte total mismatch")
    return items


def load_contract(original: Path, plan_path: Path) -> dict:
    source = checked_bytes(original, SEALED_LEGACY_SHA256)
    plan = json.loads(checked_bytes(plan_path, SEALED_PLAN_SHA256))
    items = validate_plan(plan)
    if len(items) != 10000:
        raise Refusal("The sealed selected roster must contain exactly 10000 persons")
    paths = source_paths(source)
    return {"paths": paths, "plan": plan, "items": items}


def receipt_row(raw: dict) -> dict:
    return {
        **{key: raw[key] for key in (
            "series_uid", "person_key", "rank", "expected_images", "expected_bytes", "license",
        )},
        "role": "pretraining_download_only", "source": SOURCE_DOI,
    }


def transfer_spec(raw: dict) -> TransferSpec:
    return TransferSpec(
        receipt_row(raw), raw, SEALED_PLAN_SHA256,
        NLST_ENDPOINT + "?" + urlencode({"SeriesInstanceUID": raw["series_uid"]}),
        ArchiveExpectation(
            patient_id=raw["patient_id"], study_uid=raw["study_uid"],
            series_uid=raw["series_uid"], expected_images=raw["expected_images"],
            expected_uncompressed_bytes=raw["expected_bytes"],
            non_dicom_members=("LICENSE",), dicom_bytes_only=True,
        ),
        # This is the actual sealed predecessor's HTTP response byte bound.
        max_archive_bytes=2 * raw["expected_bytes"] + 1_000_000,
    )


def inventory(archive_dir: Path, items: list[dict], receipts: ReceiptIndex) -> dict:
    """One directory scan; no archive hashing and no acceptance of existing bytes."""
    by_uid = {row["series_uid"]: row for row in items}
    finals, partials, unknown = {}, {}, []
    for path in archive_dir.iterdir():
        if path.name.endswith(".zip") and path.name[:-4] in by_uid:
            finals[path.name[:-4]] = path
        else:
            match = PARTIAL_NAME.fullmatch(path.name)
            if match and match["uid"] in by_uid:
                partials.setdefault(match["uid"], []).append(path)
            else:
                unknown.append(str(path))
    if set(receipts.by_uid) - set(by_uid):
        raise Refusal("Successful receipts include identities outside the frozen roster")
    rows = []
    for raw in items:
        row = receipt_row(raw)
        uid = row["series_uid"]
        base = {
            **row, "schema_version": 2, "plan_sha256": SEALED_PLAN_SHA256,
            "plan_row_sha256": json_digest(raw),
            "path": str(archive_dir / (uid + ".zip")),
            "preserved_partials": sorted(str(p) for p in partials.get(uid, [])),
        }
        if uid in finals:
            try:
                prior = receipts.canonical(row, raw, SEALED_PLAN_SHA256, finals[uid])
                identity = file_identity(finals[uid])
                if identity["size"] != prior["bytes"]:
                    raise Refusal("archive_size_mismatch")
                if prior.get("schema_version") == 2 and not same_persistent_identity(
                    identity, prior["file_identity"]
                ):
                    raise Refusal("archive_file_identity_mismatch")
                status = "existing_unverified"
                reason = "awaiting_full_archive_hash"
            except (OSError, Refusal) as exc:
                status, reason = "review_required", str(exc)
        elif uid in receipts.by_uid:
            status, reason = "review_required", "success_receipt_archive_missing"
        elif uid in partials:
            status, reason = "review_required", "unresolved_partial"
        else:
            status, reason = "pending", "absent_frozen_roster_row"
        rows.append({**base, "status": status, "reason": reason})
    return {"rows": rows, "unknown_artifacts": sorted(unknown)}


class SessionRecords:
    def __init__(self, directories: list[Path], items: list[dict], initial: dict):
        self.directories = directories
        self.items = items
        self.rows = {row["series_uid"]: row for row in initial["rows"]}
        self.unknown = initial["unknown_artifacts"]
        self.lock = threading.RLock()
        self.journals = {name: [Journal(d / name) for d in directories] for name in (
            "TRANSFER_RECEIPTS.jsonl", "RECONCILIATION.jsonl",
        )}
        self.last_state = float("-inf")
        self.events = 0

    def summary(self) -> dict:
        with self.lock:
            accepted = {uid for uid, row in self.rows.items() if row["status"] in SUCCESS}
            counts = {}
            for row in self.rows.values():
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            return {
                "updated_utc": iso_now(), "counts": counts,
                "verified_archives_this_session": len(accepted),
                "verified_bytes_this_session": sum(self.rows[uid]["bytes"] for uid in accepted),
                "nested_verified": {
                    "P1k": sum(row["in_P1k"] for row in self.items if row["series_uid"] in accepted),
                    "P3k": sum(row["in_P3k"] for row in self.items if row["series_uid"] in accepted),
                    "P10k": len(accepted),
                },
                "target_archives": len(self.items), "terminal_events": self.events,
                "pixels_decoded": False, "newly_evaluable_egfr_patients": 0,
                "acquisition_complete": len(accepted) == len(self.items),
                "unknown_artifacts": self.unknown,
            }

    def write_state(self, **extra):
        value = {**self.summary(), **extra}
        errors = []
        for directory in self.directories:
            try:
                temp = directory / (".STATE." + uuid.uuid4().hex + ".tmp")
                exclusive_json(temp, value)
                os.replace(temp, directory / "STATE.json")
            except OSError as exc:
                errors.append(str(exc))
        self.last_state = time.monotonic()
        if errors:
            raise OSError("State persistence failed: " + "; ".join(errors))

    def record(self, journal: str, row: dict):
        with self.lock:
            errors = []
            for writer in self.journals[journal]:
                try:
                    writer.append(row)
                except OSError as exc:
                    errors.append(str(exc))
            self.rows[row["series_uid"]] = row
            self.events += 1
            if errors:
                raise OSError("Journal persistence failed: " + "; ".join(errors))
            if time.monotonic() - self.last_state >= 30:
                self.write_state(status="running")

    def close(self):
        for writers in self.journals.values():
            for writer in writers:
                try:
                    writer.close()
                except OSError:
                    # Persistence failures are reported by append/finalization.
                    pass


def execute_session(args, contract: dict, guard) -> int:
    paths, items = contract["paths"], contract["items"]
    archive_dir = paths["DEST"].resolve(strict=True)
    if archive_dir.stat().st_dev != Path(guard.theia["root"]).stat().st_dev:
        raise Refusal("Archive directory is not on the verified THEIA volume")
    session_dir = args.session_dir.parent.resolve(strict=True) / args.session_dir.name
    if not session_dir.parent.is_relative_to(Path(guard.theia["root"])):
        raise Refusal("Session outputs must be on verified external THEIA storage")
    if session_dir.parent.stat().st_dev != archive_dir.stat().st_dev:
        raise Refusal("Session output backing device mismatch")
    if args.mirror_root is not None and args.mirror_root.resolve(strict=True).stat().st_dev != (
        paths["INTERNAL"].stat().st_dev
    ):
        raise Refusal("Small journal mirror must share the guarded internal backing device")
    original_receipts = [
        paths["PLAN"] / "NLST_TRANSFER_RECEIPTS.jsonl",
        paths["INTERNAL"] / "NLST_TRANSFER_RECEIPTS.jsonl",
    ]
    prior_outputs = [
        p for pattern in ("*/TRANSFER_RECEIPTS.jsonl", "*/RECONCILIATION.jsonl")
        for p in session_dir.parent.glob(pattern)
    ]
    receipt_paths = list(dict.fromkeys(p.resolve(strict=True) for p in (
        original_receipts + prior_outputs + args.receipt
    )))
    with exclusive_lock(paths["INTERNAL"] / "download.lock"):
        guard.check(new_work=True)
        receipts = ReceiptIndex(receipt_paths, legacy_verifier_sha256=SEALED_LEGACY_SHA256)
        initial = inventory(archive_dir, items, receipts)
        guard.check(new_work=True)
        session_dir.mkdir(exist_ok=False)
        directories = [session_dir]
        if args.mirror_root is not None:
            mirror = args.mirror_root.resolve(strict=True) / ("D01_" + session_dir.name)
            mirror.mkdir(exist_ok=False)
            directories.append(mirror)
        source_hashes = {
            name: stable_sha256(Path(__file__).with_name(name))[0] for name in (
                "nlst_continue.py", "nlst_resume_core.py", "nlst_resources.py", "nlst_transfer.py",
            )
        }
        run = {
            "schema_version": 2, "command": args.command, "argv": sys.argv,
            "started_utc": iso_now(), "pid": os.getpid(),
            "process_identity": subprocess.run(
                ["ps", "-p", str(os.getpid()), "-o", "pid=,ppid=,lstart=,command="],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout.strip(),
            "original_sha256": SEALED_LEGACY_SHA256, "plan_sha256": SEALED_PLAN_SHA256,
            "source_sha256": source_hashes, "receipt_inputs": receipts.sources,
            "paths": {k: str(v) for k, v in paths.items()}, "resources": guard.receipt(),
            "cutoff_utc": guard.cutoff.isoformat(), "deadline_utc": guard.deadline.isoformat(),
            "concurrent_transfers": 2, "legacy_hash_workers": 1,
            "max_new_archives": args.max_new_archives, "max_reuse_archives": args.max_reuse_archives,
            "legacy_state_observation": json.loads((paths["PLAN"] / "NLST_DOWNLOAD_STATE.json").read_text()),
            "acquisition_enabled": args.command == "run", "pixels_decoded": False,
        }
        for directory in directories:
            exclusive_json(directory / "RUN.json", run)
        # The complete private inventory stays external; the internal mirror is small.
        exclusive_json(session_dir / "INVENTORY.json", initial)
        records = SessionRecords(directories, items, initial)
        handlers = {}
        failure = None
        stop_reason = None
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, lambda signum, _frame: guard.request_stop(
                    "signal_" + signal.Signals(signum).name
                ))
            records.write_state(status="preflight_complete" if args.command == "preflight" else "running")
            if args.command != "preflight":
                existing = [raw for raw in items if records.rows[raw["series_uid"]]["status"] == "existing_unverified"]
                missing = [raw for raw in items if records.rows[raw["series_uid"]]["status"] == "pending"]
                if args.max_reuse_archives is not None:
                    existing = existing[:args.max_reuse_archives]
                if args.max_new_archives is not None:
                    missing = missing[:args.max_new_archives]

                def hash_existing():
                    try:
                        for raw in existing:
                            guard.check(new_work=True)
                            prior = records.rows[raw["series_uid"]]
                            result = reconcile_archive(
                                receipt_row(raw), raw, SEALED_PLAN_SHA256, archive_dir, receipts,
                                guard.check, known_partials=[Path(p) for p in prior["preserved_partials"]],
                            )
                            if result["status"] == "review_required":
                                guard.request_stop("terminal_reconciliation_review_required")
                            records.record("RECONCILIATION.jsonl", result)
                            if result["status"] == "review_required":
                                return
                    except StopSession:
                        raise
                    except BaseException:
                        guard.request_stop("reconciliation_worker_error")
                        raise

                with ThreadPoolExecutor(max_workers=1) as hashing:
                    future = hashing.submit(hash_existing)
                    try:
                        if args.command == "run":
                            # Missing rows can proceed while old bytes are still unverified.
                            run_transfers(
                                [transfer_spec(row) for row in missing], archive_dir, guard,
                                lambda row: records.record("TRANSFER_RECEIPTS.jsonl", row),
                            )
                        future.result()
                    except BaseException:
                        guard.request_stop("session_worker_stop")
                        raise
        except StopSession as exc:
            stop_reason = str(exc)
        except Exception as exc:
            failure = type(exc).__name__ + ":" + str(exc)
            guard.request_stop("session_error")
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            stop_reason = stop_reason or guard.stop_reason
            row_failures = any(
                row["status"] in {
                    "http_error", "transport_error", "verification_or_io_error", "review_required",
                } for row in records.rows.values()
            )
            abnormal_stop = stop_reason is not None and stop_reason not in {
                "cutoff_reached", "deadline_reached",
            }
            failed = bool(failure or row_failures or abnormal_stop)
            summary = {
                **records.summary(), "ended_utc": iso_now(), "status": "failed" if failed else "session_closed",
                "command": args.command, "runner_error": failure, "stop_reason": stop_reason,
                "remaining_queue": [
                    {"rank": raw["rank"], "series_uid": raw["series_uid"],
                     "status": records.rows[raw["series_uid"]]["status"]}
                    for raw in items if records.rows[raw["series_uid"]]["status"] not in SUCCESS
                ],
            }
            persistence_errors = []
            try:
                records.write_state(status=summary["status"], stop_reason=stop_reason, runner_error=failure)
            except OSError as exc:
                persistence_errors.append(str(exc))
            for directory in directories:
                try:
                    exclusive_json(directory / "SUMMARY.json", {
                        **summary, "persistence_errors": list(persistence_errors),
                    })
                except OSError as exc:
                    persistence_errors.append(str(exc))
            records.close()
        print(json.dumps({
            "session_dir": str(session_dir), **records.summary(),
            "status": "failed" if failed or persistence_errors else summary["status"],
            "stop_reason": stop_reason, "runner_error": failure, "persistence_errors": persistence_errors,
        }, sort_keys=True))
        return 1 if failed or persistence_errors else 0


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check-bindings", "preflight", "reconcile", "run"):
        command = commands.add_parser(name)
        command.add_argument("--original-downloader", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--cutoff-utc", required=True)
        command.add_argument("--deadline-utc", required=True)
        if name != "check-bindings":
            command.add_argument("--research-root", type=Path, required=True)
            command.add_argument("--backing-image", type=Path, required=True)
            command.add_argument("--theia-uuid", required=True)
            command.add_argument("--research-uuid", required=True)
            command.add_argument("--reserve-theia-bytes", type=int, required=True)
            command.add_argument("--reserve-research-bytes", type=int, required=True)
            command.add_argument("--reserve-internal-bytes", type=int, required=True)
            command.add_argument("--session-dir", type=Path, required=True,
                                 help="New directory under an existing external parent")
            command.add_argument("--mirror-root", type=Path)
            command.add_argument("--receipt", type=Path, action="append", default=[])
            command.add_argument("--max-new-archives", type=int)
            command.add_argument("--max-reuse-archives", type=int)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    try:
        cutoff, deadline = parse_utc(args.cutoff_utc), parse_utc(args.deadline_utc)
        if not utcnow() < cutoff < deadline:
            raise Refusal("Require current UTC < cutoff UTC < deadline UTC")
        contract = load_contract(args.original_downloader, args.plan)
        if args.command == "check-bindings":
            print(json.dumps({
                "status": "contract_verified", "acquisition_started": False,
                "original_sha256": SEALED_LEGACY_SHA256, "plan_sha256": SEALED_PLAN_SHA256,
                "planned_persons": len(contract["items"]),
                "expected_uncompressed_dicom_bytes": contract["plan"]["uncompressed_bytes"],
                "paths": {key: str(path) for key, path in contract["paths"].items()},
            }, sort_keys=True))
            return 0
        for limit in (args.max_new_archives, args.max_reuse_archives):
            if limit is not None and limit < 0:
                raise Refusal("Archive limits must be nonnegative")
        if args.command == "run":
            import pydicom
            if pydicom.__version__ != "3.0.2":
                raise Refusal("The acquired archive validator requires pinned pydicom 3.0.2")
        guard = ResourceGuard(
            theia_root=contract["paths"]["ROOT"], research_root=args.research_root,
            backing_image=args.backing_image, theia_uuid=args.theia_uuid,
            research_uuid=args.research_uuid, cutoff=cutoff, deadline=deadline,
            reserve_theia=args.reserve_theia_bytes, reserve_research=args.reserve_research_bytes,
            internal_root=contract["paths"]["INTERNAL"],
            reserve_internal=args.reserve_internal_bytes,
        )
        return execute_session(args, contract, guard)
    except (OSError, ValueError, ImportError, Refusal, StopSession) as exc:
        print(json.dumps({
            "status": "refused", "acquisition_started": False,
            "reason": type(exc).__name__ + ":" + str(exc),
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
