#!/usr/bin/env python3
"""Record one standalone, bounded local attempt; checkpoint logic stays in the job."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

UTC = dt.timezone.utc


def now() -> dt.datetime:
    return dt.datetime.now(UTC)


def utc(value: str) -> dt.datetime:
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Session times must include a UTC offset")
    return result.astimezone(UTC)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    digest = sha(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError(f"Input changed while being hashed: {path}")
    return {"path": str(path), "sha256": digest, "bytes": after.st_size}


def save_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can return EPERM as an orphaned group disappears. Treat it
        # as gone only if a fresh process table has no live member.
        result = subprocess.run(
            ["ps", "-axo", "pgid=,stat="], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == str(group) and not fields[1].startswith("Z"):
                raise
        return False


def signal_group(group: int, sig: int) -> None:
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True,
                        help="JSON with deadline_utc and admission_cutoff_utc")
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--job-key", required=True,
                        help="Stable run/fold identity, also used for exclusive locking")
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, action="append", default=[],
                        help="Explicit small closed outputs to hash after the process exits")
    parser.add_argument("--resume-receipt", type=Path,
                        help="Prior START.json; refuses changed command/source/input identity")
    parser.add_argument("--stop-grace-seconds", type=float, default=10.0,
                        help="Begin SIGINT this many seconds before the deadline")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Provide a literal command after --; shell evaluation is disabled")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.task_id):
        raise ValueError("Unsafe task ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job_key):
        raise ValueError("Unsafe job key")
    if not 0 <= args.stop_grace_seconds <= 60:
        raise ValueError("Stop grace must be between 0 and 60 seconds")
    session_bytes = args.session.read_bytes()
    session = json.loads(session_bytes)
    deadline = utc(session["deadline_utc"])
    cutoff = utc(session["admission_cutoff_utc"])
    if cutoff > deadline or now() >= cutoff:
        raise ValueError("Admission cutoff has passed or exceeds the fixed deadline")
    cwd = args.cwd.resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("Working directory is not a directory")
    # Do not create missing mount paths. The caller must provide an existing root.
    root = args.attempt_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Attempt root must already exist")
    lock = (root / f".{args.job_key}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError(f"Job already has an active writer: {args.job_key}")
    with lock:
        binding = {
            "task_id": args.task_id, "job_key": args.job_key,
            "command": command, "cwd": str(cwd),
            "sources": [identity(p) for p in args.source],
            "inputs": [identity(p) for p in args.input],
        }
        fingerprint = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if args.resume_receipt:
            prior = json.loads(args.resume_receipt.read_text())
            if prior.get("fingerprint") != fingerprint:
                raise ValueError("Resume refused: source/input/command fingerprint changed")
        if now() >= cutoff:
            raise ValueError("Admission cutoff passed during identity verification")
        stamp = now().strftime("%Y%m%dT%H%M%S.%fZ")
        attempt = root / f"{args.task_id}_{args.job_key}_{stamp}"
        attempt.mkdir(exist_ok=False)
        start = {
            "schema": "theia.cpu_attempt.v1", "task_id": args.task_id,
            "job_key": args.job_key, "started_utc": now().isoformat(),
            "deadline_utc": deadline.isoformat(), "cutoff_utc": cutoff.isoformat(),
            "session_sha256": hashlib.sha256(session_bytes).hexdigest(),
            "runner": identity(Path(__file__)),
            "fingerprint": fingerprint, "binding": binding,
            "resume_receipt": str(args.resume_receipt) if args.resume_receipt else None,
            "checkpoint_note": "Job owns checkpoint compatibility and atomic checkpoint writes.",
        }
        save_new(attempt / "START.json", start)
        process = None
        interrupted = []
        handlers = {}

        def request_stop(signum, _frame):
            interrupted.append(signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        start_mono = time.monotonic()
        remaining = max(0.0, (deadline - now()).total_seconds())
        end_mono = start_mono + remaining
        stop_mono = end_mono - args.stop_grace_seconds
        sent = []
        exit_code = None
        failure = None
        reason = "process_exit"
        try:
            with (attempt / "stdout.log").open("xb") as out, (
                attempt / "stderr.log"
            ).open("xb") as err:
                env = dict(os.environ)
                env.update(PYTHONUNBUFFERED="1", THEIA_ATTEMPT_DIR=str(attempt),
                           THEIA_DEADLINE_UTC=deadline.isoformat())
                # Receipt hashing/fsync may take time on external storage. Check
                # admission again at the launch boundary, not just before setup.
                if now() >= cutoff or now() >= deadline:
                    raise ValueError("Admission cutoff passed immediately before launch")
                process = subprocess.Popen(
                    command, cwd=cwd, env=env, stdout=out, stderr=err,
                    start_new_session=True,
                )
                save_new(attempt / "PROCESS.json", {
                    "pid": process.pid, "process_group": process.pid,
                    "launched_utc": now().isoformat(), "command": command,
                    "identity": subprocess.run(
                        ["ps", "-p", str(process.pid), "-o", "pid=,ppid=,lstart=,command="],
                        capture_output=True, text=True, check=False,
                    ).stdout.strip(),
                })
                print(json.dumps({"attempt": str(attempt), "pid": process.pid}), flush=True)
                while process.poll() is None:
                    current = time.monotonic()
                    if (interrupted or current >= stop_mono or now() >= deadline) and not sent:
                        reason = "parent_signal" if interrupted else "fixed_deadline"
                        signal_group(process.pid, signal.SIGINT)
                        sent.append({"signal": "SIGINT", "at_utc": now().isoformat()})
                        # External stop gets bounded grace too; never extends original deadline.
                        end_mono = min(end_mono, current + args.stop_grace_seconds)
                    term_margin = min(0.5, args.stop_grace_seconds / 2)
                    if sent and len(sent) == 1 and current >= end_mono - term_margin:
                        signal_group(process.pid, signal.SIGTERM)
                        sent.append({"signal": "SIGTERM", "at_utc": now().isoformat()})
                    if sent and current >= end_mono:
                        signal_group(process.pid, signal.SIGKILL)
                        sent.append({"signal": "SIGKILL", "at_utc": now().isoformat()})
                        break
                    time.sleep(0.1)
                exit_code = process.wait()
                # Background descendants cannot outlive a foreground task silently.
                # This group was created by this Popen and is never looked up from a
                # stale PID file. It remains allocated while descendants exist.
                if group_alive(process.pid):
                    reason = "remaining_descendants"
                    signal_group(process.pid, signal.SIGTERM)
                    sent.append({"signal": "SIGTERM", "at_utc": now().isoformat()})
                    finish_by = min(end_mono, time.monotonic() + 0.5)
                    while group_alive(process.pid) and time.monotonic() < finish_by:
                        time.sleep(0.05)
                    if group_alive(process.pid):
                        signal_group(process.pid, signal.SIGKILL)
                        sent.append({"signal": "SIGKILL", "at_utc": now().isoformat()})
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            reason = "runner_error"
            if process is not None and process.poll() is None:
                signal_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    signal_group(process.pid, signal.SIGKILL)
                    process.wait()
                exit_code = process.returncode
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            outputs, output_errors = [], []
            for path in args.output:
                try:
                    outputs.append(identity(path))
                except Exception as error:
                    output_errors.append({"path": str(path), "error": str(error)})
            receipt = {
                "schema": "theia.cpu_attempt.v1", "task_id": args.task_id,
                "job_key": args.job_key, "fingerprint": fingerprint,
                "ended_utc": now().isoformat(),
                "elapsed_seconds": time.monotonic() - start_mono,
                "exit_code": exit_code, "stop_reason": reason,
                "signals": sent, "runner_error": failure,
                "outputs": outputs, "output_errors": output_errors,
                "status": "finished" if exit_code == 0 and not sent and not failure
                          and not output_errors else "incomplete_or_failed",
                "scientific_acceptance": "not_inferred_from_exit",
            }
            save_new(attempt / "EXIT.json", receipt)
        if receipt["status"] == "finished":
            return 0
        return exit_code if exit_code is not None and 0 < exit_code < 126 else 1


if __name__ == "__main__":
    try:
        sys.exit(main(arguments()))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
