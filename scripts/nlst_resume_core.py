"""Receipt and storage primitives for the versioned NLST continuation command.

No network operation occurs in this module. Clinical metadata and real receipts
belong outside the source repository.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Callable


SUCCESS = frozenset({"archive_verified", "existing_verified", "reused_verified"})
ROW_FIELDS = (
    "series_uid", "person_key", "rank", "expected_images", "expected_bytes",
    "license", "role", "source",
)
SHA_LENGTH = 64
SEALED_LEGACY_SHA256 = "daf0d770c43fb569f5d290f509d1cc1b8d8fabc10b6102799caf2cc0ac2ff70d"


class Refusal(RuntimeError):
    """An input or safety precondition was not satisfied."""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat()


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise Refusal("Session timestamps must include UTC offset Z or +00:00")
    return parsed


def json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def file_identity(path: Path) -> dict:
    """Reject links/nonregular files; bind both inode and mutable file metadata."""
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise Refusal("Archive must be a regular file, not a symlink: " + str(path))
    return {
        "device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
        "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
    }


def stable_sha256(path: Path, tick: Callable[[], None] = lambda: None) -> tuple[str, dict]:
    before = file_identity(path)
    digest = hashlib.sha256()
    # O_NOFOLLOW prevents a link swap between lstat and open.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if (st.st_dev, st.st_ino) != (before["device"], before["inode"]):
            raise Refusal("File changed before hashing: " + str(path))
        while block := stream.read(4 * 1024 * 1024):
            tick()
            digest.update(block)
    if file_identity(path) != before:
        raise Refusal("File identity changed while hashing: " + str(path))
    return digest.hexdigest(), before


def exclusive_json(path: Path, value: object) -> None:
    """Create a durable JSON artifact without replacing any prior artifact."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Journal:
    """One durable append per terminal attempt or reconciliation decision."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.stream = path.open("x", encoding="utf-8")

    def append(self, row: dict) -> None:
        with self.lock:
            self.stream.write(json.dumps(row, sort_keys=True) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Keep the lock inode in place so all contenders lock the same object."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refusal("Another downloader holds " + str(path)) from exc
        yield
    finally:
        os.close(fd)


class ReceiptIndex:
    def __init__(self, paths: list[Path], *, legacy_verifier_sha256: str | None = None):
        # The caller must inspect/hash the actual sealed validator before opting
        # in. Legacy receipts do not separately serialize the unique-SOP check.
        self.legacy_verifier_sha256 = legacy_verifier_sha256
        self.by_uid: dict[str, list[dict]] = {}
        self.sources: list[dict] = []
        for path in paths:
            digest, identity = stable_sha256(path)
            self.sources.append({"path": str(path.resolve()), "sha256": digest, "identity": identity})
            with path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError) as exc:
                        raise Refusal(f"Malformed receipt JSON at {path}:{number}") from exc
                    if not isinstance(row, dict):
                        raise Refusal(f"Receipt is not an object at {path}:{number}")
                    if row.get("status") in SUCCESS:
                        uid = row.get("series_uid")
                        if not isinstance(uid, str):
                            raise Refusal(f"Successful receipt has no series UID at {path}:{number}")
                        item = dict(row)
                        item["_source"] = {
                            "path": str(path.resolve()), "line": number,
                            "file_sha256": digest, "receipt_sha256": json_digest(row),
                        }
                        self.by_uid.setdefault(uid, []).append(item)
            if file_identity(path) != identity:
                raise Refusal("Receipt journal changed while reading: " + str(path))

    def canonical(self, row: dict, raw_row: dict, plan_sha256: str, path: Path) -> dict:
        candidates = self.by_uid.get(row["series_uid"], [])
        if not candidates:
            raise Refusal("existing_archive_has_no_success_receipt")
        fingerprints = set()
        for receipt in candidates:
            for key in ROW_FIELDS:
                if receipt.get(key) != row[key]:
                    raise Refusal("receipt_plan_mismatch:" + key)
            if (
                receipt.get("zip_crc_pass") is not True
                or receipt.get("all_header_identities_match") is not True
                or receipt.get("images") != row["expected_images"]
            ):
                raise Refusal("receipt_verification_incomplete")
            sha = receipt.get("sha256", "")
            if not isinstance(sha, str) or len(sha) != SHA_LENGTH or any(
                c not in "0123456789abcdef" for c in sha
            ):
                raise Refusal("receipt_hash_invalid")
            size = receipt.get("bytes")
            if type(size) is not int or size <= 0:
                raise Refusal("receipt_size_invalid")
            if receipt.get("path") is not None and receipt["path"] != str(path):
                raise Refusal("receipt_path_mismatch")
            if receipt.get("schema_version") == 2:
                if receipt.get("path") != str(path):
                    raise Refusal("receipt_path_mismatch")
                if receipt.get("plan_sha256") != plan_sha256:
                    raise Refusal("receipt_plan_hash_mismatch")
                if receipt.get("plan_row_sha256") != json_digest(raw_row):
                    raise Refusal("receipt_plan_row_hash_mismatch")
                if receipt.get("unique_sop_pass") is not True:
                    raise Refusal("receipt_unique_sop_missing")
                if not isinstance(receipt.get("file_identity"), dict):
                    raise Refusal("receipt_file_identity_missing")
            elif receipt.get("schema_version") is not None:
                raise Refusal("unsupported_receipt_schema")
            elif self.legacy_verifier_sha256 != SEALED_LEGACY_SHA256:
                raise Refusal("legacy_verifier_not_confirmed")
            fingerprints.add((sha, size))
        if len(fingerprints) != 1:
            raise Refusal("conflicting_success_receipts")
        # Prefer a successor receipt so its prior inode/mtime binding is enforced.
        candidates.sort(key=lambda r: (r.get("schema_version") == 2, r.get("ended_utc", "")))
        return candidates[-1]


def partial_paths(archive_dir: Path, uid: str) -> list[Path]:
    """Match only <UID>.part[.*] and <UID>.zip.part[.*], never prefix UIDs."""
    suffix = re.compile(r"^\.(?:zip\.)?part(?:\..+)?$")
    return sorted(
        (path for path in archive_dir.glob(uid + ".*") if suffix.fullmatch(path.name[len(uid):])),
        key=str,
    )


def reconcile_archive(
    row: dict,
    raw_row: dict,
    plan_sha256: str,
    archive_dir: Path,
    receipts: ReceiptIndex,
    tick: Callable[[], None] = lambda: None,
) -> dict:
    """A full digest is required on every reuse; unknown bytes remain untouched."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", row["series_uid"]):
        raise Refusal("Invalid series UID")
    archive_dir = archive_dir.resolve(strict=True)
    path = archive_dir / (row["series_uid"] + ".zip")
    base = {
        **row, "schema_version": 2, "plan_sha256": plan_sha256,
        "plan_row_sha256": json_digest(raw_row), "path": str(path),
        "ended_utc": iso_now(),
    }
    partials = [str(p) for p in partial_paths(archive_dir, row["series_uid"])]
    base["preserved_partials"] = partials
    if not os.path.lexists(path):
        if row["series_uid"] in receipts.by_uid:
            return {**base, "status": "review_required", "reason": "success_receipt_archive_missing"}
        if partials:
            return {**base, "status": "review_required", "reason": "unresolved_partial"}
        return {**base, "status": "pending"}
    try:
        receipt = receipts.canonical(row, raw_row, plan_sha256, path)
        identity = file_identity(path)
        if identity["size"] != receipt["bytes"]:
            raise Refusal("archive_size_mismatch")
        if receipt.get("schema_version") == 2 and identity != receipt["file_identity"]:
            raise Refusal("archive_file_identity_mismatch")
        digest, identity = stable_sha256(path, tick)
        if digest != receipt["sha256"]:
            raise Refusal("archive_sha256_mismatch")
        return {
            **base, "status": "reused_verified", "sha256": digest,
            "bytes": identity["size"], "file_identity": identity,
            "images": receipt["images"], "zip_crc_pass": True,
            "all_header_identities_match": True, "unique_sop_pass": True,
            "prior_receipt": receipt["_source"],
            "legacy_file_identity_bound_now": receipt.get("schema_version") is None,
            "legacy_verifier_sha256": (
                receipts.legacy_verifier_sha256 if receipt.get("schema_version") is None else None
            ),
            "ended_utc": iso_now(),
        }
    except (OSError, Refusal) as exc:
        return {**base, "status": "review_required", "reason": str(exc)}


def publish_archive(
    partial: Path, final: Path, *, expected_identity: dict, expected_sha256: str,
    tick: Callable[[], None] = lambda: None,
) -> dict:
    """No-clobber publication bound to the exact previously verified bytes.

    Keep the partial until the linked destination is rehashed. A failed identity
    or hash check leaves all suspect names/bytes intact and emits no success.
    """
    if file_identity(partial) != expected_identity:
        raise Refusal("archive_identity_changed_before_publication")
    # rename()/replace() can clobber an unknown archive created since preflight.
    os.link(partial, final, follow_symlinks=False)
    linked = file_identity(final)
    # Creating/removing hard links changes ctime. All other fields must match.
    content_fields = ("device", "inode", "size", "mtime_ns")
    if any(linked[key] != expected_identity[key] for key in content_fields):
        raise Refusal("archive_identity_changed_during_publication")
    fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    digest, validated = stable_sha256(final, tick)
    if digest != expected_sha256:
        raise Refusal("archive_hash_changed_during_publication")
    if file_identity(partial) != validated:
        raise Refusal("partial_identity_changed_during_publication")
    partial.unlink()
    published = file_identity(final)
    if any(published[key] != validated[key] for key in content_fields):
        raise Refusal("archive_identity_changed_after_publication")
    return published
