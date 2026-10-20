"""Bounded HTTP/ZIP primitives; no frozen-plan parser or patient mapping.

Callers supply every source identity and URL after checking the actual frozen
contract. This module never extracts archives or decodes DICOM pixels.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import http.client
import ipaddress
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import threading
import time
from typing import Callable, Iterable
from urllib.parse import urlsplit
import uuid
import zipfile

try:
    from .nlst_resume_core import (
        Refusal, ROW_FIELDS, file_identity, iso_now, json_digest,
        partial_paths, publish_archive, stable_sha256, utcnow,
    )
    from .nlst_resources import StopSession
except ImportError:
    from nlst_resume_core import (
        Refusal, ROW_FIELDS, file_identity, iso_now, json_digest,
        partial_paths, publish_archive, stable_sha256, utcnow,
    )
    from nlst_resources import StopSession


BLOCK = 1024 * 1024
ACCESS_ERRORS = frozenset({401, 403, 429})
RETRYABLE_HTTP = frozenset({500, 502, 503, 504})


class ArchiveRejected(Refusal):
    pass


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        super().__init__("http_status_" + str(status))


@dataclass(frozen=True)
class ArchiveExpectation:
    patient_id: str
    series_uid: str
    expected_images: int
    expected_uncompressed_bytes: int
    # Exact non-image member names only; no wildcard or extension assumptions.
    non_dicom_members: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.patient_id or self.patient_id != self.patient_id.strip():
            raise Refusal("An explicit, nonblank source PatientID is required")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", self.series_uid):
            raise Refusal("An explicit source SeriesInstanceUID is required")
        if type(self.expected_images) is not int or self.expected_images < 1:
            raise Refusal("Expected image count must be a positive integer")
        if type(self.expected_uncompressed_bytes) is not int or self.expected_uncompressed_bytes < 1:
            raise Refusal("Expected uncompressed bytes must be a positive integer")
        if len(set(self.non_dicom_members)) != len(self.non_dicom_members):
            raise Refusal("Duplicate non-DICOM member declaration")


def verify_archive(
    path: Path, expected: ArchiveExpectation,
    tick: Callable[[], None] = lambda: None,
) -> dict:
    """Verify every ZIP member's CRC and every DICOM header without pixel decode."""
    expected.validate()
    import pydicom

    digest, identity = stable_sha256(path, tick)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise ArchiveRejected("duplicate_zip_member_name")
            for info in infos:
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                    raise ArchiveRejected("unsafe_zip_member_path")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ArchiveRejected("zip_symlink_member")
                if info.flag_bits & 1:
                    raise ArchiveRejected("encrypted_zip_member")
                if info.is_dir() and (info.file_size != 0 or info.CRC != 0):
                    raise ArchiveRejected("nonempty_or_invalid_directory_member")
            files = [info for info in infos if not info.is_dir()]
            if not set(expected.non_dicom_members).issubset({i.filename for i in files}):
                raise ArchiveRejected("declared_non_dicom_member_missing")
            images = [i for i in files if i.filename not in expected.non_dicom_members]
            if len(images) != expected.expected_images:
                raise ArchiveRejected("image_count_mismatch")
            if sum(info.file_size for info in files) != expected.expected_uncompressed_bytes:
                raise ArchiveRejected("uncompressed_size_mismatch")
            # Draining each member validates its CRC, including opaque pixel bytes.
            # Decompression of ZIP data is not DICOM pixel decoding.
            for info in files:
                tick()
                size = 0
                with archive.open(info) as stream:
                    while block := stream.read(BLOCK):
                        tick()
                        size += len(block)
                if size != info.file_size:
                    raise ArchiveRejected("member_size_mismatch")
            sops = set()
            for info in images:
                tick()
                with archive.open(info) as stream:
                    header = pydicom.dcmread(
                        stream, stop_before_pixels=True,
                        specific_tags=["PatientID", "SeriesInstanceUID", "SOPInstanceUID"],
                    )
                if str(getattr(header, "PatientID", "")).strip() != expected.patient_id:
                    raise ArchiveRejected("patient_header_mismatch")
                if str(getattr(header, "SeriesInstanceUID", "")).strip() != expected.series_uid:
                    raise ArchiveRejected("series_header_mismatch")
                sop = str(getattr(header, "SOPInstanceUID", "")).strip()
                if not sop:
                    raise ArchiveRejected("missing_sop_instance_uid")
                if sop in sops:
                    raise ArchiveRejected("duplicate_sop_instance_uid")
                sops.add(sop)
    except (zipfile.BadZipFile, EOFError, pydicom.errors.InvalidDicomError) as exc:
        raise ArchiveRejected(type(exc).__name__ + ":" + str(exc)) from exc
    if file_identity(path) != identity:
        raise ArchiveRejected("archive_identity_changed_during_verification")
    return {
        "sha256": digest, "bytes": identity["size"], "file_identity": identity,
        "images": len(images), "uncompressed_bytes": expected.expected_uncompressed_bytes,
        "zip_crc_pass": True, "all_header_identities_match": True,
        "unique_sop_pass": True, "pixels_decoded": False,
    }


@dataclass(frozen=True)
class TransferSpec:
    row: dict
    raw_row: dict
    plan_sha256: str
    url: str
    expected: ArchiveExpectation
    max_archive_bytes: int

    def validate(self) -> None:
        self.expected.validate()
        if any(key not in self.row for key in ROW_FIELDS):
            raise Refusal("Transfer requires a fully normalized receipt row")
        if self.row["series_uid"] != self.expected.series_uid:
            raise Refusal("Transfer and header series identity disagree")
        if self.row["expected_images"] != self.expected.expected_images:
            raise Refusal("Transfer and header image counts disagree")
        if self.row["expected_bytes"] != self.expected.expected_uncompressed_bytes:
            raise Refusal("Transfer and uncompressed byte expectations disagree")
        if not re.fullmatch(r"[a-f0-9]{64}", self.plan_sha256):
            raise Refusal("Transfer requires the full frozen plan SHA-256")
        if type(self.max_archive_bytes) is not int or self.max_archive_bytes < 1:
            raise Refusal("Transfer requires a positive compressed byte limit")


def receive(
    url: str, partial: Path, byte_limit: int, guard, *,
    allow_loopback_http: bool = False, io_timeout: float = 10.0,
) -> dict:
    """One HTTP request; no redirects or hidden retry behavior."""
    parsed = urlsplit(url)
    local_http = False
    if parsed.scheme == "http" and allow_loopback_http:
        try:
            local_http = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            pass
    if parsed.scheme != "https" and not local_http:
        raise Refusal("HTTPS is required outside explicit loopback fixtures")
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise Refusal("Invalid transfer URL")
    if not 0 < io_timeout <= 30:
        raise Refusal("I/O timeout must be in (0, 30] seconds")

    def timeout() -> float:
        guard.check()
        remaining = (guard.deadline - utcnow()).total_seconds()
        if remaining <= 0:
            raise StopSession("deadline_reached")
        return min(io_timeout, remaining)

    def network(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as exc:
            raise ConnectionError(type(exc).__name__ + ":" + str(exc)) from exc

    connection_type = http.client.HTTPConnection if local_http else http.client.HTTPSConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout())
    count = 0
    deadline_timer = None
    try:
        # Admission is checked after URL/connection setup, immediately before I/O.
        guard.check(new_work=True)
        network(connection.connect)
        sock = connection.sock
        if sock is None:
            raise ConnectionError("No connected socket")

        def interrupt_at_deadline():
            # Socket inactivity timeouts alone do not bound a peer that drips
            # header bytes continuously. Interrupt this exact owned socket.
            guard.request_stop("deadline_reached")
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        deadline_timer = threading.Timer(
            max(0, (guard.deadline - utcnow()).total_seconds()), interrupt_at_deadline,
        )
        deadline_timer.daemon = True
        deadline_timer.start()
        sock.settimeout(timeout())
        target = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
        network(connection.request, "GET", target, headers={
            "User-Agent": "THEIA-NLST-continuation/2",
            "Accept": "application/zip, application/octet-stream",
            "Accept-Encoding": "identity",
            "Connection": "close",
        })
        sock.settimeout(timeout())
        response = network(connection.getresponse)
        if response.status != 200:
            raise HTTPStatusError(response.status)
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise Refusal("Unexpected HTTP content encoding")
        declared = response.getheader("Content-Length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError as exc:
                raise Refusal("Invalid HTTP Content-Length") from exc
            if declared_bytes < 1 or declared_bytes > byte_limit:
                raise Refusal("HTTP Content-Length exceeds bound or is empty")
        else:
            declared_bytes = None
        with partial.open("xb") as stream:
            while not response.isclosed():
                sock.settimeout(timeout())
                block = network(response.read1, BLOCK)
                guard.check()
                if not block:
                    break
                count += len(block)
                if count > byte_limit:
                    raise Refusal("HTTP body exceeds compressed byte limit")
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if declared_bytes is not None and declared_bytes != count:
            raise ConnectionError("Incomplete HTTP body")
        if count == 0:
            raise ConnectionError("Empty HTTP body")
        return {"http_status": 200, "bytes_received": count}
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        connection.close()


def transfer_one(
    spec: TransferSpec, archive_dir: Path, guard, record: Callable[[dict], None], *,
    max_attempts: int = 3, retry_delay: float = 2.0,
    allow_loopback_http: bool = False,
) -> dict:
    """Preserve each partial and terminal attempt; retry only bounded transport errors."""
    spec.validate()
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise Refusal("At most three transport attempts are allowed")
    if not 0 <= retry_delay <= 30:
        raise Refusal("Retry delay must be between 0 and 30 seconds")
    archive_dir = archive_dir.resolve(strict=True)
    uid = spec.expected.series_uid
    final = archive_dir / (uid + ".zip")
    base = {
        **spec.row, "schema_version": 2, "plan_sha256": spec.plan_sha256,
        "plan_row_sha256": json_digest(spec.raw_row), "path": str(final),
        "pixels_decoded": False,
    }
    if os.path.lexists(final) or partial_paths(archive_dir, uid):
        result = {**base, "status": "review_required", "reason": "existing_bytes_before_transfer"}
        record(result)
        return result
    reserved = False
    try:
        guard.reserve(uid, spec.max_archive_bytes)
        reserved = True
        for attempt in range(1, max_attempts + 1):
            partial = archive_dir / f"{uid}.zip.part.{uuid.uuid4().hex}.{attempt}"
            entry = {
                **base, "attempt": attempt, "started_utc": iso_now(),
                "partial_path": str(partial),
            }
            retry = False
            started = time.monotonic()
            try:
                received = receive(
                    spec.url, partial, spec.max_archive_bytes, guard,
                    allow_loopback_http=allow_loopback_http,
                )
                entry.update(received)
                verified = verify_archive(partial, spec.expected, guard.check)
                guard.check()
                entry.update(verified)
                entry["file_identity"] = publish_archive(
                    partial, final, expected_identity=verified["file_identity"],
                    expected_sha256=verified["sha256"], tick=guard.check,
                )
                entry["status"] = "archive_verified"
            except HTTPStatusError as exc:
                entry.update(status="http_error", http_status=exc.status, error=str(exc))
                if exc.status in ACCESS_ERRORS:
                    guard.request_stop("terminal_http_" + str(exc.status))
                retry = exc.status in RETRYABLE_HTTP
            except (socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                entry.update(status="transport_error", error=type(exc).__name__ + ":" + str(exc))
                retry = True
            except StopSession as exc:
                entry.update(status="stopped", error=str(exc))
            except Exception as exc:
                entry.update(status="verification_or_io_error", error=type(exc).__name__ + ":" + str(exc))
            entry.update(
                ended_utc=iso_now(), seconds=time.monotonic() - started,
                partial_preserved=os.path.lexists(partial),
                partial_bytes=partial.stat().st_size if partial.exists() else 0,
            )
            record(entry)
            if not retry or attempt == max_attempts:
                return entry
            until = time.monotonic() + retry_delay * attempt
            while time.monotonic() < until:
                guard.check(new_work=True)
                time.sleep(min(0.1, max(0, until - time.monotonic())))
    except StopSession as exc:
        result = {**base, "status": "stopped", "error": str(exc), "ended_utc": iso_now()}
        record(result)
        return result
    except (OSError, Refusal) as exc:
        guard.request_stop("resource_or_input_refusal")
        result = {
            **base, "status": "stopped", "error": type(exc).__name__ + ":" + str(exc),
            "ended_utc": iso_now(),
        }
        record(result)
        return result
    finally:
        if reserved:
            guard.release(uid)
    raise AssertionError("Unreachable transfer state")


def run_transfers(
    specs: Iterable[TransferSpec], archive_dir: Path, guard,
    record: Callable[[dict], None], **options,
) -> list[dict]:
    """Admit in supplied frozen order, with exactly two concurrent worker slots."""
    pending = list(specs)
    seen = set()
    for spec in pending:
        spec.validate()
        if spec.expected.series_uid in seen:
            raise Refusal("Duplicate series in transfer queue")
        seen.add(spec.expected.series_uid)
    iterator = iter(pending)
    results = []
    futures = {}
    exhausted = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while futures or not exhausted:
            while len(futures) < 2 and not exhausted:
                try:
                    guard.check(new_work=True)
                    spec = next(iterator)
                except (StopIteration, StopSession):
                    exhausted = True
                    break
                future = pool.submit(transfer_one, spec, archive_dir, guard, record, **options)
                futures[future] = spec.expected.series_uid
            if futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    results.append(future.result())
    return results
