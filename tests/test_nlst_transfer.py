"""Local synthetic ZIP/DICOM and loopback-only transport fixtures."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile

from scripts.nlst_resources import StopSession
from scripts.nlst_resume_core import Refusal, utcnow
from scripts.nlst_transfer import (
    ArchiveExpectation, ArchiveRejected, TransferSpec, receive,
    run_transfers, transfer_one, verify_archive,
)
from scripts import nlst_transfer

try:
    import pydicom
except ImportError:
    pydicom = None


def dicom_bytes(patient="SYNTHETIC", series="1.2.3.4", sop="1.2.3.4.1"):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient
    ds.SeriesInstanceUID = series
    ds.StudyInstanceUID = "1.2.3"
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop
    ds.Modality = "CT"
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = b"\0" * 8
    out = io.BytesIO()
    pydicom.dcmwrite(out, ds, enforce_file_format=True)
    return out.getvalue()


def make_archive(images=None, series="1.2.3.4", extra=None):
    if images is None:
        images = [
            dicom_bytes(series=series, sop=series + ".1"),
            dicom_bytes(series=series, sop=series + ".2"),
        ]
    members = {f"series/image-{n}.dcm": data for n, data in enumerate(images)}
    members.update(extra or {})
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return out.getvalue(), sum(len(data) for data in members.values())


class FakeGuard:
    def __init__(self):
        self.deadline = utcnow() + dt.timedelta(seconds=15)
        self.cutoff = self.deadline - dt.timedelta(seconds=1)
        self.stopped = None
        self.active = {}
        self.lock = threading.RLock()

    def check(self, new_work=False, **_):
        if self.stopped:
            raise StopSession(self.stopped)
        if utcnow() >= self.deadline:
            raise StopSession("deadline_reached")
        if new_work and utcnow() >= self.cutoff:
            raise StopSession("cutoff_reached")

    def request_stop(self, reason):
        self.stopped = reason

    def reserve(self, key, size):
        with self.lock:
            self.check(new_work=True)
            if key in self.active:
                raise Refusal("duplicate fixture reservation")
            self.active[key] = size

    def release(self, key):
        with self.lock:
            self.active.pop(key)


@contextmanager
def local_server(responder):
    observations = {"requests": [], "active": 0, "peak": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with lock:
                observations["requests"].append(self.path)
                number = len(observations["requests"])
                observations["active"] += 1
                observations["peak"] = max(observations["peak"], observations["active"])
            try:
                responder(self, number)
            finally:
                with lock:
                    observations["active"] -= 1

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), observations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def respond(handler, payload=b"", status=200, declared=None):
    handler.send_response(status)
    handler.send_header("Content-Length", str(len(payload) if declared is None else declared))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(payload)


@unittest.skipUnless(pydicom is not None, "pydicom is required for real DICOM header fixtures")
class ArchiveFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root / "fixture.zip"
        self.payload, self.size = make_archive()
        self.expected = ArchiveExpectation("SYNTHETIC", "1.2.3.4", 2, self.size)
        self.path.write_bytes(self.payload)

    def test_full_verification_never_decodes_pixels(self):
        # Accessing pixel_array would fail even for this valid synthetic DICOM.
        with mock.patch.object(pydicom.dataset.Dataset, "pixel_array", new=property(
            lambda _: (_ for _ in ()).throw(AssertionError("pixel decode attempted"))
        )):
            with mock.patch.object(pydicom, "dcmread", wraps=pydicom.dcmread) as reader:
                receipt = verify_archive(self.path, self.expected)
        self.assertEqual(receipt["images"], 2)
        self.assertTrue(receipt["unique_sop_pass"])
        self.assertTrue(receipt["zip_crc_pass"])
        self.assertFalse(receipt["pixels_decoded"])
        self.assertEqual(reader.call_count, 2)
        self.assertTrue(all(call.kwargs["stop_before_pixels"] for call in reader.call_args_list))

    def test_second_header_identity_is_checked(self):
        payload, size = make_archive([
            dicom_bytes(sop="1.2.3.4.1"),
            dicom_bytes(patient="WRONG", sop="1.2.3.4.2"),
        ])
        self.path.write_bytes(payload)
        with self.assertRaisesRegex(ArchiveRejected, "patient_header_mismatch"):
            verify_archive(self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.4", 2, size))

    def test_series_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ArchiveRejected, "series_header_mismatch"):
            verify_archive(self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.99", 2, self.size))

    def test_duplicate_sop_is_rejected(self):
        image = dicom_bytes()
        payload, size = make_archive([image, image])
        self.path.write_bytes(payload)
        with self.assertRaisesRegex(ArchiveRejected, "duplicate_sop_instance_uid"):
            verify_archive(self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.4", 2, size))

    def test_exact_uncompressed_size_is_required(self):
        with self.assertRaisesRegex(ArchiveRejected, "uncompressed_size_mismatch"):
            verify_archive(self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.4", 2, self.size + 1))

    def test_exact_image_count_is_required(self):
        with self.assertRaisesRegex(ArchiveRejected, "image_count_mismatch"):
            verify_archive(self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.4", 3, self.size))

    def test_crc_corruption_is_rejected(self):
        altered = bytearray(self.payload)
        start = altered.index(b"PK\x01\x02")
        old = struct.unpack_from("<I", altered, start + 16)[0]
        struct.pack_into("<I", altered, start + 16, old ^ 1)
        self.path.write_bytes(altered)
        with self.assertRaisesRegex(ArchiveRejected, "Bad CRC"):
            verify_archive(self.path, self.expected)

    def test_non_dicom_members_must_be_explicit(self):
        payload, size = make_archive(extra={"license.txt": b"synthetic license"})
        self.path.write_bytes(payload)
        with self.assertRaisesRegex(ArchiveRejected, "image_count_mismatch"):
            verify_archive(self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.4", 2, size))
        result = verify_archive(
            self.path, ArchiveExpectation("SYNTHETIC", "1.2.3.4", 2, size, ("license.txt",))
        )
        self.assertTrue(result["zip_crc_pass"])

    def test_directory_member_cannot_hide_unverified_bytes(self):
        with zipfile.ZipFile(self.path, "a") as archive:
            archive.writestr("hidden/", b"unverified directory bytes")
        with self.assertRaisesRegex(ArchiveRejected, "nonempty_or_invalid_directory"):
            verify_archive(self.path, self.expected)

    def test_archive_mutation_during_verification_is_rejected(self):
        old_reader = pydicom.dcmread

        def mutate(*args, **kwargs):
            result = old_reader(*args, **kwargs)
            self.path.touch()
            return result

        with mock.patch.object(pydicom, "dcmread", side_effect=mutate):
            with self.assertRaisesRegex(ArchiveRejected, "archive_identity_changed"):
                verify_archive(self.path, self.expected)


@unittest.skipUnless(pydicom is not None, "pydicom is required for transfer verification fixtures")
class TransportFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.payload, self.size = make_archive()
        self.guard = FakeGuard()
        self.records = []

    def spec(self, url, uid="1.2.3.4", size=None):
        row = {
            "series_uid": uid, "person_key": "SYNTHETIC-" + uid,
            "rank": 1, "expected_images": 2, "expected_bytes": size or self.size,
            "license": "synthetic", "role": "pretraining_download_only",
            "source": "synthetic-fixture",
        }
        return TransferSpec(
            row, dict(row), "f" * 64, url,
            ArchiveExpectation("SYNTHETIC", uid, 2, size or self.size),
            64 * 1024,
        )

    def run_one(self, spec, **options):
        return transfer_one(
            spec, self.root, self.guard, self.records.append,
            allow_loopback_http=True, retry_delay=0, **options,
        )

    def test_success_publishes_only_verified_archive(self):
        with local_server(lambda h, _: respond(h, self.payload)) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "archive_verified")
        self.assertEqual(len(seen["requests"]), 1)
        self.assertEqual((self.root / "1.2.3.4.zip").read_bytes(), self.payload)
        self.assertFalse(result["partial_preserved"])
        self.assertEqual(self.guard.active, {})

    def test_401_403_429_are_not_retried(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                self.guard = FakeGuard()
                with local_server(lambda h, _: respond(h, status=status)) as (url, seen):
                    result = self.run_one(self.spec(url))
                self.assertEqual(result["status"], "http_error")
                self.assertEqual(len(seen["requests"]), 1)
                self.assertEqual(self.guard.stopped, "terminal_http_" + str(status))

    def test_transport_interruption_retries_and_retains_partial(self):
        def responder(handler, number):
            if number == 1:
                respond(handler, self.payload[:100], declared=len(self.payload))
            else:
                respond(handler, self.payload)

        with local_server(responder) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "archive_verified")
        self.assertEqual(len(seen["requests"]), 2)
        self.assertEqual(self.records[0]["status"], "transport_error")
        first = Path(self.records[0]["partial_path"])
        self.assertEqual(first.read_bytes(), self.payload[:100])
        self.assertEqual(self.records[0]["partial_bytes"], 100)

    def test_503_retries_are_bounded(self):
        with local_server(lambda h, _: respond(h, status=503)) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(len(seen["requests"]), 3)
        self.assertEqual(result["status"], "http_error")
        self.assertEqual(len(self.records), 3)

    def test_corrupt_archive_is_not_retried_or_published(self):
        with local_server(lambda h, _: respond(h, b"not a zip")) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "verification_or_io_error")
        self.assertEqual(len(seen["requests"]), 1)
        self.assertFalse((self.root / "1.2.3.4.zip").exists())
        self.assertEqual(Path(result["partial_path"]).read_bytes(), b"not a zip")

    def test_unknown_existing_bytes_prevent_network(self):
        partial = self.root / "1.2.3.4.zip.part.before"
        partial.write_bytes(b"must preserve")
        with local_server(lambda h, _: respond(h, self.payload)) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "review_required")
        self.assertEqual(seen["requests"], [])
        self.assertEqual(partial.read_bytes(), b"must preserve")

    def test_other_uid_partial_does_not_prevent_transfer(self):
        partial = self.root / "1.2.3.40.zip.part.old.1"
        partial.write_bytes(b"unrelated")
        with local_server(lambda h, _: respond(h, self.payload)) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "archive_verified")
        self.assertEqual(len(seen["requests"]), 1)
        self.assertEqual(partial.read_bytes(), b"unrelated")

    def test_mutation_after_verification_cannot_receive_success(self):
        real_verify = nlst_transfer.verify_archive

        def mutate_after_verify(path, expected, tick):
            result = real_verify(path, expected, tick)
            path.write_bytes(b"changed after verification")
            return result

        with local_server(lambda h, _: respond(h, self.payload)) as (url, _):
            with mock.patch.object(nlst_transfer, "verify_archive", side_effect=mutate_after_verify):
                result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "verification_or_io_error")
        self.assertIn("identity_changed_before_publication", result["error"])
        self.assertEqual(Path(result["partial_path"]).read_bytes(), b"changed after verification")
        self.assertFalse((self.root / "1.2.3.4.zip").exists())

    def test_expired_cutoff_prevents_network(self):
        self.guard.cutoff = utcnow() - dt.timedelta(seconds=1)
        with local_server(lambda h, _: respond(h, self.payload)) as (url, seen):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(seen["requests"], [])

    def test_new_destination_during_transfer_is_never_replaced(self):
        def responder(handler, _):
            (self.root / "1.2.3.4.zip").write_bytes(b"foreign archive")
            respond(handler, self.payload)

        with local_server(responder) as (url, _):
            result = self.run_one(self.spec(url))
        self.assertEqual(result["status"], "verification_or_io_error")
        self.assertEqual((self.root / "1.2.3.4.zip").read_bytes(), b"foreign archive")
        self.assertTrue(result["partial_preserved"])

    def test_two_concurrent_transfers_and_ordered_admission(self):
        payloads = {uid: make_archive(series=uid) for uid in ("1.2.3.10", "1.2.3.11", "1.2.3.12")}

        def responder(handler, _):
            time.sleep(0.1)
            respond(handler, payloads[handler.path[1:]][0])

        with local_server(responder) as (url, seen):
            specs = [self.spec(url + "/" + uid, uid, data[1]) for uid, data in payloads.items()]
            result = run_transfers(
                specs, self.root, self.guard, self.records.append,
                allow_loopback_http=True, retry_delay=0,
            )
        self.assertEqual(seen["peak"], 2)
        # Threads may issue the first two requests in either order; the third is
        # not admitted until a slot from the first frozen-order pair is released.
        self.assertEqual(set(seen["requests"][:2]), {"/1.2.3.10", "/1.2.3.11"})
        self.assertEqual(seen["requests"][2], "/1.2.3.12")
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row["status"] == "archive_verified" for row in result))

    def test_retry_wait_cannot_extend_cutoff(self):
        self.guard.cutoff = utcnow() + dt.timedelta(seconds=0.1)
        with local_server(lambda h, _: respond(h, status=503)) as (url, seen):
            result = transfer_one(
                self.spec(url), self.root, self.guard, self.records.append,
                allow_loopback_http=True, retry_delay=0.3,
            )
        self.assertEqual(len(seen["requests"]), 1)
        self.assertEqual(result["status"], "stopped")

    def test_stalled_body_cannot_run_past_deadline(self):
        def responder(handler, _):
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(self.payload)))
            handler.end_headers()
            handler.wfile.write(self.payload[:100])
            handler.wfile.flush()
            time.sleep(0.4)

        self.guard.deadline = utcnow() + dt.timedelta(seconds=0.15)
        self.guard.cutoff = self.guard.deadline - dt.timedelta(seconds=0.02)
        with local_server(responder) as (url, seen):
            start = time.monotonic()
            result = self.run_one(self.spec(url))
            elapsed = time.monotonic() - start
        self.assertEqual(result["status"], "stopped")
        self.assertLess(elapsed, 0.35)
        self.assertEqual(len(seen["requests"]), 1)
        self.assertFalse((self.root / "1.2.3.4.zip").exists())
        self.assertEqual(len(list(self.root.glob("*.part.*"))), 1)

    def test_access_error_stops_admission_of_remaining_queue(self):
        def responder(handler, _):
            if handler.path == "/1.2.3.10":
                respond(handler, status=403)
            else:
                time.sleep(0.1)
                respond(handler, b"stopped in flight")

        with local_server(responder) as (url, seen):
            specs = [self.spec(url + "/" + uid, uid) for uid in (
                "1.2.3.10", "1.2.3.11", "1.2.3.12",
            )]
            run_transfers(
                specs, self.root, self.guard, self.records.append,
                allow_loopback_http=True, retry_delay=0,
            )
        self.assertEqual(self.guard.stopped, "terminal_http_403")
        self.assertNotIn("/1.2.3.12", seen["requests"])
        self.assertLessEqual(len(seen["requests"]), 2)

    def test_dripping_headers_cannot_extend_absolute_deadline(self):
        def responder(handler, _):
            try:
                handler.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 123\r\nX-Padding: ")
                handler.wfile.flush()
                for _ in range(10):
                    handler.wfile.write(b"a")
                    handler.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

        self.guard.deadline = utcnow() + dt.timedelta(seconds=0.15)
        self.guard.cutoff = self.guard.deadline - dt.timedelta(seconds=0.02)
        with local_server(responder) as (url, seen):
            start = time.monotonic()
            result = self.run_one(self.spec(url))
            elapsed = time.monotonic() - start
        self.assertEqual(result["status"], "stopped")
        self.assertLess(elapsed, 0.35)
        self.assertEqual(len(seen["requests"]), 1)
        self.assertFalse((self.root / "1.2.3.4.zip").exists())


class TransportBoundaryFixtures(unittest.TestCase):
    def test_plaintext_public_url_is_refused_before_network(self):
        with self.assertRaisesRegex(Refusal, "HTTPS is required"):
            receive("http://example.com/image.zip", Path("/unused"), 100, FakeGuard())

    def test_unbound_cli_refuses_unverified_original(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = root / "original.py"
            plan = root / "plan.json"
            original.write_text("# synthetic original\n")
            plan.write_text("{}")
            future = utcnow() + dt.timedelta(hours=1)
            script = Path(__file__).resolve().parents[1] / "scripts" / "nlst_continue.py"
            result = subprocess.run([
                sys.executable, str(script), "check-bindings",
                "--original-downloader", str(original), "--plan", str(plan),
                "--cutoff-utc", future.isoformat(),
                "--deadline-utc", (future + dt.timedelta(minutes=1)).isoformat(),
            ], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            value = json.loads(result.stdout)
            self.assertEqual(value["status"], "refused")
            self.assertFalse(value["acquisition_enabled"])
            self.assertIn("original downloader SHA-256 mismatch", value["reason"])


if __name__ == "__main__":
    unittest.main()
