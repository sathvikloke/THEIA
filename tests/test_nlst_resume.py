"""Small synthetic fixtures: no real participant metadata, no public HTTP calls."""
from __future__ import annotations

import hashlib
import json
import datetime as dt
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts.nlst_resume_core import (
    ReceiptIndex, Refusal, SEALED_LEGACY_SHA256, exclusive_lock, file_identity, json_digest,
    publish_archive, reconcile_archive,
)
from scripts.nlst_resources import ResourceGuard, StopSession


class ResumeFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.row = {
            "series_uid": "1.2.3.4", "person_key": "synthetic-person",
            "rank": 1, "expected_images": 2, "expected_bytes": 1024,
            "license": "synthetic", "role": "pretraining_download_only",
            "source": "synthetic-fixture",
        }
        self.raw = {**self.row, "synthetic_source_field": 7}
        self.plan_sha = "f" * 64
        self.path = self.root / "1.2.3.4.zip"
        self.payload = b"synthetic archive bytes; this fixture only tests receipt matching"

    def receipt(self, **changes):
        record = {
            **self.row, "status": "archive_verified",
            "path": str(self.path), "images": 2, "zip_crc_pass": True,
            "all_header_identities_match": True, "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "ended_utc": "2026-01-01T00:00:00Z",
        }
        return {**record, **changes}

    def index(self, *rows):
        path = self.root / ("receipts-" + str(len(list(self.root.glob("receipts-*")))) + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        # Test-only explicit legacy verifier acknowledgement. Production must
        # inspect and hash the actual sealed source before passing this value.
        return ReceiptIndex([path], legacy_verifier_sha256=SEALED_LEGACY_SHA256)

    def reconcile(self, index):
        return reconcile_archive(self.row, self.raw, self.plan_sha, self.root, index)

    def test_legacy_success_rehashes_and_binds_identity(self):
        self.path.write_bytes(self.payload)
        result = self.reconcile(self.index(self.receipt()))
        self.assertEqual(result["status"], "reused_verified")
        self.assertEqual(result["file_identity"], file_identity(self.path))
        self.assertTrue(result["legacy_file_identity_bound_now"])
        self.assertEqual(result["plan_row_sha256"], json_digest(self.raw))

    def test_legacy_reuse_requires_confirmed_verifier(self):
        self.path.write_bytes(self.payload)
        index = self.index(self.receipt())
        index.legacy_verifier_sha256 = None
        self.assertEqual(self.reconcile(index)["reason"], "legacy_verifier_not_confirmed")

    def test_successor_receipt_is_reusable(self):
        self.path.write_bytes(self.payload)
        first = self.reconcile(self.index(self.receipt()))
        second = self.reconcile(self.index(self.receipt(), first))
        self.assertEqual(second["status"], "reused_verified")
        self.assertFalse(second["legacy_file_identity_bound_now"])

    def test_unknown_archive_is_preserved(self):
        self.path.write_bytes(self.payload)
        before = file_identity(self.path)
        result = self.reconcile(self.index())
        self.assertEqual(result["reason"], "existing_archive_has_no_success_receipt")
        self.assertEqual(file_identity(self.path), before)
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_equal_size_corruption_is_rejected(self):
        self.path.write_bytes(b"X" + self.payload[1:])
        result = self.reconcile(self.index(self.receipt()))
        self.assertEqual(result["reason"], "archive_sha256_mismatch")

    def test_mismatched_plan_field_is_rejected(self):
        self.path.write_bytes(self.payload)
        result = self.reconcile(self.index(self.receipt(person_key="different-person")))
        self.assertEqual(result["reason"], "receipt_plan_mismatch:person_key")

    def test_missing_archive_with_success_is_not_redownloaded(self):
        result = self.reconcile(self.index(self.receipt()))
        self.assertEqual(result["reason"], "success_receipt_archive_missing")

    def test_partial_is_preserved_and_requires_review(self):
        partial = self.root / "1.2.3.4.zip.part.earlier-attempt"
        partial.write_bytes(b"partial")
        result = self.reconcile(self.index())
        self.assertEqual(result["reason"], "unresolved_partial")
        self.assertEqual(partial.read_bytes(), b"partial")

    def test_other_uid_partials_do_not_block_a_fresh_row(self):
        names = ["1.2.3.40.zip.part.old.1", "1.2.3.4.5.zip.part.old.1"]
        for name in names:
            (self.root / name).write_bytes(b"unrelated")
        result = self.reconcile(self.index())
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["preserved_partials"], [])
        self.assertTrue(all((self.root / name).read_bytes() == b"unrelated" for name in names))

    def test_fresh_missing_row_is_pending(self):
        self.assertEqual(self.reconcile(self.index())["status"], "pending")

    def test_conflicting_success_receipts_refuse(self):
        self.path.write_bytes(self.payload)
        result = self.reconcile(self.index(self.receipt(), self.receipt(sha256="a" * 64)))
        self.assertEqual(result["reason"], "conflicting_success_receipts")

    def test_changed_inode_is_rejected_even_when_bytes_match(self):
        self.path.write_bytes(self.payload)
        first = self.reconcile(self.index(self.receipt()))
        self.path.rename(self.root / "preserved-old-file")
        self.path.write_bytes(self.payload)
        result = self.reconcile(self.index(first))
        self.assertEqual(result["reason"], "archive_file_identity_mismatch")

    def test_changed_unprojected_plan_field_refuses_successor_receipt(self):
        self.path.write_bytes(self.payload)
        first = self.reconcile(self.index(self.receipt()))
        self.raw["synthetic_source_field"] = 8
        result = self.reconcile(self.index(first))
        self.assertEqual(result["reason"], "receipt_plan_row_hash_mismatch")

    def test_link_archive_is_rejected(self):
        other = self.root / "real-file"
        other.write_bytes(self.payload)
        self.path.symlink_to(other)
        result = self.reconcile(self.index(self.receipt()))
        self.assertEqual(result["status"], "review_required")
        self.assertIn("not a symlink", result["reason"])

    def test_lock_contention_refuses(self):
        path = self.root / "download.lock"
        with exclusive_lock(path):
            with self.assertRaises(Refusal):
                with exclusive_lock(path):
                    self.fail("contending lock was acquired")
        with exclusive_lock(path):
            pass

    def test_publication_cannot_replace_destination(self):
        partial = self.root / "partial"
        partial.write_bytes(b"new")
        self.path.write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            publish_archive(
                partial, self.path, expected_identity=file_identity(partial),
                expected_sha256=hashlib.sha256(b"new").hexdigest(),
            )
        self.assertEqual(self.path.read_bytes(), b"old")
        self.assertEqual(partial.read_bytes(), b"new")

    def test_publication_preserves_inode(self):
        partial = self.root / "partial"
        partial.write_bytes(self.payload)
        before = file_identity(partial)
        identity = publish_archive(
            partial, self.path, expected_identity=before,
            expected_sha256=hashlib.sha256(self.payload).hexdigest(),
        )
        self.assertEqual(identity["inode"], before["inode"])
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertFalse(partial.exists())

    def test_same_size_change_before_publication_is_refused(self):
        partial = self.root / "partial"
        partial.write_bytes(self.payload)
        expected = file_identity(partial)
        partial.write_bytes(b"X" + self.payload[1:])
        with self.assertRaisesRegex(Refusal, "identity_changed_before_publication"):
            publish_archive(
                partial, self.path, expected_identity=expected,
                expected_sha256=hashlib.sha256(self.payload).hexdigest(),
            )
        self.assertTrue(partial.exists())
        self.assertFalse(self.path.exists())

    def test_replacement_inode_before_publication_is_refused(self):
        partial = self.root / "partial"
        partial.write_bytes(self.payload)
        expected = file_identity(partial)
        partial.rename(self.root / "old-partial")
        partial.write_bytes(self.payload)
        with self.assertRaisesRegex(Refusal, "identity_changed_before_publication"):
            publish_archive(
                partial, self.path, expected_identity=expected,
                expected_sha256=hashlib.sha256(self.payload).hexdigest(),
            )
        self.assertTrue(partial.exists())
        self.assertTrue((self.root / "old-partial").exists())
        self.assertFalse(self.path.exists())

    def test_mutation_during_link_preserves_both_names_without_success(self):
        partial = self.root / "partial"
        partial.write_bytes(self.payload)
        expected = file_identity(partial)
        changed = b"X" + self.payload[1:]
        real_link = os.link

        def corrupt_link(*args, **kwargs):
            real_link(*args, **kwargs)
            partial.write_bytes(changed)
            # Even restoring mtime cannot conceal the changed digest.
            os.utime(partial, ns=(partial.stat().st_atime_ns, expected["mtime_ns"]))

        with mock.patch("scripts.nlst_resume_core.os.link", side_effect=corrupt_link):
            with self.assertRaisesRegex(Refusal, "hash_changed_during_publication"):
                publish_archive(
                    partial, self.path, expected_identity=expected,
                    expected_sha256=hashlib.sha256(self.payload).hexdigest(),
                )
        self.assertEqual(partial.read_bytes(), changed)
        self.assertEqual(self.path.read_bytes(), changed)

    def test_truncated_journal_refuses(self):
        path = self.root / "bad.jsonl"
        path.write_text('{"status":')
        with self.assertRaisesRegex(Refusal, "Malformed receipt JSON"):
            ReceiptIndex([path])


class ResourceFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        self.volume = {
            "root": str(self.root), "device": self.root.stat().st_dev,
            "mount_point": str(self.root), "volume_uuid": "fixture", "free_bytes": 4000,
        }
        self.patches = [
            mock.patch("scripts.nlst_resources.mounted_volume", side_effect=lambda *a: dict(self.volume)),
            mock.patch("scripts.nlst_resources.verify_backing_image", return_value="fixture-image"),
            mock.patch("scripts.nlst_resources.utcnow", return_value=self.now),
            mock.patch("scripts.nlst_resources.power_is_ac", return_value=True),
            mock.patch("scripts.nlst_resources.shutil.disk_usage", return_value=types.SimpleNamespace(free=4000)),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.guard = ResourceGuard(
            theia_root=self.root, research_root=self.root, backing_image=self.root,
            theia_uuid="fixture", research_uuid="fixture",
            cutoff=self.now + dt.timedelta(minutes=10),
            deadline=self.now + dt.timedelta(minutes=20),
            reserve_theia=100, reserve_research=100,
        )

    def test_cutoff_stops_new_work_but_allows_current_work(self):
        with mock.patch("scripts.nlst_resources.utcnow", return_value=self.guard.cutoff):
            self.guard.check()
            with self.assertRaisesRegex(StopSession, "cutoff_reached"):
                self.guard.check(new_work=True)
            self.assertFalse(self.guard.stop.is_set())
            self.guard.check()

    def test_deadline_stops_current_work(self):
        with mock.patch("scripts.nlst_resources.utcnow", return_value=self.guard.deadline):
            with self.assertRaisesRegex(StopSession, "deadline_reached"):
                self.guard.check()

    def test_capacity_accounts_for_both_active_transfers(self):
        self.guard.reserve("first", 2000)
        with self.assertRaisesRegex(StopSession, "capacity_reserve"):
            self.guard.reserve("second", 2000)
        self.guard.release("first")
        self.assertTrue(self.guard.stop.is_set())
        with self.assertRaisesRegex(StopSession, "capacity_reserve"):
            self.guard.reserve("second", 1)

    def test_research_backing_capacity_is_checked_separately(self):
        with mock.patch(
            "scripts.nlst_resources.shutil.disk_usage",
            side_effect=[types.SimpleNamespace(free=4000), types.SimpleNamespace(free=99)],
        ):
            with self.assertRaisesRegex(StopSession, "research_capacity_reserve"):
                self.guard.check()

    def test_power_loss_stops_work(self):
        self.guard.last_power_check = float("-inf")
        with mock.patch("scripts.nlst_resources.power_is_ac", return_value=False):
            with self.assertRaisesRegex(StopSession, "ac_power_unavailable"):
                self.guard.check()
        self.assertTrue(self.guard.stop.is_set())
        with mock.patch("scripts.nlst_resources.power_is_ac", return_value=True):
            with self.assertRaisesRegex(StopSession, "ac_power_unavailable"):
                self.guard.check(new_work=True)

    def test_mount_identity_change_stops_work(self):
        self.guard.theia["device"] = -1
        with self.assertRaisesRegex(StopSession, "theia_mount_identity_changed"):
            self.guard.check()
        self.guard.theia["device"] = self.root.stat().st_dev
        with self.assertRaisesRegex(StopSession, "theia_mount_identity_changed"):
            self.guard.check(new_work=True)

    def test_deadline_crossing_inside_probe_is_terminal(self):
        current = [self.now]

        def slow_probe(**_):
            current[0] = self.guard.deadline + dt.timedelta(seconds=1)
            return True

        self.guard.last_power_check = float("-inf")
        with mock.patch("scripts.nlst_resources.utcnow", side_effect=lambda: current[0]):
            with mock.patch("scripts.nlst_resources.power_is_ac", side_effect=slow_probe):
                with self.assertRaisesRegex(StopSession, "deadline_reached"):
                    self.guard.check()
        self.assertTrue(self.guard.stop.is_set())

    def test_cutoff_crossing_inside_probe_does_not_stop_admitted_work(self):
        current = [self.now]

        def slow_probe(**_):
            current[0] = self.guard.cutoff + dt.timedelta(seconds=1)
            return True

        self.guard.last_power_check = float("-inf")
        with mock.patch("scripts.nlst_resources.utcnow", side_effect=lambda: current[0]):
            with mock.patch("scripts.nlst_resources.power_is_ac", side_effect=slow_probe):
                with self.assertRaisesRegex(StopSession, "cutoff_reached"):
                    self.guard.check(new_work=True)
                self.assertFalse(self.guard.stop.is_set())
                self.guard.check()

    def test_power_probe_budget_uses_remaining_session_time(self):
        self.guard.deadline = self.now + dt.timedelta(seconds=0.25)
        self.guard.last_power_check = float("-inf")
        with mock.patch("scripts.nlst_resources.power_is_ac", return_value=True) as probe:
            self.guard.check()
        self.assertGreater(probe.call_args.kwargs["timeout_seconds"], 0)
        self.assertLessEqual(probe.call_args.kwargs["timeout_seconds"], 0.25)

    def test_global_stop_during_probe_is_observed_before_return(self):
        def stop_probe(**_):
            self.guard.request_stop("during_probe")
            return True

        self.guard.last_power_check = float("-inf")
        with mock.patch("scripts.nlst_resources.power_is_ac", side_effect=stop_probe):
            with self.assertRaisesRegex(StopSession, "during_probe"):
                self.guard.check()

    def test_disappearing_mount_stops_work(self):
        with mock.patch("scripts.nlst_resources.Path.stat", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(StopSession, "theia_mount_unavailable"):
                self.guard.check()

    def test_external_stop_remains_terminal(self):
        self.guard.request_stop("terminal_http_403")
        with self.assertRaisesRegex(StopSession, "terminal_http_403"):
            self.guard.check()


if __name__ == "__main__":
    unittest.main()
