"""Small synthetic fixtures: no real participant metadata, no public HTTP calls."""
from __future__ import annotations

import hashlib
import contextlib
import io
import json
import datetime as dt
import os
import plistlib
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest import mock

from scripts.nlst_resume_core import (
    ReceiptIndex, Refusal, SEALED_LEGACY_SHA256, exclusive_lock, file_identity, json_digest,
    publish_archive, reconcile_archive,
)
from scripts.nlst_resources import ResourceGuard, StopSession
from scripts import nlst_resources
from scripts import nlst_continue


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

    def test_sealed_original_attempt_partial_is_preserved(self):
        partial = self.root / "1.2.3.4.zip.attempt01.part"
        partial.write_bytes(b"legacy partial")
        result = self.reconcile(self.index())
        self.assertEqual(result["reason"], "unresolved_partial")
        self.assertEqual(result["preserved_partials"], [str(partial)])
        self.assertEqual(partial.read_bytes(), b"legacy partial")

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

    def test_internal_journal_reserve_does_not_reserve_external_archive_bytes(self):
        self.guard.internal = {"root": str(self.root), "device": self.root.stat().st_dev}
        self.guard.reserve_internal = 100
        with mock.patch(
            "scripts.nlst_resources.shutil.disk_usage",
            side_effect=[types.SimpleNamespace(free=n) for n in (4000, 4000, 101)],
        ):
            self.guard.check(extra_bytes=2000)
        with mock.patch(
            "scripts.nlst_resources.shutil.disk_usage",
            side_effect=[types.SimpleNamespace(free=n) for n in (4000, 4000, 99)],
        ):
            with self.assertRaisesRegex(StopSession, "internal_capacity_reserve"):
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


class AdapterFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.paths = {"ROOT": self.root}
        for key in ("PLAN", "INTERNAL", "DEST"):
            self.paths[key] = self.root / key.lower()
            self.paths[key].mkdir()
        self.items = [{
            "series_uid": "1.2.3." + str(rank), "study_uid": "1.2.3",
            "patient_id": "SYNTHETIC" + str(rank), "person_key": "FIXTURE-" + str(rank),
            "rank": rank, "expected_images": 2, "expected_bytes": 1024,
            "license": "CC BY 4.0", "role": "pretraining",
            "in_P1k": True, "in_P3k": True,
        } for rank in (1, 2)]
        self.plan = {
            "items": self.items, "ordered_persons": 2, "uncompressed_bytes": 2048,
            "download_only": True, "pixels_decoded": False, "scientific_acceptance": False,
        }
        for key in ("PLAN", "INTERNAL"):
            (self.paths[key] / "NLST_TRANSFER_RECEIPTS.jsonl").write_text("")
        (self.paths["PLAN"] / "NLST_DOWNLOAD_STATE.json").write_text('{"status":"stopped"}')
        self.args = types.SimpleNamespace(
            session_dir=self.root / "new-session", mirror_root=None, receipt=[],
            command="preflight", max_new_archives=None, max_reuse_archives=None,
        )
        now = dt.datetime.now(dt.timezone.utc)
        self.guard = types.SimpleNamespace(
            theia={"root": str(self.root)}, cutoff=now + dt.timedelta(minutes=5),
            deadline=now + dt.timedelta(minutes=10), stop_reason=None,
            check=lambda **_: None, receipt=lambda: {"synthetic": True},
        )
        self.guard.request_stop = lambda reason: setattr(self.guard, "stop_reason", reason)
        self.contract = {"paths": self.paths, "items": self.items, "plan": self.plan}

    def legacy_archive(self):
        raw = self.items[0]
        path = self.paths["DEST"] / (raw["series_uid"] + ".zip")
        payload = b"fixture archive bytes for receipt hashing only"
        path.write_bytes(payload)
        receipt = {
            **nlst_continue.receipt_row(raw), "status": "archive_verified",
            "path": str(path), "images": raw["expected_images"], "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "zip_crc_pass": True,
            "all_header_identities_match": True,
        }
        for key in ("PLAN", "INTERNAL"):
            (self.paths[key] / "NLST_TRANSFER_RECEIPTS.jsonl").write_text(json.dumps(receipt) + "\n")
        return path

    def index(self):
        return ReceiptIndex(
            [self.paths["PLAN"] / "NLST_TRANSFER_RECEIPTS.jsonl"],
            legacy_verifier_sha256=SEALED_LEGACY_SHA256,
        )

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return nlst_continue.execute_session(self.args, self.contract, self.guard)

    def test_frozen_schema_and_url_bindings(self):
        self.assertEqual(nlst_continue.validate_plan(self.plan), self.items)
        spec = nlst_continue.transfer_spec(self.items[0])
        spec.validate()
        self.assertTrue(spec.url.endswith("getImage?SeriesInstanceUID=1.2.3.1"))
        self.assertEqual(spec.expected.patient_id, "SYNTHETIC1")
        self.assertEqual(spec.expected.study_uid, "1.2.3")
        self.assertTrue(spec.expected.dicom_bytes_only)
        self.assertEqual(spec.expected.non_dicom_members, ("LICENSE",))
        self.assertEqual(spec.max_archive_bytes, 1002048)

    def test_bad_roster_order_or_nested_flag_refuses(self):
        self.items[1]["rank"] = 1
        with self.assertRaisesRegex(Refusal, "order/rank"):
            nlst_continue.validate_plan(self.plan)
        self.items[1]["rank"] = 2
        self.items[1]["in_P1k"] = False
        with self.assertRaisesRegex(Refusal, "nested membership"):
            nlst_continue.validate_plan(self.plan)

    def test_duplicate_patient_identity_refuses(self):
        self.items[1]["patient_id"] = self.items[0]["patient_id"]
        with self.assertRaisesRegex(Refusal, "Duplicate planned identity"):
            nlst_continue.validate_plan(self.plan)

    def test_source_path_extraction_never_executes_original(self):
        source = b"""
ROOT = Path('/synthetic')
PLAN = ROOT / 'plan'
DEST = ROOT / 'archives'
INTERNAL = Path('/synthetic-internal')
raise AssertionError('must not execute')
"""
        self.assertEqual(nlst_continue.source_paths(source)["DEST"], Path("/synthetic/archives"))

    def test_inventory_credits_no_existing_bytes_without_hash(self):
        path = self.legacy_archive()
        before = file_identity(path)
        with mock.patch("scripts.nlst_continue.stable_sha256", side_effect=AssertionError("hash forbidden")):
            initial = nlst_continue.inventory(self.paths["DEST"], self.items, self.index())
        self.assertEqual([r["status"] for r in initial["rows"]], ["existing_unverified", "pending"])
        self.assertEqual(file_identity(path), before)
        records = nlst_continue.SessionRecords([self.root], self.items, initial)
        self.addCleanup(records.close)
        self.assertEqual(records.summary()["verified_archives_this_session"], 0)
        self.assertEqual(records.summary()["verified_bytes_this_session"], 0)

    def test_inventory_partial_names_are_exact_and_include_original_form(self):
        partial = self.paths["DEST"] / "1.2.3.1.zip.attempt01.part"
        partial.write_bytes(b"preserve")
        unrelated = self.paths["DEST"] / "1.2.3.20.zip.part.1"
        unrelated.write_bytes(b"also preserve")
        value = nlst_continue.inventory(self.paths["DEST"], self.items, self.index())
        self.assertEqual(value["rows"][0]["reason"], "unresolved_partial")
        self.assertEqual(value["rows"][1]["status"], "pending")
        self.assertEqual(value["unknown_artifacts"], [str(unrelated)])

    def test_missing_prior_success_is_not_admitted(self):
        self.legacy_archive().unlink()
        value = nlst_continue.inventory(self.paths["DEST"], self.items, self.index())
        self.assertEqual(value["rows"][0]["reason"], "success_receipt_archive_missing")
        self.assertEqual(value["rows"][0]["status"], "review_required")

    def test_preflight_writes_inventory_without_network_or_rehash(self):
        self.legacy_archive()
        with mock.patch("scripts.nlst_continue.run_transfers", side_effect=AssertionError("network")):
            with mock.patch("scripts.nlst_continue.reconcile_archive", side_effect=AssertionError("rehash")):
                self.assertEqual(self.execute(), 0)
        summary = json.loads((self.args.session_dir / "SUMMARY.json").read_text())
        self.assertEqual(summary["verified_archives_this_session"], 0)
        self.assertEqual(summary["counts"], {"existing_unverified": 1, "pending": 1})

    def test_absent_work_can_start_while_existing_hash_is_in_progress(self):
        self.legacy_archive()
        self.args.command = "run"
        hashing = threading.Event()
        release = threading.Event()
        original = nlst_continue.reconcile_archive

        def slow_hash(*args, **kwargs):
            hashing.set()
            self.assertTrue(release.wait(2))
            return original(*args, **kwargs)

        def transfers(specs, _directory, _guard, _record):
            self.assertTrue(hashing.wait(2))
            self.assertEqual([s.raw_row["rank"] for s in specs], [2])
            release.set()
            return []

        with mock.patch("scripts.nlst_continue.reconcile_archive", side_effect=slow_hash):
            with mock.patch("scripts.nlst_continue.run_transfers", side_effect=transfers):
                self.assertEqual(self.execute(), 0)
        summary = json.loads((self.args.session_dir / "SUMMARY.json").read_text())
        self.assertEqual(summary["verified_archives_this_session"], 1)
        self.assertEqual(summary["remaining_queue"][0]["rank"], 2)

    def test_terminal_row_error_returns_nonzero(self):
        self.args.command = "run"

        def transfers(specs, _directory, _guard, record):
            record({**specs[0].row, "status": "http_error", "http_status": 404})
            return []

        with mock.patch("scripts.nlst_continue.run_transfers", side_effect=transfers):
            self.assertEqual(self.execute(), 1)
        summary = json.loads((self.args.session_dir / "SUMMARY.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["counts"]["http_error"], 1)

    def test_reconciliation_refusal_stops_admission_before_journal_callback(self):
        self.legacy_archive()
        self.args.command = "run"
        journal_reached = threading.Event()
        original_record = nlst_continue.SessionRecords.record
        refusal = {
            **nlst_continue.receipt_row(self.items[0]), "status": "review_required",
            "reason": "archive_sha256_mismatch",
        }

        def record(records, journal, row):
            if journal == "RECONCILIATION.jsonl":
                self.assertEqual(self.guard.stop_reason, "terminal_reconciliation_review_required")
                journal_reached.set()
            return original_record(records, journal, row)

        def transfers(_specs, _directory, guard, _record):
            self.assertTrue(journal_reached.wait(2))
            self.assertEqual(guard.stop_reason, "terminal_reconciliation_review_required")
            return []

        with mock.patch("scripts.nlst_continue.reconcile_archive", return_value=refusal):
            with mock.patch.object(nlst_continue.SessionRecords, "record", new=record):
                with mock.patch("scripts.nlst_continue.run_transfers", side_effect=transfers):
                    self.assertEqual(self.execute(), 1)
        summary = json.loads((self.args.session_dir / "SUMMARY.json").read_text())
        self.assertEqual(summary["stop_reason"], "terminal_reconciliation_review_required")
        self.assertEqual(summary["verified_archives_this_session"], 0)
        self.assertEqual(summary["counts"], {"review_required": 1, "pending": 1})

    def test_journal_error_still_preserves_internal_mirror(self):
        second = self.root / "mirror"
        second.mkdir()
        initial = nlst_continue.inventory(self.paths["DEST"], self.items, self.index())
        records = nlst_continue.SessionRecords([self.root, second], self.items, initial)
        self.addCleanup(records.close)
        row = {**nlst_continue.receipt_row(self.items[0]), "status": "http_error"}
        writer = records.journals["TRANSFER_RECEIPTS.jsonl"][0]
        with mock.patch.object(writer, "append", side_effect=OSError("external unavailable")):
            with self.assertRaisesRegex(OSError, "Journal persistence failed"):
                records.record("TRANSFER_RECEIPTS.jsonl", row)
        self.assertEqual(json.loads((second / "TRANSFER_RECEIPTS.jsonl").read_text()), row)


class MountedVolumeFixtures(unittest.TestCase):
    def test_diskutil_receives_mount_point_instead_of_nested_project_root(self):
        # Mock only the observed host filesystem, preserving actual path traversal.
        volume = Path("/synthetic-volume")
        project = volume / "nested" / "project"
        info = plistlib.dumps({"VolumeUUID": "synthetic", "MountPoint": str(volume)})
        # autospec forwards self for path methods patched on the class.
        with mock.patch.object(Path, "resolve", autospec=True, side_effect=lambda p, **_: p):
            with mock.patch.object(Path, "is_mount", autospec=True, side_effect=lambda p: p == volume):
                with mock.patch.object(Path, "stat", autospec=True, side_effect=lambda p: types.SimpleNamespace(
                    st_dev=1 if p == Path("/") else 2,
                )):
                    with mock.patch.object(Path, "is_dir", return_value=True):
                        with mock.patch("scripts.nlst_resources.os.access", return_value=True):
                            with mock.patch("scripts.nlst_resources.shutil.disk_usage", return_value=types.SimpleNamespace(free=1234)):
                                with mock.patch("scripts.nlst_resources.run_command", return_value=info) as probe:
                                    result = nlst_resources.mounted_volume(project, "synthetic")
        self.assertEqual(probe.call_args.args[0][-1], str(volume))
        self.assertEqual(result["root"], str(project))
        self.assertEqual(result["mount_point"], str(volume))


if __name__ == "__main__":
    unittest.main()
