"""Tiny synthetic metadata tests; no research arrays, model imports, or downloads."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "standalone_receipts", REPO / "execution_support/n08/standalone_receipts.py")
sr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sr)
ORIGINAL_MODULE = sr.Evidence.module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True))
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def test_directory():
    # Fail clearly instead of allocating fixtures on a nearly full internal disk.
    path = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if not str(path).startswith("/Volumes/"):
        raise RuntimeError("Set TMPDIR to the authorized external fixture directory")
    return tempfile.TemporaryDirectory(prefix="native-receipts-test-", dir=path)


class NativeFixture:
    """Synthetic producer envelopes with real matrix and pinned source contracts.

    Runtime probing and the frozen split SHA are patched only in test code.
    The synthetic accepted verdict has no scientific authority; full-mode tests
    stub the acceptance helper and test its invocation/refusal boundaries.
    """
    def __init__(self, directory):
        self.root = Path(directory) / "research"
        self.out = self.root / "revamp/cpu_runs/test"
        for directory in (self.out / "tmp", self.out / "attempts", self.out / "units"):
            directory.mkdir(parents=True)
        self.driver = ORIGINAL_MODULE(sr.Evidence(), "driver")
        self.dm = ORIGINAL_MODULE(sr.Evidence(), "data_manifest")
        put(self.root / "theia/engine/train.py",
            b'TRAINING_CODE = ("theia/engine/train.py",)\n')
        put(self.root / "configs/default.yaml", b"synthetic: true\n")
        put(self.root / ".venv/bin/python", b"synthetic runtime bytes\n")
        self.runtime = {"python": "synthetic", "executable": str(self.root / ".venv/bin/python"),
                        "packages": {}, "python_binary_sha256": sha(self.root / ".venv/bin/python")}
        self.manifest = self.root / self.driver.PF / "source/manifest_S1_v1/CONTENT_MANIFEST.json"
        self.rows = self.manifest.with_name("rows.jsonl")
        self.acceptance = self.root / self.driver.PF / "compute/M1_ACCEPTANCE_S1A_v1.json"
        self.splits = self.root / self.driver.SPLITS
        self.npz = put(self.root / "data/processed/SYNTHETIC.npz", b"not an actual array\n")
        put(self.rows, b'{"patient_id":"SYNTHETIC","npz":"data/processed/SYNTHETIC.npz",'
                       b'"labels":{"EGFR":1},"has_mask":true}\n')
        partition = {"indices": [0], "patient_ids": ["SYNTHETIC"], "egfr_n": 1,
                     "egfr_n_pos": 1, "sha256_patient_ids": sr.digest(["SYNTHETIC"])}
        put(self.splits, {"rows_file": {"n_rows": 1}, "folds": [
            {"fold": f, "train_microbatches_per_epoch": 1,
             **{p: copy.deepcopy(partition) for p in ("inner_train", "inner_val", "outer_test")}}
            for f in range(5)]})
        self.driver.SPLITS_SHA = sha(self.splits)
        body = {"dataset_version": "M1", "parent_version": "synthetic", "rows_path": str(self.rows),
                "rows_sha256": sha(self.rows), "n_rows": 1, "preprocessing": {"synthetic": True},
                "entries": [{"patient_id": "SYNTHETIC", "npz_path": "data/processed/SYNTHETIC.npz",
                             "npz_sha256": sha(self.npz), "npz_bytes": self.npz.stat().st_size,
                             "has_mask": True}]}
        body["manifest_body_sha256"] = self.dm.body_sha(body)
        put(self.manifest, body)
        checker = put(self.root / "synthetic/checker.py", b"# synthetic acceptance checker\n")
        registry = put(self.root / "synthetic/registry.json", {"synthetic": True})
        put(self.acceptance, {
            "decision": "accepted", "test_fixture_run": False, "options": {"allow_test_fixtures": False},
            "checks": [{"status": "pass", "synthetic": True}],
            "manifest": str(self.manifest.relative_to(self.root)),
            "manifest_body_sha256": body["manifest_body_sha256"],
            "evidence_sha256": {str(self.manifest.relative_to(self.root)): sha(self.manifest)},
            "checker": {"path": str(checker.relative_to(self.root)), "sha256": sha(checker)},
            "registry": {"path": str(registry.relative_to(self.root)), "sha256": sha(registry)},
        })
        self.data = self.dm.verify_manifest(self.manifest, self.rows, "M1", self.root,
                                            self.acceptance, rehash=False)
        self.base_inputs = [sr.SUPPORT / "FIT_MATRIX_M1.csv", self.splits, self.manifest,
                            self.rows, self.acceptance, checker, registry]
        self.units = {}
        self.make_unit("smoke", 0)

    def load_module(self, evidence, name):
        module = ORIGINAL_MODULE(evidence, name)
        if name == "driver":
            module.SPLITS_SHA = sha(self.splits)
        return module

    def patches(self):
        fixture = self
        return (
            mock.patch.object(sr, "runtime_identity", return_value=self.runtime),
            mock.patch.object(sr.Evidence, "module",
                              lambda evidence, name: fixture.load_module(evidence, name)),
            mock.patch.object(sr.sys, "executable", str(self.root / ".venv/bin/python")),
        )

    def make_unit(self, kind="fit", order=1):
        row = self.driver.select_row(sr.SUPPORT / "FIT_MATRIX_M1.csv", kind, order)
        unit = self.out / "units" / row["fit_id"]
        unit.mkdir()
        fold = self.out / "checkpoints" / row["run_id"] / ("fold" + row["fold"])
        session = put(unit / "SESSION.json", {
            "admission_cutoff_utc": self.driver.FIXED_CUTOFF, "deadline_utc": self.driver.FIXED_DEADLINE})
        control = put(unit / "COOP_CONTROL.json", {
            "deadline_utc": self.driver.FIXED_STOP, "check_every_steps": 1,
            "min_free_bytes": 50_000_000_000, "ckpt_forecast_bytes": 351227877,
            "fit_bytes_budget": 1100000000, "max_battery_pause_s": 0, "epoch_forecast_s": 0})
        sources = sr.source_paths(self.root, sr.Evidence())
        binding = {
            "schema": "theia.corrected_cpu.execution_binding.v1", "created_utc": "2026-10-20T09:00:00Z",
            "root": str(self.root), "output_root": str(self.out), "unit_dir": str(unit),
            "kind": kind, "row": row, "support_dir": str(sr.SUPPORT),
            "python": str(self.root / ".venv/bin/python"), "data": self.data,
            "manifest": str(self.manifest), "rows": str(self.rows), "data_acceptance": str(self.acceptance),
            "splits": str(self.splits), "source_sha256": {str(p): sha(p) for p in sources},
            "input_sha256": {str(p): sha(p) for p in [*self.base_inputs, session, control]},
            "runtime": self.runtime, "scientific_acceptance": self.driver.SCIENCE,
            "encoder_sha256": self.driver.ENCODER_SHA, "admission_cutoff_utc": self.driver.FIXED_CUTOFF,
            "cooperative_stop_utc": self.driver.FIXED_STOP, "deadline_utc": self.driver.FIXED_DEADLINE,
            "resource_policy": {"external_reserve_bytes": 50_000_000_000}, "fold_dir": str(fold),
            "trainer_argv": self.driver.build_argv(row, self.root, self.out, control),
            "environment": {**json.loads(row["env_json"]), "PYTHONDONTWRITEBYTECODE": "1",
                            "TMPDIR": str(self.out / "tmp")},
        }
        if kind == "fit":
            smoke = self.units["smoke"]
            paths = [smoke["unit"] / n for n in ("EXECUTION_BINDING.json", "RESULT.json", "WORKER_RESULT.json")]
            paths.append(smoke["exit"])
            binding["smoke_dependency"] = {str(p): sha(p) for p in paths}
            binding["input_sha256"].update(binding["smoke_dependency"])
        put(unit / "EXECUTION_BINDING.json", binding)
        artifact_ids = {}
        for key, name in sr.ARTIFACTS.items():
            # Unique tiny placeholder bytes, deliberately not scientific output.
            payload = (row["fit_id"] + " " + name + "\n").encode()
            artifact_ids[key] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        data = {**self.data, "npz_rehashed": True, "verified_utc": "2026-10-20T09:02:00Z"}
        fi = {
            **{k: data[k] for k in sr.DATA_KEYS[:5]}, "init_seed": int(row["init_seed"]),
            "partition_seed": 1337, "initial_state_sha256": sr.digest(row["init_seed"]),
            "objective": {"region_pooling": row["region_pooling"], "ground_weight": float(row["ground_weight"]),
                          "ground_target": row["ground_target"], "readout_map": row["readout_map"]},
            "selection_rule": {"monitor": '[["egfr_logloss", -1.0]]', "min_epochs": 10,
                               "patience": 8, "max_epochs": 2 if kind == "smoke" else 40, "ties": "earliest"},
            **{k + "_sha256": artifact_ids[k]["sha256"] for k in ("best_pt", "last_pt")},
        }
        worker = {
            "schema": "theia.corrected_cpu.technical_receipt.v1", "kind": kind, "fit_id": row["fit_id"],
            **{k: row[k] for k in ("run_id", "cell", "fold", "init_seed")},
            "execution_binding": str(unit / "EXECUTION_BINDING.json"),
            "execution_binding_sha256": sha(unit / "EXECUTION_BINDING.json"), "status": "accepted",
            "trainer_exit": 0, "canonical_adoption": False, "scientific_fit_count": int(kind == "fit"),
            "ended_utc": "2026-10-20T09:03:00Z" if kind == "smoke" else "2026-10-20T09:08:00Z",
            "data": data, "fit_identity": fi,
            "fit_identity_sha256": sr.digest(fi),
            "acceptance": {"accepted": True, "failed_checks": [], "fit_id": row["fit_id"],
                           "fold_dir": str(fold), "identities": artifact_ids,
                           "checks": [{"check": sr.VOLATILE_CHECK, "ok": True, "detail": data["verified_utc"]},
                                      {"check": "synthetic-only verdict", "ok": True, "detail": None}]},
        }
        put(unit / "WORKER_RESULT.json", worker)
        task = "M03" if kind == "smoke" else "M04"
        attempt = self.out / "attempts" / (task + "_" + row["fit_id"] + "_synthetic")
        command = [binding["python"], str(sr.FILES["driver"]), "_worker", "--binding",
                   str(unit / "EXECUTION_BINDING.json")]
        launch = {"task_id": task, "job_key": row["fit_id"], "command": command, "cwd": str(self.root),
                  "sources": [identity(p) for p in sources], "inputs": [identity(unit / "EXECUTION_BINDING.json")]}
        put(attempt / "START.json", {
            "schema": "theia.cpu_attempt.v1", "task_id": task, "job_key": row["fit_id"],
            "started_utc": "2026-10-20T09:01:00Z" if kind == "smoke" else "2026-10-20T09:06:00Z",
            "deadline_utc": self.driver.FIXED_DEADLINE,
            "cutoff_utc": self.driver.FIXED_CUTOFF, "session_sha256": sha(session),
            "runner": identity(sr.FILES["wrapper"]), "fingerprint": sr.digest(launch),
            "binding": launch, "resume_receipt": None})
        put(attempt / "PROCESS.json", {"pid": 9999, "process_group": 9999,
            "launched_utc": "2026-10-20T09:01:01Z" if kind == "smoke" else "2026-10-20T09:06:01Z",
            "command": command,
            "identity": "9999 9998 Tue Oct 20 09:01:01 2026 synthetic fixture child"})
        exit_path = put(attempt / "EXIT.json", {
            "schema": "theia.cpu_attempt.v1", "task_id": task, "job_key": row["fit_id"],
            "fingerprint": sr.digest(launch),
            "ended_utc": "2026-10-20T09:04:00Z" if kind == "smoke" else "2026-10-20T09:09:00Z",
            "elapsed_seconds": 180, "exit_code": 0, "stop_reason": "process_exit", "signals": [],
            "runner_error": None, "outputs": [identity(unit / "WORKER_RESULT.json")], "output_errors": [],
            "status": "finished", "scientific_acceptance": "not_inferred_from_exit"})
        result = put(unit / "RESULT.json", {
            "schema": "theia.corrected_cpu.result.v1", "kind": kind, "fit_id": row["fit_id"],
            "status": "accepted", "wrapper_exit": 0, "resource_stops": [], "e05_exit_receipt": str(exit_path),
            "technical_receipt": str(unit / "WORKER_RESULT.json"),
            "execution_binding_sha256": sha(unit / "EXECUTION_BINDING.json"),
            "scientific_fit_count": int(kind == "fit"), "canonical_adoption": False,
            "ended_utc": "2026-10-20T09:05:00Z" if kind == "smoke" else "2026-10-20T09:10:00Z"})
        value = {"unit": unit, "fold": fold, "exit": exit_path, "result": result, "binding": binding, "worker": worker}
        self.units["smoke" if kind == "smoke" else order] = value
        return value

    def materialize_outputs(self, value):
        for name in sr.ARTIFACTS.values():
            put(value["fold"] / name, (value["binding"]["row"]["fit_id"] + " " + name + "\n").encode())

    def rebind(self, value):
        """Recompute envelope hashes after intentionally edited synthetic content."""
        unit = value["unit"]
        worker = json.loads((unit / "WORKER_RESULT.json").read_text())
        worker["execution_binding_sha256"] = sha(unit / "EXECUTION_BINDING.json")
        put(unit / "WORKER_RESULT.json", worker)
        start_path = value["exit"].with_name("START.json")
        start = json.loads(start_path.read_text())
        start["binding"]["inputs"] = [identity(unit / "EXECUTION_BINDING.json")]
        start["fingerprint"] = sr.digest(start["binding"])
        put(start_path, start)
        terminal = json.loads(value["exit"].read_text())
        terminal["fingerprint"] = start["fingerprint"]
        terminal["outputs"] = [identity(unit / "WORKER_RESULT.json")]
        put(value["exit"], terminal)
        result = json.loads(value["result"].read_text())
        result["execution_binding_sha256"] = sha(unit / "EXECUTION_BINDING.json")
        put(value["result"], result)


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_directory()
        self.addCleanup(self.temp.cleanup)
        self.fx = NativeFixture(self.temp.name)
        for patch in self.fx.patches():
            patch.start()
            self.addCleanup(patch.stop)
        self.fit = self.fx.make_unit()

    def verify(self, paths=None, mode="structural"):
        return sr.verify_results(paths or [self.fit["result"]], root=self.fx.root,
                                 output_root=self.fx.out, mode=mode)

    def edit(self, path, mutate):
        value = json.loads(path.read_text())
        mutate(value)
        put(path, value)

    def refuses(self, fragment, paths=None):
        with self.assertRaisesRegex(sr.Refusal, fragment):
            self.verify(paths)

    def test_structural_is_zero_credit_and_never_opens_scientific_outputs(self):
        self.assertFalse(self.fit["fold"].exists())
        with mock.patch.object(sr, "scientific", side_effect=AssertionError("scientific path called")):
            report = self.verify()
        self.assertEqual(report["scientific_fit_count"], 0)
        self.assertEqual(report["units"][0]["scientific_fit_count"], 0)
        self.assertFalse(report["full_matrix_verified"])
        self.assertFalse(report["canonical_adoption"])
        self.assertFalse(report["readout_ready"])
        self.assertEqual(report["status"], "structural_only")

    def test_smoke_structural_remains_zero(self):
        self.assertEqual(self.verify([self.fx.units["smoke"]["result"]])["scientific_fit_count"], 0)

    def test_duplicate_unit_refused(self):
        self.refuses("Duplicate native unit", [self.fit["result"], self.fit["result"]])

    def test_altered_source_refused(self):
        (self.fx.root / "configs/default.yaml").write_text("changed: true\n")
        self.refuses("Bound source changed")

    def test_altered_input_refused(self):
        self.fx.rows.write_bytes(self.fx.rows.read_bytes() + b"\n")
        self.refuses("Bound input changed")

    def test_unbound_worker_mutation_refused(self):
        self.edit(self.fit["unit"] / "WORKER_RESULT.json", lambda x: x.update(ended_utc="2026-10-20T09:03:01Z"))
        self.refuses("E05 file identity mismatch")

    def test_runtime_mismatch_refused(self):
        with mock.patch.object(sr, "runtime_identity", return_value={}):
            self.refuses("Runtime identity changed")

    def test_exact_frozen_row_required_even_with_rehashed_envelope(self):
        self.edit(self.fit["unit"] / "EXECUTION_BINDING.json", lambda x: x["row"].update(ground_weight="0.123"))
        self.fx.rebind(self.fit)
        self.refuses("exact frozen matrix")

    def test_literal_command_required(self):
        self.edit(self.fit["unit"] / "EXECUTION_BINDING.json", lambda x: x["trainer_argv"].append("--resume"))
        self.fx.rebind(self.fit)
        self.refuses("Literal trainer")

    def test_environment_required(self):
        self.edit(self.fit["unit"] / "EXECUTION_BINDING.json", lambda x: x["environment"].update(OMP_NUM_THREADS="1"))
        self.fx.rebind(self.fit)
        self.refuses("Frozen execution environment")

    def test_failed_clean_exit_claim_refused(self):
        self.edit(self.fit["exit"], lambda x: x.update(stop_reason="fixed_deadline"))
        self.refuses("E05 did not finish cleanly")

    def test_missing_clean_exit_reason_refused(self):
        self.edit(self.fit["exit"], lambda x: x.update(stop_reason=None))
        self.refuses("E05 did not finish cleanly")

    def test_process_identity_required(self):
        self.edit(self.fit["exit"].with_name("PROCESS.json"), lambda x: x.update(identity=""))
        self.refuses("Owned process identity")

    def test_process_group_mismatch_refused(self):
        self.edit(self.fit["exit"].with_name("PROCESS.json"), lambda x: x.update(process_group=10000))
        self.refuses("Owned process identity")

    def test_late_launch_refused(self):
        self.edit(self.fit["exit"].with_name("PROCESS.json"),
                  lambda x: x.update(launched_utc="2026-10-20T12:10:00Z"))
        self.refuses("chronology")

    def test_worker_ending_after_exit_refused(self):
        self.edit(self.fit["unit"] / "WORKER_RESULT.json", lambda x: x.update(ended_utc="2026-10-20T09:11:00Z"))
        self.fx.rebind(self.fit)
        self.refuses("chronology")

    def test_boolean_exit_refused(self):
        self.edit(self.fit["exit"], lambda x: x.update(exit_code=False))
        self.refuses("E05 did not finish cleanly")

    def test_extra_input_not_ignored(self):
        self.edit(self.fit["unit"] / "EXECUTION_BINDING.json",
                  lambda x: x["input_sha256"].update({str(self.fx.npz): sha(self.fx.npz)}))
        self.fx.rebind(self.fit)
        self.refuses("Incomplete/extra input")

    def test_smoke_dependency_hash_checked(self):
        self.edit(self.fit["unit"] / "EXECUTION_BINDING.json",
                  lambda x: x["smoke_dependency"].update({str(self.fx.units["smoke"]["result"]): "0" * 64}))
        self.fx.rebind(self.fit)
        self.refuses("Smoke dependency binding")

    def test_smoke_entire_process_chain_checked(self):
        self.edit(self.fx.units["smoke"]["exit"].with_name("PROCESS.json"), lambda x: x.update(identity=""))
        self.refuses("Owned process identity")

    def test_fit_attempt_cannot_start_before_smoke_completion(self):
        self.edit(self.fit["exit"].with_name("START.json"),
                  lambda x: x.update(started_utc="2026-10-20T09:04:00Z"))
        self.refuses("Fit attempt started before its bound smoke completed")

    def test_fit_attempt_can_start_at_smoke_completion_after_earlier_preparation(self):
        self.edit(self.fit["exit"].with_name("START.json"),
                  lambda x: x.update(started_utc="2026-10-20T09:05:00Z"))
        self.assertLess(self.fit["binding"]["created_utc"], "2026-10-20T09:05:00Z")
        self.assertEqual(self.verify()["scientific_fit_count"], 0)

    def test_symlink_metadata_refused(self):
        path = self.fit["unit"] / "WORKER_RESULT.json"
        replacement = path.with_name("real.json")
        path.rename(replacement)
        path.symlink_to(replacement)
        with self.assertRaises(sr.Refusal):
            self.verify()

    def test_duplicate_json_keys_refused(self):
        self.fit["result"].write_text('{"schema":"one","schema":"two"}')
        self.refuses("Duplicate JSON key")

    def test_report_is_new_only_and_bounded(self):
        report = self.verify()
        destination = self.fx.out / "consumption/report.json"
        sr.save_report(destination, report, self.fx.out)
        with self.assertRaises(FileExistsError):
            sr.save_report(destination, report, self.fx.out)
        with self.assertRaises(sr.Refusal):
            sr.save_report(self.fx.out / "escaped.json", report, self.fx.out)

    def fake_acceptance(self, value, *, accept_hook=None, identity_hook=None):
        verdict = copy.deepcopy(value["worker"]["acceptance"])
        fi = copy.deepcopy(value["worker"]["fit_identity"])
        verdict["checks"][0]["detail"] = "fresh volatile timestamp"
        accept = mock.Mock(side_effect=accept_hook, return_value=verdict)
        fit_identity = mock.Mock(side_effect=identity_hook, return_value=fi)
        return types.SimpleNamespace(accept=accept, fit_identity=fit_identity, canon_sha=sr.digest)

    def science_patch(self, helper):
        fixture = self.fx
        return mock.patch.object(sr.Evidence, "module", lambda evidence, name:
                                 helper if name == "accept_fit" else fixture.load_module(evidence, name))

    def test_full_calls_current_unchanged_data_and_acceptance_contract(self):
        self.fx.materialize_outputs(self.fit)
        helper = self.fake_acceptance(self.fit)
        with self.science_patch(helper):
            report = self.verify(mode="full")
        self.assertEqual(report["scientific_fit_count"], 1)
        self.assertFalse(report["full_matrix_verified"])
        self.assertEqual(helper.accept.call_count, 1)
        row, fold, expected = helper.accept.call_args.args
        self.assertEqual(row, self.fit["binding"]["row"])
        self.assertEqual(fold, self.fit["fold"])
        self.assertTrue(expected["data"]["npz_rehashed"])
        self.assertEqual(expected["max_epochs"], 40)
        self.assertEqual(expected["encoder_sha256"], self.fx.driver.ENCODER_SHA)
        self.assertEqual(expected, self.fx.driver.expected_for(self.fit["binding"], expected["data"]))
        self.assertEqual(helper.accept.call_args.kwargs, {"trainer_exit": 0})
        self.assertEqual(helper.fit_identity.call_count, 1)

    def test_full_smoke_still_zero_credit(self):
        smoke = self.fx.units["smoke"]
        self.fx.materialize_outputs(smoke)
        helper = self.fake_acceptance(smoke)
        with self.science_patch(helper):
            report = self.verify([smoke["result"]], mode="full")
        self.assertEqual(report["scientific_fit_count"], 0)
        expected = helper.accept.call_args.args[2]
        self.assertTrue(expected["allow_truncation"])
        self.assertEqual(expected["max_epochs"], 2)

    def test_full_failed_fresh_verdict_refused(self):
        self.fx.materialize_outputs(self.fit)
        helper = self.fake_acceptance(self.fit)
        helper.accept.return_value["accepted"] = False
        with self.science_patch(helper), self.assertRaisesRegex(sr.Refusal, "Fresh technical verdict"):
            self.verify(mode="full")

    def test_full_changed_fresh_fit_identity_refused(self):
        self.fx.materialize_outputs(self.fit)
        helper = self.fake_acceptance(self.fit)
        helper.fit_identity.return_value["initial_state_sha256"] = "0" * 64
        with self.science_patch(helper), self.assertRaisesRegex(sr.Refusal, "Fresh scientific fit identity"):
            self.verify(mode="full")

    def test_real_acceptance_helper_rejects_placeholder_training_outputs(self):
        self.fx.materialize_outputs(self.fit)
        with self.assertRaises(json.JSONDecodeError):
            self.verify(mode="full")

    def test_full_changed_nonvolatile_verdict_detail_refused(self):
        self.fx.materialize_outputs(self.fit)
        helper = self.fake_acceptance(self.fit)
        helper.accept.return_value["checks"][1]["detail"] = "changed"
        with self.science_patch(helper), self.assertRaisesRegex(sr.Refusal, "Fresh technical verdict"):
            self.verify(mode="full")

    def test_full_requires_bound_python_for_metric_libraries(self):
        with mock.patch.object(sr.sys, "executable", "/different/python"):
            with self.assertRaisesRegex(sr.Refusal, "bound research Python"):
                self.verify(mode="full")

    def test_full_refuses_mutated_array_without_array_decode(self):
        self.fx.materialize_outputs(self.fit)
        self.fx.npz.write_bytes(b"changed placeholder\n")
        helper = self.fake_acceptance(self.fit)
        with self.science_patch(helper), self.assertRaisesRegex(RuntimeError, "npz bytes differ"):
            self.verify(mode="full")
        helper.accept.assert_not_called()

    def test_full_refuses_checkpoint_symlink_before_helper(self):
        self.fx.materialize_outputs(self.fit)
        target = self.fit["fold"] / "best.pt"
        target.unlink()
        target.symlink_to(self.fx.npz)
        helper = self.fake_acceptance(self.fit)
        with self.science_patch(helper), self.assertRaisesRegex(sr.Refusal, "Path boundary"):
            self.verify(mode="full")
        helper.accept.assert_not_called()

    def test_full_refuses_mutation_during_helper(self):
        self.fx.materialize_outputs(self.fit)
        verdict = copy.deepcopy(self.fit["worker"]["acceptance"])
        def mutate(*_args, **_kwargs):
            target = self.fit["fold"] / "best.pt"
            target.write_bytes(b"x" * target.stat().st_size)
            return verdict
        helper = self.fake_acceptance(self.fit, accept_hook=mutate)
        with self.science_patch(helper), self.assertRaisesRegex(sr.Refusal, "Scientific file changed"):
            self.verify(mode="full")

    def test_runtime_rechecked_after_full_verification(self):
        self.fx.materialize_outputs(self.fit)
        helper = self.fake_acceptance(self.fit)
        # Structural fit + its smoke dependency, then full verification.
        with self.science_patch(helper), mock.patch.object(
                sr, "runtime_identity", side_effect=[self.fx.runtime, self.fx.runtime, {}]):
            with self.assertRaisesRegex(sr.Refusal, "Runtime changed during"):
                self.verify(mode="full")

    def test_metadata_changed_during_full_refused(self):
        self.fx.materialize_outputs(self.fit)
        verdict = copy.deepcopy(self.fit["worker"]["acceptance"])
        def mutate(*_args, **_kwargs):
            self.fx.rows.write_bytes(self.fx.rows.read_bytes() + b"\n")
            return verdict
        helper = self.fake_acceptance(self.fit, accept_hook=mutate)
        with self.science_patch(helper), self.assertRaisesRegex(sr.Refusal, "Evidence changed"):
            self.verify(mode="full")

    def test_duplicate_checkpoint_between_different_units_refused(self):
        second = self.fx.make_unit(order=2)
        worker_path = second["unit"] / "WORKER_RESULT.json"
        def mutate(worker):
            item = self.fit["worker"]["acceptance"]["identities"]["best_pt"]
            worker["acceptance"]["identities"]["best_pt"] = item
            worker["fit_identity"]["best_pt_sha256"] = item["sha256"]
            worker["fit_identity_sha256"] = sr.digest(worker["fit_identity"])
        self.edit(worker_path, mutate)
        self.fx.rebind(second)
        self.refuses("Duplicate native unit evidence: best_pt", [self.fit["result"], second["result"]])

    def test_initialization_must_match_within_seed(self):
        second = self.fx.make_unit(order=2)
        self.assertEqual(second["binding"]["row"]["init_seed"], self.fit["binding"]["row"]["init_seed"])
        def mutate(worker):
            worker["fit_identity"]["initial_state_sha256"] = "0" * 64
            worker["fit_identity_sha256"] = sr.digest(worker["fit_identity"])
        self.edit(second["unit"] / "WORKER_RESULT.json", mutate)
        self.fx.rebind(second)
        self.refuses("Initialization differs within a seed", [self.fit["result"], second["result"]])

    def test_different_seeds_cannot_share_initialization(self):
        second = self.fx.make_unit(order=26)
        def mutate(worker):
            worker["fit_identity"]["initial_state_sha256"] = self.fit["worker"]["fit_identity"]["initial_state_sha256"]
            worker["fit_identity_sha256"] = sr.digest(worker["fit_identity"])
        self.edit(second["unit"] / "WORKER_RESULT.json", mutate)
        self.fx.rebind(second)
        self.refuses("Different seeds share initialization", [self.fit["result"], second["result"]])

    def test_unreviewed_producer_pin_refused(self):
        with mock.patch.dict(sr.PINS, driver="0" * 64):
            self.refuses("Unreviewed consumer dependency")

    def test_signed_off_claim_without_fit_identity_refused(self):
        self.edit(self.fit["unit"] / "WORKER_RESULT.json", lambda x: x.pop("fit_identity"))
        self.fx.rebind(self.fit)
        self.refuses("Fit identity digest")

    def test_boolean_producer_fit_count_refused(self):
        self.edit(self.fit["result"], lambda x: x.update(scientific_fit_count=True))
        self.refuses("Producer fit count")

    def test_metadata_directory_symlink_refused(self):
        source = self.fx.root / "configs"
        replacement = self.fx.root / "moved_configs"
        source.rename(replacement)
        source.symlink_to(replacement, target_is_directory=True)
        self.refuses("Noncanonical metadata path")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_directory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "metadata.json"

    def test_size_bound(self):
        put(self.path, b"12345")
        with mock.patch.object(sr, "METADATA_LIMIT", 4), self.assertRaisesRegex(sr.Refusal, "size bound"):
            sr.Evidence().read(self.path)

    def test_same_size_metadata_change_detected(self):
        put(self.path, {"a": 1})
        evidence = sr.Evidence()
        evidence.read(self.path)
        put(self.path, {"a": 2})
        with self.assertRaisesRegex(sr.Refusal, "Evidence changed"):
            evidence.unchanged()

    def test_scientific_inode_replacement_detected(self):
        put(self.path, b"large placeholder")
        evidence = sr.Evidence()
        evidence.watch_large(self.path, self.path.parent)
        replacement = put(self.path.with_name("replacement"), self.path.read_bytes())
        replacement.replace(self.path)
        with self.assertRaisesRegex(sr.Refusal, "Scientific file changed"):
            evidence.unchanged()

    def test_normal_real_e05_wrapper_receipts_match_native_contract(self):
        """Real wrapper + a tiny substitute worker, never a trainer invocation."""
        base = Path(self.temp.name)
        unit, attempts = base / "unit", base / "attempts"
        unit.mkdir()
        attempts.mkdir()
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = (now + dt.timedelta(minutes=1)).isoformat()
        deadline = (now + dt.timedelta(minutes=2)).isoformat()
        session = put(unit / "SESSION.json", {"admission_cutoff_utc": cutoff, "deadline_utc": deadline})
        binding_path = put(unit / "EXECUTION_BINDING.json", {"synthetic": True})
        worker_path = unit / "WORKER_RESULT.json"
        stub = base / "tiny_worker.py"
        stub.write_text("import datetime,json,pathlib,time\n"
                        "time.sleep(0.2)\n"
                        "pathlib.Path(" + repr(str(worker_path)) + ").write_text(json.dumps("
                        "{'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))\n")
        command = [sys.executable, str(stub), "_worker", "--binding", str(binding_path)]
        argv = [sys.executable, str(sr.FILES["wrapper"]), "--session", str(session),
                "--attempt-root", str(attempts), "--task-id", "M03", "--job-key", "synthetic",
                "--cwd", str(base), "--source", str(stub), "--input", str(binding_path),
                "--output", str(worker_path), "--", *command]
        result = subprocess.run(argv, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exit_path = next(attempts.glob("*/EXIT.json"))
        terminal = json.loads(exit_path.read_text())
        self.assertEqual(terminal["stop_reason"], "process_exit")
        binding = {"kind": "smoke", "row": {"fit_id": "synthetic"}, "python": sys.executable,
                   "root": str(base), "source_sha256": {str(stub): sha(stub)},
                   "created_utc": (now - dt.timedelta(seconds=1)).isoformat()}
        worker = json.loads(worker_path.read_text())
        document = {"e05_exit_receipt": str(exit_path), "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        driver = types.SimpleNamespace(FIXED_CUTOFF=cutoff, FIXED_DEADLINE=deadline)
        with mock.patch.dict(sr.FILES, driver=stub):
            actual = sr._native_e05(binding, worker, document, unit, base, sr.Evidence(), driver)
        self.assertEqual(actual, exit_path)


if __name__ == "__main__":
    unittest.main()
