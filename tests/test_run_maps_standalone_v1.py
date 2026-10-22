"""Synthetic native maps boundaries; never import or run the map/model program."""
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

import test_readout_standalone_v1 as fixture

SOURCE = fixture.SUPPORT / "run_maps_standalone_v1.py"


class MapsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ReadoutTests.setUpClass()
        cls.rd = fixture.ReadoutTests.rd
        cls.maps = fixture.load(SOURCE, "maps_under_test") if SOURCE.exists() else None

    def setUp(self):
        self.assertIsNotNone(self.maps, "Native maps component is not implemented")
        fixture.ReadoutTests.setUp(self)
        self.run_id = self.rows[0]["run_id"]
        self.cell = self.rows[0]["cell"]
        self.selected = [r for r in self.rows if r["run_id"] == self.run_id]
        data = {k: fixture.digest(k) for k in self.native.DATA_KEYS}
        data.update(dataset_version="M1", n_entries=5, acceptance={"decision": "accepted"})
        self.common = {
            "root": str(self.root), "output_root": str(self.out),
            "support_dir": str(fixture.SUPPORT), "python": str(self.root / ".venv/bin/python"),
            "runtime": {"synthetic": True}, "source_sha256": {},
            "environment": {"OMP_NUM_THREADS": "6", "TMPDIR": str(self.out / "tmp")},
            "encoder_sha256": "e" * 64, "scientific_acceptance": {"threads": 6},
            **{k: str(self.root / k) for k in ("rows", "manifest", "splits", "data_acceptance")},
        }
        for bundle in self.bundles.values():
            bundle["binding"].update(self.common)
            bundle["binding"]["resource_policy"] = {
                "internal_reserve_bytes": 1000000000, "external_reserve_bytes": 2000000000,
                "checkpoint_forecast_bytes": 1100000000, "ac_required": True}
            bundle["worker"]["data"] = data
            for name, document in (("EXECUTION_BINDING.json", bundle["binding"]),
                                   ("WORKER_RESULT.json", bundle["worker"])):
                item = fixture.put(bundle["unit"] / name, document)
                bundle["evidence"].files[item["path"]] = item
        self.native_loader = mock.patch.object(self.maps, "load_native", return_value=self.native)
        self.native_loader.start()
        self.addCleanup(self.native_loader.stop)

    def context(self, full=True):
        return self.maps.five_context(self.root, self.out, self.cell, self.run_id, "best", full=full)

    def test_full_mode_forwards_exact_five_and_explicit_native_identity(self):
        context = self.context()
        self.assertEqual(len(context["checkpoints"]), 5)
        self.assertEqual(self.native.scientific.call_count, 5)
        self.assertEqual([c["fold"] for c in context["checkpoints"]], list(range(5)))
        self.assertTrue(all("native_bundle_sha256" in c for c in context["checkpoints"]))
        self.assertTrue(all("fit_receipt_sha256" not in c for c in context["checkpoints"]))

    def test_missing_fold_refused(self):
        (self.out / "units" / self.selected[0]["fit_id"] / "RESULT.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing|Missing"):
            self.context()

    def test_wrong_cell_refused(self):
        with self.assertRaisesRegex(ValueError, "cell|Cell"):
            self.maps.five_context(self.root, self.out, "E", self.run_id, "best")

    def test_structural_report_cannot_satisfy_full_request(self):
        real = self.native.verify_results
        with mock.patch.object(self.native, "verify_results",
                               side_effect=lambda paths, **kw: real(paths, **(kw | {"mode": "structural"}))):
            with self.assertRaisesRegex(ValueError, "full|Full"):
                self.context()

    def test_common_runtime_substitution_refused(self):
        bundle = next(iter(self.bundles.values()))
        bundle["binding"]["runtime"] = {"changed": True}
        item = fixture.put(bundle["unit"] / "EXECUTION_BINDING.json", bundle["binding"])
        bundle["evidence"].files[item["path"]] = item
        with self.assertRaisesRegex(ValueError, "common|Common"):
            self.context()

    def test_data_substitution_refused(self):
        bundle = next(iter(self.bundles.values()))
        bundle["worker"]["data"] = bundle["worker"]["data"] | {"rows_sha256": "0" * 64}
        item = fixture.put(bundle["unit"] / "WORKER_RESULT.json", bundle["worker"])
        bundle["evidence"].files[item["path"]] = item
        with self.assertRaisesRegex(ValueError, "data|Data"):
            self.context()

    def test_checkpoint_path_substitution_refused(self):
        bundle = next(iter(self.bundles.values()))
        bundle["binding"]["fold_dir"] += "-changed"
        item = fixture.put(bundle["unit"] / "EXECUTION_BINDING.json", bundle["binding"])
        bundle["evidence"].files[item["path"]] = item
        with self.assertRaisesRegex(ValueError, "fold|checkpoint"):
            self.context()

    def test_frozen_seed_or_fold_substitution_refused(self):
        bundle = next(iter(self.bundles.values()))
        original = copy.deepcopy(bundle["binding"])
        for key, changed in (("init_seed", "7"), ("fold", "4")):
            bundle["binding"] = copy.deepcopy(original)
            bundle["binding"]["row"][key] = changed
            item = fixture.put(bundle["unit"] / "EXECUTION_BINDING.json", bundle["binding"])
            bundle["evidence"].files[item["path"]] = item
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "row|units|Duplicate|initialization"):
                self.context()

    def test_stale_native_inputs_refused(self):
        bundle = next(iter(self.bundles.values()))
        fixture.put(bundle["unit"] / "WORKER_RESULT.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "identity|changed"):
            self.context()

    def test_stale_target_refused(self):
        self.out.joinpath("maps").mkdir()
        with self.assertRaisesRegex(ValueError, "exist|stale"):
            self.maps.new_target(self.out, self.out / "maps")

    def test_literal_command_has_exact_bound_arguments(self):
        context = self.context()
        command = self.maps.map_command(context, self.out / "maps/new")
        self.assertEqual(command[0], self.common["python"])
        self.assertEqual(command[command.index("--rows") + 1], self.common["rows"])
        self.assertEqual(command[command.index("--splits") + 1], self.common["splits"])
        self.assertEqual(command[command.index("--which") + 1], "best")
        self.assertNotIn("--maps-cmd", command)
        self.assertEqual(command[1], str(self.root / self.maps.MAP_REL))

    def map_fixture(self, which="best"):
        context = self.context()
        context["which"] = which
        if which == "last":
            for checkpoint in context["checkpoints"]:
                checkpoint["path"] = str(Path(checkpoint["path"]).with_name("last.pt"))
        directory = self.out / "maps" / "synthetic"
        directory.mkdir(parents=True)
        membership = {f"synthetic-{fold}": fold for fold in range(5)}
        sources = {str(self.root / name): fixture.digest(name) for name in self.maps.CODE_FILES}
        data = {k: context["data"][k] for k in self.maps.DATA_KEYS}
        patients = {pid: {"fold": fold, "p": .5, "union": {}} for pid, fold in membership.items()}
        patient_set = {"n": 5, "n_spatial": 5, "per_fold_n": {str(f): 1 for f in range(5)},
                       "patient_ids_sha256": fixture.digest(sorted(patients))}
        definition = {**self.maps.MAP_DEFINITION, "readout_map": self.maps.READOUT_MAP[self.cell],
                      "code_sha256": {name: sources[str(self.root / name)] for name in self.maps.CODE_FILES}
                                    | {"maps_script": self.maps.MAP_SHA}}
        doc = {"schema": "n08.maps.v2", "cell": self.cell, "run": self.run_id, "which": which,
               "readout_map": self.maps.READOUT_MAP[self.cell],
               "status": "control_passed", "data": data, "map_definition": definition,
               "checkpoints": [{k: c[k] for k in ("fold", "path", "bytes", "sha256")}
                               for c in context["checkpoints"]],
               "control": {"passed": True}, "control_passed": True, "per_patient": patients,
               "patient_set": patient_set, "folds": {r["fold"]: {"n": 1, "region_pooling": r["region_pooling"]}
                                                   for r in context["rows"]}}
        return context, directory, membership, sources, doc

    def test_native_output_full_context_and_membership(self):
        context, _, membership, sources, doc = self.map_fixture()
        self.assertTrue(self.maps.check_native_output(doc, context, sources, membership))

    def test_native_output_rejects_patient_fold_counts_definition_and_checkpoint_mutations(self):
        context, _, membership, sources, doc = self.map_fixture()
        changes = [
            lambda d: d["per_patient"].pop("synthetic-0"),
            lambda d: d["per_patient"]["synthetic-0"].update(fold=1),
            lambda d: d["patient_set"].update(n_spatial=0),
            lambda d: d["map_definition"].update(readout_map="query0"),
            lambda d: d["checkpoints"][0].update(bytes=999),
            lambda d: d["checkpoints"].append(copy.deepcopy(d["checkpoints"][0])),
            lambda d: d["control"].update(passed=False),
        ]
        for change in changes:
            with self.subTest(change=change):
                altered = copy.deepcopy(doc)
                change(altered)
                with self.assertRaises(ValueError):
                    self.maps.check_native_output(altered, context, sources, membership)

    def test_last_probability_must_be_finite(self):
        context, _, membership, sources, doc = self.map_fixture(which="last")
        doc["per_patient"]["synthetic-0"]["p"] = float("nan")
        with self.assertRaisesRegex(ValueError, "probability"):
            self.maps.check_native_output(doc, context, sources, membership)

    def test_expected_membership_uses_bound_summary_and_frozen_fold(self):
        context = self.context()
        folds = []
        for checkpoint in context["checkpoints"]:
            fold = checkpoint["fold"]
            pid = "synthetic-" + str(fold)
            folds.append({"outer_test": {"patient_ids": [pid]}})
            item = fixture.put(Path(checkpoint["path"]).with_name("fold_summary.json"),
                               {"summary": {"oof": [{"patient_id": pid}]}})
            checkpoint["fold_summary"] = {k: item[k] for k in ("bytes", "sha256")}
        fixture.put(Path(context["common"]["splits"]), {"folds": folds})
        expected = self.maps.expected_membership(context, self.native.Evidence())
        self.assertEqual(expected, {f"synthetic-{f}": f for f in range(5)})
        folds[0]["outer_test"]["patient_ids"] = ["substituted"]
        fixture.put(Path(context["common"]["splits"]), {"folds": folds})
        with self.assertRaisesRegex(ValueError, "OOF membership"):
            self.maps.expected_membership(context, self.native.Evidence())

    def terminal_fixture(self):
        context, directory, membership, sources, doc = self.map_fixture()
        # Tiny source placeholders only; actual map methods are never imported.
        for path, _sha in list(sources.items()):
            item = fixture.put(Path(path), {"synthetic_source": path})
            sources[path] = item["sha256"]
        doc["map_definition"]["code_sha256"].update(
            {name: sources[str(self.root / name)] for name in self.maps.CODE_FILES})
        sources = dict(sorted(sources.items()))
        driver = self.native.Evidence().module("driver")
        binding = {
            "schema": self.maps.BINDING_SCHEMA, "context": context, "maps_dir": str(directory),
            "created_utc": "2026-10-20T10:00:00Z", "source_sha256": sources,
            "argv": self.maps.map_command(context, directory),
            "session": {"admission_cutoff_utc": driver.FIXED_CUTOFF, "deadline_utc": driver.FIXED_DEADLINE},
            "resource_policy": {"internal_reserve_bytes": 1000000000, "external_reserve_bytes": 2000000000,
                                "theia_reserve_bytes": 50000000000, "backing_reserve_bytes": 100000000000,
                                "checkpoint_forecast_bytes": 1100000000, "ac_required": True},
            "volumes": {"internal": {"path": "/"}, "theia": {"path": "/Volumes/" + self.root.parts[2]}},
        }
        binding_path = directory / "MAP_BINDING.json"
        binding_id = fixture.put(binding_path, binding)
        fixture.put(directory / "SESSION.json", binding["session"])
        receipt_path, good, bad, log = self.maps.receipt_paths(binding)
        output_id = fixture.put(good, doc)
        log_id = fixture.put(log, {"synthetic_log": True})
        receipt = {"schema": self.maps.SCHEMA, "status": "accepted", "failed_reasons": [], "exit_code": 0,
                   "authority": "native_e05_pending_terminal_verification",
                   "canonical_adoption": False, "final_readout_complete": False,
                   "binding": binding_id, "context": context, "argv": binding["argv"],
                   "cwd": self.common["root"], "runtime": self.common["runtime"], "source_sha256": sources,
                   "started_utc": "2026-10-20T10:01:01Z", "ended_utc": "2026-10-20T10:02:00Z",
                   "output": output_id, "log": log_id,
                   **{k: doc[k] for k in ("control", "patient_set", "map_definition")}}
        receipt_id = fixture.put(receipt_path, receipt)
        attempt = self.out / "map_attempts" / ("M06_" + self.maps.job_key(context) + "_synthetic")
        command = self.maps.worker_command(binding, binding_path)
        evidence = self.native.Evidence()
        launch = {"task_id": "M06", "job_key": self.maps.job_key(context), "command": command,
                  "cwd": self.common["root"], "sources": [self.maps.evidence_item(evidence, Path(p)) for p in sources],
                  "inputs": [binding_id]}
        start = {"schema": "theia.cpu_attempt.v1", "task_id": "M06", "job_key": self.maps.job_key(context),
                 "resume_receipt": None, "binding": launch, "fingerprint": fixture.digest(launch),
                 "runner": self.maps.evidence_item(evidence, self.native.FILES["wrapper"]),
                 "session_sha256": evidence.sha(directory / "SESSION.json"),
                 "cutoff_utc": driver.FIXED_CUTOFF, "deadline_utc": driver.FIXED_DEADLINE,
                 "started_utc": "2026-10-20T10:01:00Z"}
        fixture.put(attempt / "START.json", start)
        fixture.put(attempt / "PROCESS.json", {"command": command, "pid": 123, "process_group": 123,
                    "identity": "synthetic process", "launched_utc": "2026-10-20T10:01:00Z"})
        terminal = {"schema": "theia.cpu_attempt.v1", "task_id": "M06", "job_key": self.maps.job_key(context),
                    "fingerprint": start["fingerprint"], "status": "finished", "exit_code": 0, "signals": [],
                    "runner_error": None, "output_errors": [], "stop_reason": "process_exit", "outputs": [receipt_id],
                    "ended_utc": "2026-10-20T10:02:01Z"}
        exit_path = attempt / "EXIT.json"
        fixture.put(exit_path, terminal)
        fixture.put(directory / "EXECUTION_RESULT.json", {"schema": "theia.corrected_cpu.map_execution_result.v1",
                    "wrapper_exit": 0, "resource_stops": [],
                    "e05_exit": str(exit_path), "binding": binding_id, "ended_utc": "2026-10-20T10:02:02Z"})
        patches = [
            mock.patch.object(self.native, "runtime_identity", return_value=self.common["runtime"]),
            mock.patch.object(self.maps, "map_sources", return_value=sources),
            mock.patch.object(self.maps, "five_context", return_value=context),
            mock.patch.object(self.maps, "expected_membership", return_value=membership),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return receipt_path, exit_path, receipt, terminal, good, bad, sources

    def test_native_terminal_receipt_verifies_with_explicit_exit(self):
        receipt, terminal, *_ = self.terminal_fixture()
        result = self.maps.verify_map_receipt(receipt, terminal)
        self.assertEqual(result["status"], "verified_current")
        self.assertFalse(result["final_readout_complete"])
        self.maps.five_context.assert_called_once()

    def test_terminal_nonzero_signalled_controlfailed_and_output_mutation_refused(self):
        receipt, exit_path, _, terminal, good, bad, _ = self.terminal_fixture()
        for change in ({"exit_code": 1}, {"exit_code": -2}, {"signals": [{"signal": "SIGINT"}]},
                       {"status": "incomplete_or_failed"}, {"stop_reason": "parent_signal"}):
            fixture.put(exit_path, terminal | change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "E05 failed"):
                self.maps.verify_map_receipt(receipt, exit_path)
        fixture.put(exit_path, terminal)
        fixture.put(bad, {"control_failed": True})
        with self.assertRaisesRegex(ValueError, "control_failed"):
            self.maps.verify_map_receipt(receipt, exit_path)
        bad.unlink()
        fixture.put(good, {"mutated": True})
        with self.assertRaisesRegex(ValueError, "output/log identity"):
            self.maps.verify_map_receipt(receipt, exit_path)

    def test_receipt_mutation_and_context_substitution_refused(self):
        receipt_path, exit_path, receipt, terminal, *_ = self.terminal_fixture()
        receipt["context"] = receipt["context"] | {"run_id": "substituted"}
        changed_id = fixture.put(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "output hash"):
            self.maps.verify_map_receipt(receipt_path, exit_path)
        fixture.put(exit_path, terminal | {"outputs": [changed_id]})
        with self.assertRaisesRegex(ValueError, "context equality"):
            self.maps.verify_map_receipt(receipt_path, exit_path)

    def test_e05_source_and_command_mutations_refused(self):
        receipt, exit_path, _, _, _, _, sources = self.terminal_fixture()
        start_path = exit_path.with_name("START.json")
        start = json.loads(start_path.read_text())
        changed = copy.deepcopy(start)
        changed["binding"]["command"][-1] = "/different/binding.json"
        changed["fingerprint"] = fixture.digest(changed["binding"])
        fixture.put(start_path, changed)
        with self.assertRaisesRegex(ValueError, "command"):
            self.maps.verify_map_receipt(receipt, exit_path)
        fixture.put(start_path, start)
        fixture.put(Path(next(iter(sources))), {"source": "changed"})
        with self.assertRaisesRegex(ValueError, "sources"):
            self.maps.verify_map_receipt(receipt, exit_path)

    def test_resource_failure_in_coordinator_cannot_be_accepted(self):
        receipt, exit_path, *_ = self.terminal_fixture()
        result_path = receipt.parent / "EXECUTION_RESULT.json"
        result = json.loads(result_path.read_text())
        fixture.put(result_path, result | {"resource_stops": ["capacity unavailable"]})
        with self.assertRaisesRegex(ValueError, "resource result"):
            self.maps.verify_map_receipt(receipt, exit_path)

    def test_real_e05_tiny_child_terminal_contract(self):
        receipt, _, _, _, _, _, sources = self.terminal_fixture()
        target = receipt.parent
        binding_path = target / "MAP_BINDING.json"
        binding = json.loads(binding_path.read_text())
        # A short current fixture window keeps this oracle repeatable after the
        # production session closes. Production fixed-window checks are separate.
        driver = self.native.Evidence().module("driver")
        now = datetime.datetime.now(datetime.timezone.utc)
        driver.FIXED_CUTOFF = (now + datetime.timedelta(seconds=30)).isoformat()
        driver.FIXED_DEADLINE = (now + datetime.timedelta(seconds=90)).isoformat()
        binding["session"] = {"admission_cutoff_utc": driver.FIXED_CUTOFF,
                              "deadline_utc": driver.FIXED_DEADLINE}
        binding["created_utc"] = now.isoformat()
        binding_id = fixture.put(binding_path, binding)
        fixture.put(target / "SESSION.json", binding["session"])
        receipt_doc = json.loads(receipt.read_text())
        fixture.put(receipt, receipt_doc | {"binding": binding_id})
        # The only child is a tiny JSON timestamp writer. This command substitution
        # is a test mock, never a production option or a maps/model invocation.
        stub = [sys.executable, "-c",
                "import datetime,json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
                "r=json.loads(p.read_text()); t=datetime.datetime.now(datetime.timezone.utc).isoformat(); "
                "r.update(started_utc=t,ended_utc=t); p.write_text(json.dumps(r))", str(receipt)]
        attempts = self.out / "map_attempts"
        command = [sys.executable, str(self.native.FILES["wrapper"]),
                   "--session", str(target / "SESSION.json"), "--attempt-root", str(attempts),
                   "--task-id", "M06", "--job-key", self.maps.job_key(binding["context"]),
                   "--cwd", str(self.root), "--stop-grace-seconds", "60"]
        for source in sources:
            command += ["--source", source]
        command += ["--input", str(binding_path), "--output", str(receipt), "--", *stub]
        before = set(attempts.glob("*/EXIT.json"))
        run = subprocess.run(command, capture_output=True, text=True,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        created = set(attempts.glob("*/EXIT.json")) - before
        self.assertEqual(len(created), 1)
        exit_path = created.pop()
        coordinator = json.loads((target / "EXECUTION_RESULT.json").read_text())
        fixture.put(target / "EXECUTION_RESULT.json", coordinator | {
            "e05_exit": str(exit_path), "binding": binding_id,
            "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        with mock.patch.object(self.maps, "worker_command", return_value=stub), \
                mock.patch.object(self.native.Evidence, "module", return_value=driver):
            verified = self.maps.verify_map_receipt(receipt, exit_path)
        self.assertEqual(verified["status"], "verified_current")

    def test_production_fixed_window_cannot_be_extended(self):
        receipt, exit_path, *_ = self.terminal_fixture()
        path = receipt.parent / "MAP_BINDING.json"
        binding = json.loads(path.read_text())
        binding["session"]["deadline_utc"] = "2026-10-20T13:30:00Z"
        fixture.put(path, binding)
        with self.assertRaisesRegex(ValueError, "Fixed map session"):
            self.maps.verify_map_receipt(receipt, exit_path)

    def test_worker_true_exit_and_control_failed_outputs_preserved(self):
        receipt_path, exit_path, _, _, good, bad, _ = self.terminal_fixture()
        original_doc = json.loads(good.read_text())
        binding_path = receipt_path.parent / "MAP_BINDING.json"
        binding = json.loads(binding_path.read_text())
        _, _, _, log = self.maps.receipt_paths(binding)
        for returncode in (0, 3, -2):
            for path in (receipt_path, good, bad, log):
                path.unlink(missing_ok=True)
            def fake_maps(argv, **kwargs):
                self.assertEqual(argv, binding["argv"])
                self.assertEqual(kwargs["cwd"], self.common["root"])
                kwargs["stdout"].write(b"synthetic map process log\n")
                fixture.put(good if returncode == 0 else bad,
                            original_doc if returncode == 0 else original_doc | {"control_passed": False})
                return types.SimpleNamespace(returncode=returncode)
            with mock.patch.dict(os.environ, {
                    **self.common["environment"], "THEIA_ATTEMPT_DIR": str(exit_path.parent)}), \
                    mock.patch.object(self.maps.os, "getpid", return_value=123), \
                    mock.patch.object(self.maps.os, "getpgrp", return_value=123), \
                    mock.patch.object(self.maps.os, "getcwd", return_value=self.common["root"]), \
                    mock.patch.object(self.maps, "resource", return_value={"synthetic": True}), \
                    mock.patch.object(self.maps.subprocess, "run", side_effect=fake_maps):
                status = self.maps.worker(binding_path)
            receipt = json.loads(receipt_path.read_text())
            with self.subTest(returncode=returncode):
                self.assertEqual(receipt["exit_code"], returncode)
                self.assertEqual(status, int(returncode != 0))
                self.assertEqual(receipt["status"], "accepted" if returncode == 0 else "failed")
                self.assertTrue(Path(receipt["output"]["path"]).is_file())
                self.assertTrue(log.is_file())

    def test_resource_post_probe_safe_stop_is_binding(self):
        times = iter([datetime.datetime(2026, 10, 20, 12, 27, 59, tzinfo=datetime.timezone.utc),
                      datetime.datetime(2026, 10, 20, 12, 28, 1, tzinfo=datetime.timezone.utc)])
        driver = types.SimpleNamespace(
            FIXED_STOP="2026-10-20T12:28:00Z", utc=self.native.utc, now=lambda: next(times),
            resource_check=lambda *_a, **_kw: {"volumes": {
                name: {"free_bytes": 1000000000000} for name in ("theia", "backing")}})
        with self.assertRaisesRegex(ValueError, "during probe"):
            self.maps.resource(self.resource_binding(), driver)

    def test_map_source_pin_refuses_changed_retained_program(self):
        context = self.context()
        (self.root / ".git").mkdir()
        expected = {
            str(self.root / self.maps.MAP_REL): "wrong",
            str(fixture.SUPPORT / "standalone_receipts.py"): self.maps.NATIVE_SHA,
            str(self.root / self.maps.MAP_REL.parent.parent / "data_manifest.py"): self.native.PINS["data_manifest"],
            **{str(self.root / name): sha for name, sha in self.maps.MAP_ONLY.items()},
            str(self.maps.SELF): fixture.digest("self"),
        }
        evidence = types.SimpleNamespace(sha=lambda path: expected[str(path)])
        with self.assertRaisesRegex(ValueError, "Map source changed"):
            self.maps.map_sources(context, evidence, self.native)

    def resource_binding(self):
        return {"resource_policy": {
            "internal_reserve_bytes": 1000000000, "external_reserve_bytes": 2000000000,
            "theia_reserve_bytes": 50000000000, "backing_reserve_bytes": 100000000000,
            "checkpoint_forecast_bytes": 1100000000, "ac_required": True},
            "volumes": {name: {"path": "/" + name, "device": number}
                        for number, name in enumerate(("theia", "backing", "internal"))}}

    def probe_resources(self, free):
        binding = self.resource_binding()
        driver = self.native.Evidence().module("driver")
        now = datetime.datetime(2026, 10, 20, 11, tzinfo=datetime.timezone.utc)
        devices = {v["path"]: v["device"] for v in binding["volumes"].values()}
        with mock.patch.object(driver, "now", return_value=now), \
                mock.patch.object(driver.Path, "stat",
                                  lambda path: types.SimpleNamespace(st_dev=devices[str(path)])), \
                mock.patch.object(driver.shutil, "disk_usage",
                                  side_effect=lambda path: types.SimpleNamespace(free=free[Path(path).name])), \
                mock.patch.object(driver.subprocess, "check_output", return_value="Now drawing from 'AC Power'"):
            return self.maps.resource(binding, driver, admission=True)

    def test_theia_reserve_rejected_independently(self):
        with self.assertRaisesRegex(ValueError, "theia"):
            self.probe_resources({"theia": 49999999999, "backing": 200000000000, "internal": 4000000000})

    def test_research_reserve_rejected_independently(self):
        with self.assertRaisesRegex(ValueError, "backing"):
            self.probe_resources({"theia": 200000000000, "backing": 99999999999, "internal": 4000000000})

    def test_distinct_reserves_keep_forecast_at_exact_boundary(self):
        observed = self.probe_resources({"theia": 51100000000, "backing": 101100000000, "internal": 1000000000})
        self.assertEqual(observed["volumes"]["theia"]["required_bytes"], 51100000000)
        self.assertEqual(observed["volumes"]["backing"]["required_bytes"], 101100000000)

    def test_lower_inheritance_cannot_reduce_current_floors(self):
        policy = self.maps.minimum_resource_policy([{
            "internal_reserve_bytes": 500000000, "external_reserve_bytes": 2000000000,
            "checkpoint_forecast_bytes": 1}])
        self.assertEqual(policy["theia_reserve_bytes"], 50000000000)
        self.assertEqual(policy["backing_reserve_bytes"], 100000000000)
        self.assertEqual(policy["internal_reserve_bytes"], 1000000000)
        self.assertEqual(policy["checkpoint_forecast_bytes"], 1100000000)

    def test_higher_reserve_from_any_bound_fold_is_retained(self):
        selected = [b for b in self.bundles.values() if b["binding"]["row"]["run_id"] == self.run_id]
        last = selected[-1]
        last["binding"]["resource_policy"].update(external_reserve_bytes=150000000000,
                                                  backing_reserve_bytes=200000000000,
                                                  checkpoint_forecast_bytes=2000000000)
        item = fixture.put(last["unit"] / "EXECUTION_BINDING.json", last["binding"])
        last["evidence"].files[item["path"]] = item
        policy = self.maps.bound_resource_policy(self.context(), self.native.Evidence())
        self.assertEqual(policy["theia_reserve_bytes"], 150000000000)
        self.assertEqual(policy["backing_reserve_bytes"], 200000000000)
        self.assertEqual(policy["checkpoint_forecast_bytes"], 2000000000)

    def test_bound_distinct_floor_cannot_be_lowered(self):
        receipt, exit_path, *_ = self.terminal_fixture()
        path = receipt.parent / "MAP_BINDING.json"
        binding = json.loads(path.read_text())
        binding["resource_policy"]["backing_reserve_bytes"] = 50000000000
        fixture.put(path, binding)
        with self.assertRaisesRegex(ValueError, "resource floors"):
            self.maps.verify_map_receipt(receipt, exit_path)

    def test_worker_setup_crossing_time_boundary_never_spawns(self):
        receipt_path, exit_path, _, _, good, bad, _ = self.terminal_fixture()
        binding_path = receipt_path.parent / "MAP_BINDING.json"
        binding = json.loads(binding_path.read_text())
        _, _, _, log = self.maps.receipt_paths(binding)
        original_doc = json.loads(good.read_text())
        driver = self.native.Evidence().module("driver")
        current = [datetime.datetime(2026, 10, 20, 12, 9, 59, tzinfo=datetime.timezone.utc)]
        healthy = {"volumes": {name: {"free_bytes": 1000000000000, "required_bytes": 0}
                               for name in ("theia", "backing", "internal")}}
        def resources(_binding, admission=False):
            if admission and current[0] >= driver.utc(driver.FIXED_CUTOFF):
                raise ValueError("Fixed admission cutoff reached")
            return copy.deepcopy(healthy)
        original_unchanged = self.native.Evidence.unchanged
        original_open = Path.open
        for stage in ("metadata", "log"):
            for crossing in ("2026-10-20T12:10:01Z", "2026-10-20T12:28:01Z"):
                for path in (receipt_path, good, bad, log):
                    path.unlink(missing_ok=True)
                current[0] = driver.utc("2026-10-20T12:09:59Z")
                def unchanged(evidence):
                    original_unchanged(evidence)
                    if stage == "metadata":
                        current[0] = driver.utc(crossing)
                def opened(path, *args, **kwargs):
                    stream = original_open(path, *args, **kwargs)
                    if path == log and args == ("xb",) and stage == "log":
                        current[0] = driver.utc(crossing)
                    return stream
                def maps_stub(*_args, **_kwargs):
                    fixture.put(good, original_doc)
                    return types.SimpleNamespace(returncode=0)
                with mock.patch.dict(os.environ, {
                        **self.common["environment"], "THEIA_ATTEMPT_DIR": str(exit_path.parent)}), \
                        mock.patch.object(self.maps.os, "getpid", return_value=123), \
                        mock.patch.object(self.maps.os, "getpgrp", return_value=123), \
                        mock.patch.object(self.maps.os, "getcwd", return_value=self.common["root"]), \
                        mock.patch.object(driver, "now", side_effect=lambda: current[0]), \
                        mock.patch.object(driver, "resource_check", side_effect=resources), \
                        mock.patch.object(self.native.Evidence, "module", return_value=driver), \
                        mock.patch.object(self.native.Evidence, "unchanged", unchanged), \
                        mock.patch.object(Path, "open", opened), \
                        mock.patch.object(self.maps.subprocess, "run", side_effect=maps_stub) as launch:
                    status = self.maps.worker(binding_path)
                with self.subTest(stage=stage, crossing=crossing):
                    self.assertEqual(launch.call_count, 0, "Map subprocess started after the time boundary")
                    receipt = json.loads(receipt_path.read_text())
                    self.assertEqual(status, 1)
                    self.assertEqual(receipt["status"], "failed")
                    self.assertIsNone(receipt["exit_code"])
                    self.assertTrue(log.exists())


if __name__ == "__main__":
    unittest.main()
