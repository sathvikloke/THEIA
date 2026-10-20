"""Verify native corrected-CPU receipts without creating legacy controller records.

Structural mode never opens NPZ/checkpoint/training-output files and always grants
zero fit credit. Full mode reuses the unchanged data/acceptance helpers. It does
not load a model, but does read all declared NPZ and checkpoint bytes.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parents[2]
SUPPORT = REPO / "execution_support/n08"
PINS = {
    "driver": "9595bde814a2528a027e3cf599d78df78270cdbf6a179a3138eb9def99eb55cb",
    "wrapper": "f1381e6a134fdf7d3c879625021fc9473f71ea4b6b0572e1d1ab4a0897ca6de0",
    "accept_fit": "3d58149f0cfd124d6c370d623e3422a35f734b06bd239511782f66c82e486568",
    "data_manifest": "83acb26595e87075a3943c889298659a4a028c389698bf0de4af95905a56f9ff",
}
FILES = {
    "driver": REPO / "scripts/run_corrected_cpu.py",
    "wrapper": REPO / "scripts/run_cpu_task.py",
    "accept_fit": SUPPORT / "accept_fit.py",
    "data_manifest": SUPPORT / "data_manifest.py",
}
DATA_KEYS = (
    "dataset_version", "manifest_body_sha256", "rows_sha256", "npz_all_sha256",
    "annotation_preprocessing_sha256", "n_entries", "acceptance",
)
ARTIFACTS = {
    "best_pt": "best.pt", "last_pt": "last.pt", "fold_summary": "fold_summary.json",
    "log_jsonl": "log.jsonl", "grounding_microbatch_jsonl": "grounding_microbatch.jsonl",
    "attempts_jsonl": "attempts.jsonl",
}
VOLATILE_CHECK = "data re-verified at this acceptance boundary (full npz re-hash)"
METADATA_LIMIT = 8 * 1024 * 1024
SCHEMA = "theia.corrected_cpu.consumption.v1"


class Refusal(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Refusal(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integer(value, label):
    require(type(value) is int or isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value),
            label + ": expected a nonboolean integer")
    return int(value)


def utc(value):
    require(isinstance(value, str), "Missing UTC timestamp")
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None and result.utcoffset() == dt.timedelta(0), "Explicit UTC required")
    return result


def path_under(value, base, *, exists=True):
    path, base = Path(value), Path(base).resolve(strict=True)
    require(path.is_absolute(), "Absolute path required: " + str(path))
    resolved = path.resolve(strict=exists)
    require(path == resolved and resolved.is_relative_to(base), "Path boundary/symlink violation: " + str(path))
    return resolved


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


class Evidence:
    """Stable byte snapshots; source modules execute exactly the bytes checked."""

    def __init__(self):
        self.files = {}
        self.large_files = {}

    def read(self, path, *, binary=False):
        path = Path(path)
        require(path.is_absolute() and path == path.resolve(strict=True),
                "Noncanonical metadata path: " + str(path))
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode), "Not a regular metadata file: " + str(path))
        require(before.st_size <= METADATA_LIMIT, "Metadata size bound: " + str(path))
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "File replaced before read")
            content = stream.read(METADATA_LIMIT + 1)
        after = path.lstat()
        require(len(content) <= METADATA_LIMIT and self._identity(before) == self._identity(after),
                "Metadata changed while reading: " + str(path))
        item = {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if str(path) in self.files:
            require(item == self.files[str(path)], "Evidence changed between reads: " + str(path))
        self.files[str(path)] = item
        return content if binary else json.loads(content, object_pairs_hook=unique_object)

    @staticmethod
    def _identity(st):
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    def sha(self, path):
        self.read(path, binary=True)
        return self.files[str(path)]["sha256"]

    def module(self, name):
        content = self.read(FILES[name], binary=True)
        require(hashlib.sha256(content).hexdigest() == PINS[name], "Unreviewed consumer dependency: " + name)
        module = types.ModuleType("_standalone_" + name)
        module.__file__ = str(FILES[name])
        exec(compile(content, str(FILES[name]), "exec"), module.__dict__)
        return module

    def unchanged(self):
        for name in list(self.files):
            self.read(Path(name), binary=True)
        for name, identity in self.large_files.items():
            path = Path(name)
            require(path == path.resolve(strict=True)
                    and self._identity(path.lstat()) == identity, "Scientific file changed: " + name)

    def watch_large(self, path, base):
        """Helpers hash large bytes; hold their file identities stable across that work."""
        path = path_under(path, base)
        current = path.lstat()
        require(stat.S_ISREG(current.st_mode), "Not a regular scientific file: " + str(path))
        identity = self._identity(current)
        require(str(path) not in self.large_files or self.large_files[str(path)] == identity,
                "Scientific file changed before verification: " + str(path))
        self.large_files[str(path)] = identity


def runtime_identity(python, root):
    # Metadata only: importing torch/numpy/model code is deliberately unnecessary.
    code = (
        "import importlib.metadata as m,json,platform,sys;"
        "names=['torch','numpy','scikit-learn','torchvision','transformers','timm','PyYAML'];"
        "print(json.dumps({'python':platform.python_version(),'executable':sys.executable,"
        "'packages':{n:m.version(n) for n in names}}))"
    )
    result = json.loads(subprocess.check_output(
        [str(python), "-I", "-c", code], cwd=root, text=True, timeout=30,
    ))
    before = python.stat()
    h = hashlib.sha256()
    with python.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    require(Evidence._identity(before) == Evidence._identity(python.stat()), "Runtime changed while hashing")
    result["python_binary_sha256"] = h.hexdigest()
    return result


def source_paths(root, evidence):
    trainer = root / "theia/engine/train.py"
    names = None
    for statement in ast.parse(evidence.read(trainer, binary=True)).body:
        if isinstance(statement, ast.Assign) and any(getattr(t, "id", None) == "TRAINING_CODE" for t in statement.targets):
            names = ast.literal_eval(statement.value)
    require(isinstance(names, (tuple, list)) and names, "Missing static training source closure")
    paths = []
    for name in names:
        require(isinstance(name, str) and name.endswith(".py"), "Unexpected training source type")
        paths.append(path_under(root / name, root))
    require(len(paths) == len(set(paths)) and trainer in paths, "Invalid training source closure")
    paths += [root / "configs/default.yaml", *FILES.values()]
    paths += [root / name for name in ("requirements.txt", "setup.py") if (root / name).is_file()]
    return set(paths)


def _strict_identity(evidence, record, path):
    require(isinstance(record, dict), "Missing E05 file identity")
    evidence.read(path, binary=True)
    require(record == evidence.files[str(path)], "E05 file identity mismatch: " + str(path))


def _matrix(driver, evidence):
    matrix = SUPPORT / "FIT_MATRIX_M1.csv"
    require(evidence.sha(matrix) == driver.MATRIX_SHA, "Frozen matrix changed")
    rows = list(csv.DictReader(evidence.read(matrix, binary=True).decode().splitlines()))
    require(len(rows) == 50 and len({r["fit_id"] for r in rows}) == 50, "Duplicate or incomplete matrix")
    units = {(r["cell"], integer(r["fold"], "fold"), integer(r["init_seed"], "seed")) for r in rows}
    require(units == {(c, f, s) for c in "ABCDE" for f in range(5) for s in (42, 7)}, "Frozen matrix units changed")
    for row in rows:
        driver.validate_row(row)
    return matrix, rows


def _native_e05(binding, worker, result, unit, output, evidence, driver):
    kind, row = binding["kind"], binding["row"]
    task = "M03" if kind == "smoke" else "M04"
    exit_path = path_under(result["e05_exit_receipt"], output / "attempts")
    require(exit_path.name == "EXIT.json" and exit_path.parent.parent == output / "attempts",
            "E05 receipt must be in one direct attempt directory")
    require(exit_path.parent.name.startswith(task + "_" + row["fit_id"] + "_"), "E05 attempt identity changed")
    terminal = evidence.read(exit_path)
    start = evidence.read(exit_path.with_name("START.json"))
    process = evidence.read(exit_path.with_name("PROCESS.json"))
    for receipt in (start, terminal):
        require(receipt.get("schema") == "theia.cpu_attempt.v1" and receipt.get("task_id") == task
                and receipt.get("job_key") == row["fit_id"], "E05 task/job/schema mismatch")
    require(terminal.get("status") == "finished" and type(terminal.get("exit_code")) is int
            and terminal["exit_code"] == 0 and terminal.get("signals") == []
            and terminal.get("runner_error") is None and terminal.get("output_errors") == []
            and terminal.get("stop_reason") == "process_exit", "E05 did not finish cleanly")
    launch = start["binding"]
    fingerprint = digest(launch)
    require(start.get("fingerprint") == terminal.get("fingerprint") == fingerprint, "E05 fingerprint mismatch")
    command = [binding["python"], str(FILES["driver"]), "_worker", "--binding", str(unit / "EXECUTION_BINDING.json")]
    require(launch.get("task_id") == task and launch.get("job_key") == row["fit_id"]
            and launch.get("command") == command and launch.get("cwd") == binding["root"],
            "E05 command/cwd/task mismatch")
    require(process.get("command") == command and type(process.get("pid")) is int and process["pid"] > 0
            and type(process.get("process_group")) is int and process["process_group"] == process["pid"]
            and isinstance(process.get("identity"), str) and process["identity"].strip(),
            "Owned process identity missing or changed")
    require(start.get("resume_receipt") is None, "Unexpected wrapper-level resume")
    _strict_identity(evidence, start.get("runner"), FILES["wrapper"])
    sources = launch.get("sources")
    require(isinstance(sources, list) and len(sources) == len(binding["source_sha256"]), "E05 source closure mismatch")
    by_path = {r.get("path"): r for r in sources if isinstance(r, dict)}
    require(len(by_path) == len(sources) and set(by_path) == set(binding["source_sha256"]), "E05 source identities differ")
    for name, item in by_path.items():
        _strict_identity(evidence, item, Path(name))
    require(isinstance(launch.get("inputs"), list) and len(launch["inputs"]) == 1, "E05 input closure mismatch")
    _strict_identity(evidence, launch["inputs"][0], unit / "EXECUTION_BINDING.json")
    require(isinstance(terminal.get("outputs"), list) and len(terminal["outputs"]) == 1, "E05 output closure mismatch")
    _strict_identity(evidence, terminal["outputs"][0], unit / "WORKER_RESULT.json")
    session = evidence.read(unit / "SESSION.json")
    require(start.get("session_sha256") == evidence.sha(unit / "SESSION.json"), "E05 session changed")
    require(session == {"admission_cutoff_utc": driver.FIXED_CUTOFF, "deadline_utc": driver.FIXED_DEADLINE},
            "Session window changed")
    cutoff, deadline = utc(driver.FIXED_CUTOFF), utc(driver.FIXED_DEADLINE)
    require(utc(start["cutoff_utc"]) == cutoff and utc(start["deadline_utc"]) == deadline, "E05 window mismatch")
    created, begun, launched, ended = map(utc, (
        binding["created_utc"], start["started_utc"], process["launched_utc"], terminal["ended_utc"],
    ))
    require(created <= begun <= launched < cutoff and launched <= utc(worker["ended_utc"]) <= ended <= deadline
            and ended <= utc(result["ended_utc"]), "Receipt chronology/window mismatch")
    return exit_path


def structural(result_path, *, root, output_root, _smoke=False):
    """Return internal verified metadata, never scientific acceptance or fit credit."""
    root, output = Path(root).resolve(strict=True), Path(output_root).resolve(strict=True)
    require(output.is_relative_to(root / "revamp/cpu_runs"), "Output root outside research CPU runs")
    result_path = path_under(result_path, output / "units")
    unit = result_path.parent
    require(result_path.name == "RESULT.json" and unit.parent == output / "units", "Not a native unit RESULT")
    evidence = Evidence()
    driver = evidence.module("driver")
    for name in ("wrapper", "accept_fit", "data_manifest"):
        require(evidence.sha(FILES[name]) == PINS[name], "Unreviewed consumer dependency: " + name)
    result = evidence.read(result_path)
    binding_path, worker_path = unit / "EXECUTION_BINDING.json", unit / "WORKER_RESULT.json"
    binding, worker = evidence.read(binding_path), evidence.read(worker_path)
    require(binding.get("schema") == "theia.corrected_cpu.execution_binding.v1", "Binding schema mismatch")
    require(result.get("schema") == "theia.corrected_cpu.result.v1"
            and worker.get("schema") == "theia.corrected_cpu.technical_receipt.v1", "Native receipt schema mismatch")
    kind, row = binding.get("kind"), binding.get("row")
    require(kind in ("fit", "smoke") and isinstance(row, dict), "Invalid native unit kind/row")
    require(not _smoke or kind == "smoke", "Smoke dependency is not a smoke")
    require(binding.get("root") == str(root) and binding.get("output_root") == str(output)
            and binding.get("unit_dir") == str(unit) and binding.get("support_dir") == str(SUPPORT),
            "Binding root/unit/support mismatch")
    matrix, matrix_rows = _matrix(driver, evidence)
    wanted = driver.select_row(matrix, kind, integer(row.get("order_index"), "order_index"))
    require(row == wanted and unit.name == row["fit_id"], "Unit differs from exact frozen matrix row")
    require(kind == "smoke" or not ("-smoke-" in row["fit_id"] or row["fit_id"].startswith("n08parity")),
            "Bootstrap identity cannot be a full fit")
    for document in (result, worker):
        require(document.get("kind") == kind and document.get("fit_id") == row["fit_id"]
                and document.get("status") == "accepted" and document.get("canonical_adoption") is False,
                "Native unit is not an accepted standalone result")
        require(type(document.get("scientific_fit_count")) is int
                and document["scientific_fit_count"] == int(kind == "fit"), "Producer fit count inconsistent")
        require(document.get("execution_binding_sha256") == evidence.sha(binding_path), "Binding digest mismatch")
    require(type(result.get("wrapper_exit")) is int and result["wrapper_exit"] == 0
            and result.get("resource_stops") == [] and not worker.get("error"),
            "Wrapper/resource/worker failure")
    require(type(worker.get("trainer_exit")) is int and worker["trainer_exit"] == 0, "Trainer did not exit zero")
    require(result.get("technical_receipt") == str(worker_path)
            and worker.get("execution_binding") == str(binding_path), "Native receipt path mismatch")
    for key in ("run_id", "cell"):
        require(worker.get(key) == row[key], "Worker row mismatch: " + key)
    for key in ("fold", "init_seed"):
        require(integer(worker.get(key), key) == integer(row[key], key), "Worker row mismatch: " + key)
    require(binding.get("scientific_acceptance") == driver.SCIENCE
            and binding.get("encoder_sha256") == driver.ENCODER_SHA, "Scientific method/reference changed")
    require((binding.get("admission_cutoff_utc"), binding.get("cooperative_stop_utc"), binding.get("deadline_utc"))
            == (driver.FIXED_CUTOFF, driver.FIXED_STOP, driver.FIXED_DEADLINE), "Bound execution window changed")
    fold = path_under(output / "checkpoints" / row["run_id"] / ("fold" + row["fold"]), output, exists=False)
    require(binding.get("fold_dir") == str(fold), "Bound checkpoint path mismatch")
    require(binding.get("trainer_argv") == driver.build_argv(row, root, output, unit / "COOP_CONTROL.json"),
            "Literal trainer command changed")
    env = {**json.loads(row["env_json"]), "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(output / "tmp")}
    require(binding.get("environment") == env, "Frozen execution environment changed")
    require(binding.get("python") == str(root / ".venv/bin/python"), "Unexpected runtime path")
    require(runtime_identity(Path(binding["python"]), root) == binding.get("runtime"), "Runtime identity changed")
    expected_sources = {str(p) for p in source_paths(root, evidence)}
    require(set(binding.get("source_sha256", {})) == expected_sources, "Incomplete/extra source closure")
    for name, sha in binding["source_sha256"].items():
        require(evidence.sha(Path(name)) == sha, "Bound source changed: " + name)
    manifest = root / driver.PF / "source/manifest_S1_v1/CONTENT_MANIFEST.json"
    expected_paths = {
        "manifest": manifest, "rows": manifest.parent / "rows.jsonl",
        "data_acceptance": root / driver.PF / "compute/M1_ACCEPTANCE_S1A_v1.json",
        "splits": root / driver.SPLITS,
    }
    for key, path in expected_paths.items():
        require(binding.get(key) == str(path_under(path, root)), "Input path changed: " + key)
    acceptance = evidence.read(expected_paths["data_acceptance"])
    require(acceptance.get("test_fixture_run") is False
            and (acceptance.get("options") or {}).get("allow_test_fixtures") is False
            and acceptance.get("checks") and all(c.get("status") == "pass" for c in acceptance["checks"]),
            "Real source acceptance absent/failed")
    require(acceptance.get("decision") == "accepted", "Data acceptance decision failed")
    extra = [path_under(root / acceptance[key]["path"], root) for key in ("registry", "checker")]
    base_inputs = {matrix, *expected_paths.values(), *extra}
    dependencies = {}
    smoke_finished = None
    if kind == "fit":
        smoke_row = driver.select_row(matrix, "smoke", 0)
        smoke = structural(output / "units" / smoke_row["fit_id"] / "RESULT.json",
                           root=root, output_root=output, _smoke=True)
        dependencies = {str(p): smoke["evidence"].sha(p) for p in (
            smoke["unit"] / "EXECUTION_BINDING.json", smoke["unit"] / "RESULT.json",
            smoke["unit"] / "WORKER_RESULT.json", smoke["e05_exit"],
        )}
        require(binding.get("smoke_dependency") == dependencies, "Smoke dependency binding mismatch")
        for key in ("source_sha256", "runtime", "scientific_acceptance", "encoder_sha256"):
            require(smoke["binding"].get(key) == binding.get(key), "Stale smoke: " + key)
        for path in base_inputs:
            require(smoke["binding"]["input_sha256"].get(str(path)) == binding["input_sha256"].get(str(path)),
                    "Smoke input changed: " + str(path))
        require(all(smoke["worker"]["data"].get(k) == worker.get("data", {}).get(k) for k in DATA_KEYS),
                "Smoke data identity changed")
        smoke_finished = utc(smoke["result"]["ended_utc"])
        evidence.files.update(smoke["evidence"].files)
    else:
        require(not binding.get("smoke_dependency"), "Smoke has a recursive dependency")
    wanted_inputs = {str(p) for p in base_inputs | {unit / "SESSION.json", unit / "COOP_CONTROL.json"}} | set(dependencies)
    require(set(binding.get("input_sha256", {})) == wanted_inputs, "Incomplete/extra input closure")
    for name, sha in binding["input_sha256"].items():
        require(evidence.sha(Path(name)) == sha, "Bound input changed: " + name)
    require(evidence.sha(expected_paths["splits"]) == driver.SPLITS_SHA, "Frozen splits changed")
    for key, path in zip(("registry", "checker"), extra):
        require(evidence.sha(path) == acceptance[key]["sha256"], "Source acceptance binding mismatch")
    accepted_manifest = path_under(root / acceptance["manifest"], root)
    require(accepted_manifest == manifest
            and acceptance["evidence_sha256"][acceptance["manifest"]] == evidence.sha(manifest),
            "Source acceptance manifest mismatch")
    control = evidence.read(unit / "COOP_CONTROL.json")
    policy = binding["resource_policy"]
    require(control == {
        "deadline_utc": driver.FIXED_STOP, "check_every_steps": 1,
        "min_free_bytes": policy["external_reserve_bytes"], "ckpt_forecast_bytes": 351227877,
        "fit_bytes_budget": 1100000000, "max_battery_pause_s": 0, "epoch_forecast_s": 0,
    }, "Cooperative control contract changed")
    stored_data, declared_data = worker.get("data", {}), binding.get("data", {})
    require(all(stored_data.get(k) == declared_data.get(k) for k in DATA_KEYS)
            and stored_data.get("dataset_version") == "M1" and stored_data.get("npz_rehashed") is True
            and stored_data.get("verified_utc"), "Worker data binding incomplete")
    require(stored_data["acceptance"] == {
        "decision": "accepted", "sha256": evidence.sha(expected_paths["data_acceptance"]),
    }, "Worker source acceptance identity mismatch")
    verdict, identity = worker.get("acceptance", {}), worker.get("fit_identity", {})
    require(verdict.get("accepted") is True and verdict.get("failed_checks") == []
            and verdict.get("checks") and all(c.get("ok") is True for c in verdict["checks"]),
            "Technical acceptance is absent/failed")
    require(verdict.get("fit_id") == row["fit_id"] and verdict.get("fold_dir") == str(fold),
            "Acceptance unit/fold mismatch")
    require(worker.get("fit_identity_sha256") == digest(identity), "Fit identity digest mismatch")
    for key in DATA_KEYS[:5]:
        require(identity.get(key) == stored_data.get(key), "Fit data identity mismatch: " + key)
    require(identity.get("init_seed") == integer(row["init_seed"], "seed")
            and identity.get("partition_seed") == 1337 and identity.get("initial_state_sha256"),
            "Fit initialization identity mismatch")
    require(identity.get("objective") == {
        "region_pooling": row["region_pooling"], "ground_weight": float(row["ground_weight"]),
        "ground_target": row["ground_target"], "readout_map": row["readout_map"],
    }, "Fit objective mismatch")
    require(identity.get("selection_rule") == {
        "monitor": '[["egfr_logloss", -1.0]]', "min_epochs": 10, "patience": 8,
        "max_epochs": 2 if kind == "smoke" else 40, "ties": "earliest",
    }, "Fit selection method mismatch")
    for key in ARTIFACTS:
        item = verdict.get("identities", {}).get(key)
        require(isinstance(item, dict) and type(item.get("bytes")) is int and item["bytes"] > 0
                and re.fullmatch("[a-f0-9]{64}", item.get("sha256", "")), "Missing accepted output identity: " + key)
    for key in ("best_pt", "last_pt"):
        require(identity.get(key + "_sha256") == verdict["identities"][key]["sha256"], "Checkpoint identity mismatch")
    e05_exit = _native_e05(binding, worker, result, unit, output, evidence, driver)
    if smoke_finished is not None:
        require(smoke_finished <= utc(evidence.read(e05_exit.with_name("START.json"))["started_utc"]),
                "Fit attempt started before its bound smoke completed")
    common = {
        "source_sha256": binding["source_sha256"], "runtime": binding["runtime"],
        "inputs": {str(p): binding["input_sha256"][str(p)] for p in sorted(base_inputs)},
        "data": {k: stored_data[k] for k in DATA_KEYS}, "scientific_acceptance": binding["scientific_acceptance"],
        "encoder_sha256": binding["encoder_sha256"],
    }
    evidence.unchanged()
    return {
        "binding": binding, "worker": worker, "result": result, "unit": unit, "fold": fold,
        "e05_exit": e05_exit, "evidence": evidence, "driver": driver,
        "execution_set_sha256": digest(common), "matrix_ids": {r["fit_id"] for r in matrix_rows},
    }


def _verdict_digest(value):
    copy = json.loads(json.dumps(value))
    for check in copy.get("checks", []):
        if check.get("check") == VOLATILE_CHECK:
            check["detail"] = "<volatile: verified_utc>"
    return digest(copy)


def scientific(bundle):
    """Full current verification using the exact unchanged acceptance implementation."""
    evidence, binding, worker = bundle["evidence"], bundle["binding"], bundle["worker"]
    require(Path(sys.executable) == Path(binding["python"]),
            "Run full verification with the bound research Python interpreter")
    # Refuse path redirection before the unchanged helper opens a checkpoint.
    for name in ARTIFACTS.values():
        evidence.watch_large(bundle["fold"] / name, Path(binding["output_root"]) / "checkpoints")
    manifest = evidence.read(Path(binding["manifest"]))
    for entry in manifest["entries"]:
        evidence.watch_large(Path(binding["root"]) / entry["npz_path"], Path(binding["root"]))
    for line in evidence.read(Path(binding["rows"]), binary=True).decode().splitlines():
        if line.strip():
            row = json.loads(line, object_pairs_hook=unique_object)
            evidence.watch_large(Path(binding["root"]) / row["npz"], Path(binding["root"]))
    dm, af = evidence.module("data_manifest"), evidence.module("accept_fit")
    data = dm.verify_manifest(binding["manifest"], binding["rows"], "M1", binding["root"],
                              binding["data_acceptance"], rehash=True)
    require(all(data.get(k) == worker["data"].get(k) for k in DATA_KEYS), "Current data identity differs")
    splits = evidence.read(Path(binding["splits"]))
    require(not dm.partition_invariance(binding["rows"], splits), "Frozen partition invariance failed")
    # Same expected values as the pinned driver's expected_for, using the already
    # verified helper module instead of reopening executable helper source.
    labels = dm.rows_labels(binding["rows"])
    microbatches = {fold["train_microbatches_per_epoch"] for fold in splits["folds"]}
    require(len(microbatches) == 1, "Inconsistent expected microbatch counts")
    expected = {
        **binding["scientific_acceptance"], "splits": splits, "data": data,
        "rows_labels": {p: v["egfr"] for p, v in labels.items()},
        "rows_has_mask": {p: v["has_mask"] for p, v in labels.items()},
        "rows_sha256": data["rows_sha256"], "microbatches_per_epoch": microbatches.pop(),
        "encoder_sha256": binding["encoder_sha256"],
    }
    if binding["kind"] == "smoke":
        expected.update(max_epochs=2, allow_truncation=True)
    verdict = af.accept(binding["row"], bundle["fold"], expected, trainer_exit=worker["trainer_exit"])
    require(verdict.get("accepted") is True and _verdict_digest(verdict) == _verdict_digest(worker["acceptance"]),
            "Fresh technical verdict differs or fails")
    identity = af.fit_identity(binding["row"], verdict, expected)
    require(af.canon_sha(identity) == worker["fit_identity_sha256"] == af.canon_sha(worker["fit_identity"]),
            "Fresh scientific fit identity differs")
    require(runtime_identity(Path(binding["python"]), Path(binding["root"])) == binding["runtime"],
            "Runtime changed during scientific verification")
    evidence.unchanged()
    return {"current_verdict_sha256": _verdict_digest(verdict), "current_fit_identity_sha256": af.canon_sha(identity)}


def verify_results(results, *, root, output_root, mode="structural"):
    require(mode in ("structural", "full"), "Unknown verification mode")
    require(results and len(results) <= 51, "Supply 1–51 explicit native results")
    bundles = [structural(path, root=root, output_root=output_root) for path in results]
    seen = {key: set() for key in ("unit", "fold", "fit_identity", "best_pt")}
    initial_by_seed, sets = {}, set()
    for bundle in bundles:
        worker, binding = bundle["worker"], bundle["binding"]
        values = {"unit": binding["row"]["fit_id"], "fold": str(bundle["fold"]),
                  "fit_identity": worker["fit_identity_sha256"],
                  "best_pt": worker["acceptance"]["identities"]["best_pt"]["sha256"]}
        for key, value in values.items():
            require(value not in seen[key], "Duplicate native unit evidence: " + key)
            seen[key].add(value)
        if binding["kind"] == "fit":
            sets.add(bundle["execution_set_sha256"])
            seed = integer(binding["row"]["init_seed"], "seed")
            initial_by_seed.setdefault(seed, set()).add(worker["fit_identity"]["initial_state_sha256"])
    require(len(sets) <= 1, "Native execution-set identities differ")
    require(all(len(values) == 1 for values in initial_by_seed.values()), "Initialization differs within a seed")
    require(len(set.union(*initial_by_seed.values())) == len(initial_by_seed) if initial_by_seed else True,
            "Different seeds share initialization")
    units = []
    for bundle in bundles:
        binding, worker = bundle["binding"], bundle["worker"]
        current = scientific(bundle) if mode == "full" else {}
        evidence = sorted(bundle["evidence"].files.values(), key=lambda x: x["path"])
        units.append({
            "fit_id": binding["row"]["fit_id"], "kind": binding["kind"],
            "status": "verified_current" if mode == "full" else "structural_only",
            "scientific_fit_count": int(mode == "full" and binding["kind"] == "fit"),
            "execution_set_sha256": bundle["execution_set_sha256"],
            "native_bundle_sha256": digest(evidence), "inputs": evidence,
            "fit_identity_sha256": worker["fit_identity_sha256"], **current,
        })
    # Recheck shared metadata after all units, so later verification cannot hide a race.
    for bundle in bundles:
        bundle["evidence"].unchanged()
    verified = {u["fit_id"] for u in units if u["scientific_fit_count"] == 1}
    return {
        "schema": SCHEMA, "mode": mode, "authority": "e05_native_execution",
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "verified_current" if mode == "full" else "structural_only",
        "scientific_fit_count": len(verified), "units": units,
        "full_matrix_verified": verified == bundles[0]["matrix_ids"],
        "canonical_adoption": False, "readout_ready": False,
        "lineage": "Native standalone receipts; no legacy release, controller receipt or historical witness fabricated.",
    }


def save_report(path, report, output_root):
    directory = Path(output_root).resolve(strict=True) / "consumption"
    directory.mkdir(exist_ok=True)
    path = path_under(path, directory, exists=False)
    require(path.parent == directory and path.suffix == ".json", "Report must be a new direct consumption/*.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--result", action="append", required=True, type=Path)
    parser.add_argument("--mode", choices=("structural", "full"), default="structural")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = verify_results(args.result, root=args.root, output_root=args.output_root, mode=args.mode)
        save_report(args.out, report, args.output_root)
        print(json.dumps({"status": report["status"], "scientific_fit_count": report["scientific_fit_count"],
                          "report": str(args.out)}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "refused", "scientific_fit_count": 0,
                          "reason": type(exc).__name__ + ": " + str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
