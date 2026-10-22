"""Native readout boundary tests: synthetic metadata, no scientific execution."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "execution_support/n08/readout_standalone_v1.py"
SUPPORT = SOURCE.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These metadata tests must not execute NumPy/SciPy scientific code.
        np = types.ModuleType("numpy")
        scipy = types.ModuleType("scipy")
        stats = types.ModuleType("scipy.stats")
        stats.beta = object()
        stats.rankdata = lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("statistics reached"))
        with mock.patch.dict("sys.modules", {"numpy": np, "scipy": scipy, "scipy.stats": stats}):
            cls.rd = load(SOURCE, "native_readout_under_test") if SOURCE.exists() else None

    def setUp(self):
        self.assertIsNotNone(self.rd, "Versioned native readout adapter has not been implemented")
        self.temp = tempfile.TemporaryDirectory(prefix="native-readout-metadata-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / "revamp/cpu_runs/synthetic"
        self.out.mkdir(parents=True)
        self.native = self.rd.load_native_verifier()
        with (SUPPORT / "FIT_MATRIX_M1.csv").open() as stream:
            self.rows = list(csv.DictReader(stream))
        self.bundles = {}
        for row in self.rows:
            unit = self.out / "units" / row["fit_id"]
            fold = self.out / "checkpoints" / row["run_id"] / ("fold" + row["fold"])
            identity = {"initial_state_sha256": digest(row["init_seed"])}
            fit_sha = digest(identity | {"fit_id": row["fit_id"]})
            identities = {key: {"sha256": digest([row["fit_id"], key]), "bytes": 1}
                          for key in ("best_pt", "last_pt", "fold_summary")}
            worker = {"fit_identity": identity | {"fit_id": row["fit_id"]},
                      "fit_identity_sha256": fit_sha, "data": {"dataset_version": "M1"},
                      "acceptance": {"identities": identities}}
            binding = {"row": row, "root": str(self.root), "output_root": str(self.out),
                       "unit_dir": str(unit), "fold_dir": str(fold)}
            result = {"synthetic": True}
            files = {}
            for name, value in (("EXECUTION_BINDING.json", binding),
                                ("WORKER_RESULT.json", worker), ("RESULT.json", result)):
                item = put(unit / name, value)
                files[item["path"]] = item
            evidence = types.SimpleNamespace(files=files, unchanged=lambda: None)
            self.bundles[str(unit / "RESULT.json")] = {
                "binding": binding | {"kind": "fit"}, "worker": worker,
                "result": result, "unit": unit, "fold": fold, "evidence": evidence,
                "execution_set_sha256": "a" * 64,
                "matrix_ids": {r["fit_id"] for r in self.rows},
            }
        self.structural = mock.patch.object(
            self.native, "structural", side_effect=lambda path, **_kw: self.bundles[str(path)])
        self.science = mock.patch.object(self.native, "scientific", side_effect=lambda bundle: {
            "current_verdict_sha256": digest(bundle["worker"]["acceptance"]),
            "current_fit_identity_sha256": bundle["worker"]["fit_identity_sha256"],
        })
        self.structural.start()
        self.science.start()
        self.addCleanup(self.structural.stop)
        self.addCleanup(self.science.stop)
        self.loader = mock.patch.object(self.rd, "load_native_verifier", return_value=self.native)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def verify(self):
        return self.rd.verify_native_fits(self.root, self.out)

    def test_full_verifier_boundary_yields_native_fifty_fit_provenance(self):
        verified = self.verify()
        self.assertEqual(verified["verification"]["scientific_fit_count"], 50)
        self.assertEqual(verified["verification"]["mode"], "full")
        self.assertEqual(len(verified["fits"]), 50)
        self.assertEqual(verified["execution_set_sha256"], "a" * 64)
        record = next(iter(verified["fits"].values()))
        self.assertIn("native_bundle_sha256", record)
        self.assertNotIn("receipt_sha256_now", record)
        self.assertNotIn("campaign_fingerprint", record)
        self.assertNotIn("training_release_sha256", record)

    def test_structural_zero_credit_cannot_supply_readout_fits(self):
        real = self.native.verify_results
        with mock.patch.object(self.native, "verify_results",
                               side_effect=lambda paths, **kw: real(paths, **(kw | {"mode": "structural"}))):
            with self.assertRaisesRegex(self.rd.AlignmentError, "full|50"):
                self.verify()

    def test_incomplete_fifty_roster_refused_before_scientific_boundary(self):
        next(iter(self.bundles.values()))["unit"].joinpath("RESULT.json").unlink()
        with self.assertRaisesRegex(self.rd.AlignmentError, "50|missing"):
            self.verify()

    def test_native_verifier_failure_is_not_accepted(self):
        with mock.patch.object(self.native, "verify_results", side_effect=self.native.Refusal("fixture failure")):
            with self.assertRaisesRegex(self.rd.AlignmentError, "fixture failure"):
                self.verify()

    def test_duplicate_checkpoint_between_distinct_fits_refused(self):
        first, second = list(self.bundles.values())[:2]
        second["worker"]["acceptance"]["identities"]["best_pt"] = copy.deepcopy(
            first["worker"]["acceptance"]["identities"]["best_pt"])
        with self.assertRaisesRegex(self.rd.AlignmentError, "Duplicate|shared"):
            self.verify()

    def test_cross_seed_initialization_reuse_refused(self):
        for bundle in self.bundles.values():
            bundle["worker"]["fit_identity"]["initial_state_sha256"] = "f" * 64
        with self.assertRaisesRegex(self.rd.AlignmentError, "initialization|initial-state"):
            self.verify()

    def test_cross_execution_set_refused(self):
        next(iter(self.bundles.values()))["execution_set_sha256"] = "b" * 64
        with self.assertRaisesRegex(self.rd.AlignmentError, "execution-set"):
            self.verify()

    def test_worker_mutation_after_full_verification_refused(self):
        first = next(iter(self.bundles.values()))
        put(first["unit"] / "WORKER_RESULT.json", first["worker"] | {"changed": True})
        with self.assertRaisesRegex(self.rd.AlignmentError, "changed|identity"):
            self.verify()

    def test_native_bundle_digest_mutation_refused(self):
        real = self.native.verify_results
        def mutate(paths, **kwargs):
            report = real(paths, **kwargs)
            report["units"][0]["native_bundle_sha256"] = "0" * 64
            return report
        with mock.patch.object(self.native, "verify_results", side_effect=mutate):
            with self.assertRaisesRegex(self.rd.AlignmentError, "bundle"):
                self.verify()

    def test_missing_native_maps_refuses_final_endpoint_without_statistics(self):
        verified = self.verify()
        with mock.patch.object(self.rd, "readout", side_effect=AssertionError("scientific readout reached")):
            with self.assertRaisesRegex(self.rd.AlignmentError, "native_map_integration_unavailable"):
                self.rd.require_native_maps(verified, None, "best")

    def test_legacy_map_path_does_not_bypass_native_map_requirement(self):
        verified = self.verify()
        maps = self.root / "legacy_maps"
        maps.mkdir()
        put(maps / "MAP_RECEIPT.json", {"status": "accepted", "schema": "n08.map_receipt.v1"})
        with self.assertRaisesRegex(self.rd.AlignmentError, "native_map_integration_unavailable"):
            self.rd.require_native_maps(verified, maps, "last")

    def test_design_duplicate_and_wrong_objective_are_refused(self):
        duplicate = copy.deepcopy(self.rows)
        duplicate[1] = duplicate[0]
        self.assertTrue(self.rd.matrix_identity_problems(duplicate))
        changed = copy.deepcopy(self.rows)
        changed[0]["ground_weight"] = "0.9"
        self.assertTrue(self.rd.check_design(changed, "M1"))

    def test_cli_refuses_unavailable_maps_before_full_verification(self):
        destination = self.root / "refusal"
        with mock.patch.object(self.rd, "verify_native_fits",
                               side_effect=AssertionError("expensive full verification reached")) as verify:
            with mock.patch.object(self.rd, "readout",
                                   side_effect=AssertionError("scientific readout reached")) as readout:
                status = self.rd.main(["--root", str(self.root), "--output-root", str(self.out),
                                       "--out-dir", str(destination)])
        result = json.loads((destination / "N08_NATIVE_READOUT_REFUSAL_best.json").read_text())
        self.assertEqual(status, 2)
        self.assertEqual(result["refusal"], "native_map_integration_unavailable")
        self.assertEqual(result["native_full_fits_verified"], 0)
        self.assertEqual(result["scientific_readout_count"], 0)
        self.assertFalse(result["readout_complete"])
        verify.assert_not_called()
        readout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
