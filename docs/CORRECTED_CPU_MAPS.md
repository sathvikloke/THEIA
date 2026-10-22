# Native five-fold maps component

`execution_support/n08/run_maps_standalone_v1.py` runs one M1 cell/seed's **five accepted folds**, for either `best` or `last`. It does not wait for all 50 fits and does not call the all-matrix readout API. This component has been tested on synthetic metadata only; no actual maps or scientific verification were run during implementation.

The published native readout entry still refuses with `native_map_integration_unavailable`. A separately reviewed downstream consumer change is required before these receipts can enable any final readout. Passing component tests does not complete M06/M07 scientifically or establish canonical adoption.

## Execution contract

Use the bound research interpreter. All paths below must be explicit absolute private paths; the maps directory must be new and beneath the native output root's `maps/` directory.

```sh
PYTHON run_maps_standalone_v1.py \
  --root ROOT --output-root OUT \
  --cell CELL --run RUN --which best \
  --maps-dir OUT/maps/NEW_PRIVATE_ATTEMPT
```

Here `PYTHON` is `ROOT/.venv/bin/python` and the runner is the absolute path to the new component. `RUN`, `CELL` and all five folds must match the pinned M1 matrix exactly. `best` and `last` are distinct jobs with new target directories; the component does not retry or choose a checkpoint by outcome.

The coordinator first performs zero-credit structural verification to reconstruct the five-fold context and its bound paths. It refuses conflicting runtime, source, data, initialization or checkpoint evidence. It measures the mounted workspace/backing association and reuses the pinned CPU driver's resource checks. The current-session minimum reserves are **1,000,000,000 internal bytes, 50,000,000,000 THEIA bytes and 100,000,000,000 Research/backing bytes**. Each external requirement additionally includes the existing 1,100,000,000-byte forecast. Higher applicable reserves or forecasts from any of the five bound source bindings are retained. AC power is required. No model or dataset is loaded at this preparation stage.

The map binding records distinct `theia_reserve_bytes` and `backing_reserve_bytes`; validation reconstructs their required values from the current floors and the five source bindings. Preparation, coordinator monitoring and worker checks use the same enforcement function. The unchanged driver's shared probe enforces a common baseline, then the maps component checks the distinct requirements against that same measured snapshot. Its returned samples record the actual distinct byte requirements. A default or lower inherited shared reserve cannot reduce either current-session floor.

Only after preflight does preparation create the new maps directory, its `MAP_BINDING.json` and `SESSION.json`, and the intended owned **`OUT/map_attempts` directory**. E05 requires that attempt root to exist; callers must not ask it to create a missing mount path. Temporary files stay in the already bound external `OUT/tmp`.

The coordinator constructs this literal argument array:

```text
PYTHON REPO/scripts/run_cpu_task.py
  --session MAPS_DIR/SESSION.json
  --attempt-root OUT/map_attempts
  --cwd ROOT
  --task-id M06
  --job-key RUN-WHICH
  --stop-grace-seconds 60
  [--source PATH for every sorted source in MAP_BINDING.source_sha256]
  --input MAPS_DIR/MAP_BINDING.json
  --output MAPS_DIR/MAP_RECEIPT_RUN_WHICH.json
  --
  PYTHON REPO/execution_support/n08/run_maps_standalone_v1.py
    _worker --binding MAPS_DIR/MAP_BINDING.json
```

There is no production command override or test-authority flag. The fixed session is admission before **2026-10-20 12:10 UTC**, safe stop beginning at **12:28 UTC**, and hard deadline **12:30 UTC**. The coordinator samples resources every 15 seconds, checks time after probes, and signals only its owned E05 child on a resource/stop failure. E05 controls the worker's process group and enforces its absolute deadline. The worker checks admission before full verification, then rechecks admission/resources **after the metadata recheck and log creation, immediately before maps launch**. If either setup step crosses cutoff or safe-stop, the map subprocess is not started; the log and failure receipt remain. Partial files and failed receipts are preserved.

Inside E05, the worker proves its live PID/process group matches `PROCESS.json`, the command/cwd/source/input/session matches `START.json`, and its environment matches the frozen CPU/offline/thread settings. It then calls the unchanged native verifier with `mode="full"` for exactly the same five results. Structural status or five-fold context changes cannot authorize a map job.

The retained map command is:

```text
PYTHON ROOT/revamp/campaign/nine_hour_corrected_inputs_2026-10-02/n08/src/n08/readout/n08_maps_param.py
  --cell CELL --run RUN --which WHICH
  --ckpt-root OUT/checkpoints --out-dir MAPS_DIR
  --splits BOUND_SPLITS --rows BOUND_ROWS
  --data-manifest BOUND_MANIFEST --dataset-version M1
  --data-acceptance BOUND_ACCEPTANCE
```

Checkpoint configuration, CPU operation, reload controls, map definitions, query aggregation, scoring, batching and OOF membership remain those of the retained program. No extra scientific options are introduced. Its original research path is required because it discovers its import root through `.git`; the wrapper verifies that discovery still resolves to the bound research root.

## Source and receipt identities

The retained map program remains byte-identical:

```text
0688823dd61fc4f0bd3f8a9ed8850251a8a19c29e80565a674734b7e4c3aa7ed
```

The published native verifier remains pinned at:

```text
013e165ee000bd7876f89c25eabc6911c98359fed148d12fb729034144319b42
```

The E05 source list includes the full source closure already bound by the five fits, the new runner, native verifier, unchanged original map program and its sibling `data_manifest.py`, plus the map-only `spatial_comparators.py` and `spatial_projection.py` sources. Map-only additions belong to this map binding; existing fit bindings are not altered.

The worker writes schema `theia.corrected_cpu.map_receipt.v1`. Its nested `context.checkpoints` records contain the exact fold/fit/path/bytes/hash, `native_result_path`, `native_bundle_sha256`, `execution_set_sha256` and fresh fit-identity hash. The receipt also binds the command, cwd, runtime, map sources, true map exit code, log/output identities, controls, patient set and map definition. Native digests never masquerade as legacy fit-receipt hashes.

A worker receipt's `accepted` status is provisional until terminal verification. The coordinator records actual wrapper exit, resource stops and the single new E05 EXIT path in `EXECUTION_RESULT.json`. `verify_map_receipt(receipt_path, exit_path)` is a separate Python API requiring those two explicit paths. It checks E05 START/PROCESS/EXIT, exact receipt output hash, clean exit with no signals, coordinator success, unchanged sources/session/context, fresh full verification of the same five fits, current map/log hashes, selected checkpoints, controls, accepted OOF membership, frozen folds, spatial counts and finite last-checkpoint probabilities.

E05 hashes the closed worker receipt. The receipt does not hash the future E05 EXIT, avoiding a self-hash cycle. Nonzero or signalled jobs, missing/control-failed outputs, stale targets, altered context or changed sources/outputs remain unusable. Existing outputs are never adopted or overwritten.

The original producer's `check_output` body and the map program's `MAP_DEFINITION`/`CODE_FILES` assignments are retained exactly. Native consumer checks add the required native provenance edges while retaining the existing patient/fold/map consistency requirements. There is no old controller, release, witness or accepted-receipt alias.

## Focused validation

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp python3 -m unittest discover \
  -s tests -p test_run_maps_standalone_v1.py -v
```

Tests use tiny synthetic metadata, the real native batch-verifier function with explicit structural/scientific substitutions, and one actual E05 wrapper run whose child only updates a synthetic JSON receipt. Worker tests replace the map subprocess with a fixture writer; source/runtime/resource/membership substitutions are confined to tests. No production override enables these substitutions.

The tests check five-fold forwarding and refusal, bound context and source changes, output/OOF/fold/map checks, true nonzero exits, control-failed preservation, resource stop handling and native terminal receipt validation. They do not validate model outputs, repeat earlier full suites, grant real fit/map credit, or exercise actual drive/resource admission.
