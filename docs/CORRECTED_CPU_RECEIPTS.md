# Consuming native corrected CPU receipts

`execution_support/n08/standalone_receipts.py` verifies receipts produced by the
reviewed `run_corrected_cpu.py` and `run_cpu_task.py` implementations. It produces
the new `theia.corrected_cpu.consumption.v1` schema. It creates no historical
controller release, `ACCEPTED.json`, campaign authorization, or fit witness.
Those older consumer contracts remain unchanged.

The authority for the execution record is the actual E05
`START.json → PROCESS.json → EXIT.json → WORKER_RESULT.json` chain, together with
the native `EXECUTION_BINDING.json` and `RESULT.json`. This is a local integrity
chain, not a cryptographic signature or proof against an actor able to rewrite
every receipt and source file. Recorded process identity is historical; the
consumer does not attach to a possibly recycled PID.

## Modes and credit

The default `--mode structural` reads bounded metadata and source files and
queries runtime package metadata. It does not open training outputs, checkpoint
bytes, or referenced NPZ files. It reports `status: structural_only`,
`scientific_fit_count: 0`, `full_matrix_verified: false`, and zero credit for
every unit, even when a producer claimed successful acceptance.

`--mode full` additionally rehashes all declared NPZ data and runs the unchanged,
hash-pinned `data_manifest.py` and `accept_fit.py` scientific checks. It requires
the fresh verdict, output identities, and fit identity to reproduce the stored
ones. Only the documented data-verification UTC detail is normalized. A smoke
always contributes zero fits. A fully verified matrix unit contributes one.
No model is loaded and no pixel array is decoded, but the full mode reads large
files and can use substantially more resources than structural inspection.
Do not run it without the task's resource authorization.
Run full mode with the bound research `.venv/bin/python` so the unchanged
metric helper executes in the same recorded package environment. The verifier
refuses full mode under a different interpreter.

Both modes report `canonical_adoption: false` and `readout_ready: false`. An
explicit downstream native readout branch is separate work. A partial accepted
set is not the 50-unit matrix; `full_matrix_verified` becomes true only after
current full verification of exactly the frozen matrix's 50 distinct fits.

## What is checked

- Exact native schemas, matrix row, derived smoke row, unit paths, literal
  trainer command, six-thread environment, scientific settings, and fixed
  October 20 cutoff/cooperative-stop/deadline.
- The reviewed producer/helper source pins, complete trainer source closure,
  complete bound inputs, frozen matrix/split hashes, accepted source-data
  checker/registry/manifest identities, and current Python/package identity.
- Successful E05 status with `stop_reason: process_exit`, no signals or runner
  errors, true integer zero exits, exact command/cwd/source/input/output
  fingerprints, owned process/group identity, session hash, and chronology.
- For every full fit, the complete structural chain of its bound smoke, all four
  recorded smoke-dependency hashes, matching source/runtime/data/method, and
  smoke completion no later than the fit's E05 attempt start. Preparation may
  begin before smoke completion.
- Stored technical verdict and fit identity consistency; duplicate units,
  folds, checkpoint hashes and fit identities; shared input/runtime/source
  identity across fits; equal initialization within a seed and distinct
  initialization between seeds.
- Canonical paths, regular files, bounded metadata reads, duplicate JSON-key
  refusal, stable metadata snapshots, and final rechecks. Full verification
  also holds the inode/size/mtime/ctime of every scientific file stable across
  helper verification and the final report boundary.

The unchanged scientific helper retains the frozen membership/OOF checks,
label and metric recomputation, checkpoint hashes and expected sizes,
selection-history/patience replay, complete finite microbatch ledgers,
grounding objectives, initialization/encoder identities, and CPU method
checks. The consumer does not substitute a wrapper exit code for those checks.

## Explicit invocation

Run from the repository with the research root and output root explicitly
specified. Each `--result` names one native `units/<fit_id>/RESULT.json`; repeat
that option to verify more units. The consumer does not discover or silently
add unrelated results.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 execution_support/n08/standalone_receipts.py \
  --root /Volumes/THEIA_Workspace/THEIA \
  --output-root /Volumes/THEIA_Workspace/THEIA/revamp/cpu_runs/2026-10-20_until_0730/n08 \
  --result /absolute/path/to/the/output/units/FIT_ID/RESULT.json \
  --mode structural \
  --out /Volumes/THEIA_Workspace/THEIA/revamp/cpu_runs/2026-10-20_until_0730/n08/consumption/inspection.json
```

The result path placeholder must be replaced with an actual unit under the given
output root. A report must be a new direct `consumption/*.json` file under that
root. Existing reports are never overwritten. A refusal exits nonzero and
prints zero fit credit; a successful structural inspection exits zero but still
grants zero scientific credit. Successful reports bind the exact small input
identities and the native evidence-bundle digest used for each unit.

## Bounded fixture tests

```sh
TMPDIR=/Volumes/THEIA_Workspace/THEIA/revamp/cpu_runs/2026-10-20_until_0730/n08/tmp \
PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s tests -p test_standalone_receipts.py -v
```

Tests require an existing external `TMPDIR` and create only tiny synthetic
files. They reuse the real public matrix and reviewed source files, patch the
synthetic split pin/runtime only inside test code, and execute the real E05
wrapper on a tiny stub child. Full-mode orchestration tests use the unchanged
data-manifest byte verifier on placeholder bytes and a stub technical acceptance
helper. They test forwarding and refusal behavior and confer no scientific
acceptance on those fixtures. No real patient arrays, checkpoints, models,
training jobs, or network operations are involved.
