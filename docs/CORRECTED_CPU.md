# Corrected M1 CPU execution

`scripts/run_corrected_cpu.py` prepares and executes one October 20 CPU unit
against the actual mounted THEIA trainer. It uses the exported acceptance
helpers in `execution_support/n08`, the frozen 50-row M1 matrix, the real
accepted M1 manifest and the frozen patient splits. It does not use the old
campaign controller or create a historical release.

Preparation reads small source/input files, package metadata and storage/power
information. It does not load a model, read NPZ arrays or launch training.
All generated bindings, controls, temporary files, checkpoints and receipts
remain in the new external output directory.

```sh
ROOT=/Volumes/THEIA_Workspace/THEIA
DRIVER=/Users/sathvikloke/.codex/theia_planning/2026-10-15_execution_backlog/publish_repo/scripts/run_corrected_cpu.py
"$ROOT/.venv/bin/python" "$DRIVER" prepare \
  --root "$ROOT" \
  --output-root "$ROOT/revamp/cpu_runs/2026-10-20_until_0730/n08" \
  --kind smoke
```

Review the resulting `EXECUTION_BINDING.json` and printed literal trainer
arguments before executing the printed `next_command`. Execution needs the
current session's CPU slot approval. The smoke is cell B, mean pooling, union
grounding, weight 0.5, initialization 42, partition seed 1337 and fold 0. It uses
the retained 40-epoch schedule with `train.stop_after_epochs=2`, has no resume
flag, and always contributes zero scientific fits.

After a successful smoke, prepare a full fit with `--kind fit --order-index 1`,
then execute its binding. Repeat the fixed matrix order one unit at a time.
Full-fit preparation requires this output root's accepted smoke result, actual
zero-exit E05 receipt and technical acceptance with rehashed arrays. Its source,
configuration, helpers, runtime and data identities must still match. Those
smoke receipts are added to the fit's frozen input identities, so later receipt
changes also refuse execution.
The driver does not schedule successors or grant resource slots. A shared lock
prevents simultaneous executions within the output root. Full fits retain the
matrix's resume flag and zero retry allowance; existing unit/checkpoint paths
are refused during preparation. Interrupted outputs are preserved for review,
and this adapter does not automatically retry or resume them.

The binding records SHA-256 identities for the trainer's complete static
`TRAINING_CODE` list, configuration, helpers, driver, E05 wrapper, matrix,
splits, rows, acceptance, checker, registry and controls. It records the runtime
binary and installed package versions. Its declared data identity comes from
metadata verification; the trainer verifies NPZ bytes at its own input/output
boundaries, and the driver rehashes all manifest arrays before technical
acceptance. Sources, controls and input identities are checked again before
training and acceptance. The binding never hashes itself or future outputs.

Execution wraps both training and acceptance in E05. Admission closes at
12:10 UTC, cooperative trainer stopping is set to 12:28 UTC and the E05 fixed
deadline is 12:30 UTC on October 20, 2026. The monitor checks storage device
identities, free space and known AC power every 15 seconds. Initial checks also
bind the mounted image to its backing volume. Reserve defaults are 1 GB
internal and 2 GB plus a 1.1 GB checkpoint forecast on both external volumes.
Unknown power, lost AC, a changed device, insufficient space or deadline
failure signals only the wrapper owned by this invocation. E05 then stops its
foreground process group with bounded grace. No timing value extends the
fixed session deadline. This driver does not measure or certify available RAM;
the session coordinator controls overlap with other work.

`WORKER_RESULT.json` contains the actual trainer exit, the unchanged
`accept_fit.accept` technical verdict and compatible fit identity.
`RESULT.json` also requires a successful E05 exit and no resource stop.
Only an accepted full fit contributes one scientific fit. These are explicitly
new standalone receipts, not `n08.fit_receipt.v2` controller receipts.
Existing readout consumers that require the old receipt wrapper need a
separately reviewed adapter; no canonical adoption or 50-fit completion is
claimed here. A Python exit of zero alone never establishes acceptance.

Focused verification uses no models or patient arrays:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p test_corrected_cpu.py -v
```

The checks cover the pinned roster, exact B smoke, full-fit arguments, absolute
data paths, changed source refusal, immutable writes, mount changes, unknown
power, a probe that crosses the fixed deadline, and missing or stale smoke
refusal. They establish adapter
behavior only; the actual smoke and each fit still require their own receipts.
