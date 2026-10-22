# Corrected M1 CPU execution — session-portable v2 (M02/E05)

`scripts/run_corrected_cpu.py` (v1) compiled the 2026-10-20 admission cutoff,
cooperative stop and deadline into the reviewed source, so it cannot admit work
in any later session. v1, its receipts and its expired window are preserved
unchanged as historical evidence.

`scripts/run_corrected_cpu_v2.py` is a versioned successor derived from v1 by
asserted textual edits. It changes operational timing and reserve plumbing only:

- `prepare` requires `--session-window FILE`, a JSON object (or one nested under
  `window`) with explicit UTC `start_utc`, `admission_cutoff_utc`,
  `cooperative_stop_utc` and `deadline_utc`. The window must be ordered and its
  cutoff must still be in the future. The file's path, SHA-256 and values are
  recorded in the new binding and its hash joins the bound input closure, so an
  edited window refuses later execution.
- The trainer `--deadline-utc`, the unit `SESSION.json` and `COOP_CONTROL.json`
  take their values from that recorded window.
- THEIA and backing-volume reserves are distinct (`--theia-reserve-bytes`,
  default 50,000,000,000; `--backing-reserve-bytes`, default 100,000,000,000),
  each plus the checkpoint forecast; the internal reserve is unchanged.
- Bindings use schema `theia.corrected_cpu.execution_binding.v2` and
  `driver_version` `corrected_cpu_v2`. v2 refuses to admit or execute v1
  bindings, and a v2 fit binds only a v2 smoke in the same output root.

Matrix, split, data, initialization, loss, epochs, CPU/offline environment,
source closure, smoke requirement, acceptance replay and receipt shapes
(`result.v1`, `technical_receipt.v1`) are unchanged. Apart from output paths and
the stop time, a v2 smoke trainer command is identical to the reviewed v1 one.

```sh
"$ROOT/.venv/bin/python" scripts/run_corrected_cpu_v2.py prepare \
  --root "$ROOT" --output-root "$ROOT/revamp/cpu_runs/<new-session>/n08" \
  --kind smoke --session-window "$ROOT/revamp/cpu_runs/<new-session>/SESSION_WINDOW.json"
```

Changing a future session's window is not permission to extend an earlier job:
existing units are never re-prepared, and an expired cutoff refuses admission.

## Native consumer

`execution_support/n08/standalone_receipts_v2.py` is the matching native fit
verifier, derived from the reviewed v1 verifier the same way. It pins the v2
driver, requires v2 bindings, re-reads the hash-bound window file, checks the
E05 session, chronology, trainer stop and cooperative-control contract against
that window, and emits `theia.corrected_cpu.consumption.v2`.

## Verification

```sh
python3 -m unittest tests.test_corrected_cpu_v2 tests.test_corrected_cpu
TMPDIR=<external dir> python3 -m unittest tests.test_standalone_receipts_v2
```

The first command ran 26 cases (11 new window-boundary cases plus the unchanged
v1 driver cases). The derived verifier suite ran 53 cases: the v1 synthetic cases
re-expressed against v2 bindings plus 5 window-specific refusals. These are
synthetic fixtures with no arrays, models or trainer jobs. Test success is not a
smoke, a fit, or scientific acceptance.

## Not yet ported

`run_maps_standalone_v1.py` and the native readout still bind the v1 window and
verifier. They need equivalent v2 successors before maps/readout can consume v2
fits. No v2 smoke or fit has executed yet.
