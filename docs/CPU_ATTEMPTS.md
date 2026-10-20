# Standalone CPU attempts

`scripts/run_cpu_task.py` runs one foreground command without a model controller
or scheduler. It adds immutable attempt receipts, exclusive ownership of a
run/fold key, bounded stopping and file identity checks. The actual training
program still owns its checkpoint, RNG, optimizer and data-compatibility logic.
This wrapper does not certify a scientific result.

Create an existing output directory on the intended mounted volume. Do not
create a replacement `/Volumes/...` directory if that volume is absent. A
session JSON needs timezone-qualified absolute timestamps:

```json
{
  "admission_cutoff_utc": "2026-10-20T12:10:00Z",
  "deadline_utc": "2026-10-20T12:30:00Z"
}
```

These are the October 20 session times, not defaults for later sessions. Expired
sessions are refused. Input hashing, retries or process interruptions never
extend them.

```sh
python scripts/run_cpu_task.py \
  --session /actual/session.json \
  --attempt-root /actual/existing/attempts \
  --cwd /actual/repository \
  --task-id M04 --job-key cellA-init42-part1337-fold0 \
  --source /actual/repository/theia/engine/train.py \
  --source /actual/dependency-lock.txt \
  --input /actual/bound-config.json \
  --input /actual/content-manifest.json \
  -- /actual/environment/bin/python -m theia.engine.train <verified-job-arguments>
```

Replace placeholders with the task's actual reviewed command. Include the
task's full source/config/input identity manifests; a hash of one Python entry
point does not bind all its imports. Credentials should not appear in command
arguments or published receipts. The wrapper inherits the current environment
and does not hash or disclose its environment variables.

Every attempt gets separate `START.json`, `PROCESS.json`, stdout/stderr logs and
`EXIT.json`. Failure and interruption remain visible even when a child exits
zero after saving a checkpoint. Optional `--output` arguments hash specific
closed files; a missing declared output causes a nonzero wrapper exit. Files
are neither moved nor removed. Receipts may contain local paths and belong in
local storage; only reviewed aggregate summaries belong in the public repo.

For resumption, retain the job's own resume option in the literal command and
pass `--resume-receipt /prior/START.json`. This requires identical command,
working directory, source hashes and input hashes. Each attempt has a new
directory. The job must also verify the checkpoint's internal fingerprint.
A renewed user-authorized session may supply a new timing file; the old timing
file and receipts stay unchanged.

The default stopping sequence begins with SIGINT ten seconds before the fixed
deadline, then sends SIGTERM near the deadline and SIGKILL if necessary. Jobs
must handle SIGINT to write a useful checkpoint. The wrapper uses its own child
process group, not stale PID files. Detached descendants that deliberately leave
that process group are unsupported: run a foreground program, not a daemon.
The wrapper also stops remaining in-group descendants if the foreground exits.

Before launching data work, separately verify actual mounted storage, backing
volume capacity, available memory and power for that job. This small adapter
does not select a CPU budget or infer that an unavailable dataset can be used.

Focused verification:

```sh
python -m unittest discover -s tests -p test_cpu_task_runner.py -v
```

The tests execute tiny subprocesses for failure, expired admission, interruption,
checkpoint resumption, source-mismatch refusal and missing output. They are
plumbing checks and contribute no accepted fits or patient results.
