# Standalone nine-hour execution runbook

This runbook is for a future session that Sathvik actually starts. It creates no automation and launches no job by itself. Claude owns execution; Codex need not remain online.

## Session contract

Record UTC start, absolute end=start+9h, admission cutoff=end−20min, selected task IDs, checkout commit/dirty patch hash, interpreter/environment, hardware and output root. A wait, quota reset, crash or retry never extends that window. If the session has different explicit timing, record it rather than silently adopting an old October 14 deadline.

Use an append-only session directory outside tracked source, for example `output/execution_sessions/<UTC-start>/`. Record one attempt directory per task/run/fold. Bind all paths through a local `paths.json`; the GitHub repository must not contain a person's absolute data paths or private keys. Each task has one writer. Parallel independent CPU/I/O tasks may proceed within measured memory/storage capacity; use one heavy accelerator job until coexistence is demonstrated. Do not idle ready work for an hourly review boundary.

## Execution order

1. Read `TASKS.json`, the last session's summary and only changed evidence. Determine which outputs already match the current inputs/methods. Do not rerun unchanged stability, C22, RIDER or other historical suites merely to raise task counts.
2. Select ready tasks in priority order. E01/E02 establish provenance and portability; they are not a general platform campaign. Release a dependent task as soon as its exact inputs are closed. A missing external cohort or reader blocks only the tasks that use it.
3. Before a long job, check actual mounted output storage, free space at every backing volume, power, memory and accelerator availability. Compute the required reserve from the output/workspace budget. The October 14 internal free-space observation is not today's gate. Do not relocate active sources, delete unrelated projects or invent mount directories.
4. Execute the recorded command with unbuffered logs and captured exit status. Save predictions/checkpoints and technical diagnostics before deriving summaries. Preserve failed outputs and rejected attempts. Confirm resume compatibility by config/data/checkpoint fingerprints, not just a filename.
5. For an approved fixed analysis, Claude may run its existing statistical code and produce tables/figures. The humans retain selection of endpoints, new thresholds/eligibility rules, scientific interpretation and acceptance. A missing prespecification becomes one precise item in `HUMAN_REVIEW.md`; continue unrelated ready tasks.
6. Do not start a new long attempt after the admission cutoff. Checkpoint cooperative jobs before the deadline. Reconcile in-flight transfers and partial files; stop only positively identified processes owned by this session. Preserve outputs even if incomplete.
7. Write a compact return: task IDs attempted/completed, actual run/fold/subject counts, output paths/hashes, failures, remaining dependencies, measured runtimes and the next ready task. Commit safe code and aggregate reports to the execution branch. Never equate successful subprocess exit with scientific acceptance.

## Required return artifacts

- `session.json`: immutable timing, scope, hardware, code and input versions.
- `attempts.jsonl`: task ID, attempt ID, command, start/end, exit code, failure reason, resume checkpoint, output-manifest hash.
- `outputs_manifest.json`: sizes and SHA256 for task outputs; sensitive artifacts are referenced locally, not copied to GitHub.
- `SUMMARY.md`: actual completed experiments/data, exact remaining work, and next-session commands.
- `DECISIONS_NEEDED.md`: only decisions genuinely required to proceed on a particular task; named artifact, reason, options and unaffected work.

## Local and HPC execution

The existing training code has resume/fold/deadline facilities in the local audited checkout. Inspect its actual help after E01; old shell comments claiming otherwise are stale. Use task-specific configs, output roots and fold selection. Do not launch `run_full.py` or an old master prompt as a substitute for the current run matrix.

E06 prepares an HPC job wrapper only for an available authorized cluster. Its job array must preserve the same run/fold/config/input IDs as the local route. Record the scheduler job ID and per-job exit state. Do not assume a cluster hostname, account, partition, walltime entitlement or paid capacity. Lack of HPC does not block acquisition, saved-data work, or a measured local route.

Provider quotas pause the model interaction, not justify new accounts, purchases or repeated probes. Existing deterministic jobs may continue only inside the recorded session window. Do not impose a new arbitrary dollar, launch or turn cap; do stop for actual provider limits and resource failures.

## Prompt to give Claude

> Work through `planning/execution_backlog_2026-10-15/START_HERE.md` and `TASKS.json` for this explicitly authorized nine-hour session. Own the execution backlog with Sathvik; there are no teammate assignments. Start by recording the actual session timing and current code/input identities. Complete ready computational tasks, release dependencies immediately, and continue independent work when a particular task needs data or a human decision. Reuse verified existing results. Produce raw outputs, technical checks and reproducible aggregate tables/figures; leave scientific interpretation, acceptance and manuscript prose to the human research team. Do not revive the old Codex controller. Use the runbook's checkpoint and return contract, keep sensitive data local, and stop by the fixed recorded deadline.
