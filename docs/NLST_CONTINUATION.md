# Receipt-aware NLST continuation (D01, version 2)

The successor is bound to the actual sealed 2026-10-14 downloader and selected
10,000-person NLST plan. It never edits those sources. Source hashes, full plan
rows, frozen order, unique patient/person/series identities, nested P1k/P3k flags,
role, license, and byte totals are checked before a session starts. The original
script's literal ROOT, PLAN, DEST, and INTERNAL paths are parsed without executing
it. All archive and partial bytes remain on verified external storage.

## Commands and scope

- `check-bindings` checks source/plan pins and the explicit UTC window only.
- `preflight` holds the original `INTERNAL/download.lock`, checks actual resources,
  reads canonical receipts, inventories every frozen row, and writes a new session
  manifest. It performs no archive hashing or HTTP request.
- `reconcile` additionally hashes existing receipt-matched archives. It never
  downloads. `--max-reuse-archives` bounds this independent work.
- `run` additionally admits genuinely absent rows in frozen relative order with
  two HTTP workers. One optional reconciliation worker runs concurrently; the
  missing-row queue does not wait for all legacy archive hashes. Setting
  `--max-reuse-archives 0` defers all legacy hashing. `--max-new-archives` bounds
  newly admitted rows. Limits do not imply completion of the full roster.

All commands require `--original-downloader`, `--plan`, `--cutoff-utc`, and
`--deadline-utc`. Other commands also require `--research-root`, `--backing-image`,
`--theia-uuid`, `--research-uuid`, `--reserve-theia-bytes`,
`--reserve-research-bytes`, `--reserve-internal-bytes`, and `--session-dir`.
The session directory must be new, under an existing external THEIA parent.
`--mirror-root` adds small private receipts and summaries on the guarded internal
volume. `--receipt` may add explicit earlier journals. Prior successor transfer
and reconciliation journals under sibling session directories are discovered
alongside both original canonical receipt copies.

Production `run` requires the existing pinned `pydicom==3.0.2`; it installs nothing.
Use `python3 scripts/nlst_continue.py COMMAND --help` for the full argument list.
Actual session command lines and measured resource identities belong in private
execution receipts, not this public source document.

## Resume and preservation guarantees

Existing bytes are initially `existing_unverified`, never counted as accepted.
Reuse requires every successful receipt to match the normalized frozen row,
canonical path, size and digest; version-2 receipts additionally bind the full
plan and raw-row hashes and previous device/inode/mtime/ctime identity. Each reuse
rereads the entire archive SHA-256 and checks stable identity before and after.
Conflicting, malformed, missing-success, unknown, or mismatched files require
review and are never silently redownloaded or overwritten.

Legacy receipts lack a historical stat identity and explicit unique-SOP flag.
They are permitted only after checking the sealed predecessor hash, whose actual
validator checks every DICOM PatientID, StudyInstanceUID, SeriesInstanceUID and
unique SOPInstanceUID. Reconciliation explicitly states when the current file
identity is first bound; it does not invent historical identity evidence.

Partials from both `<UID>.zip.attemptNN.part` and successor
`<UID>.zip.part.*` forms are preserved. Exact UID matching prevents a longer UID
from blocking an unrelated shorter UID. A missing row with a known partial is
held for review. Failed new attempts use distinct exclusive names and remain on
disk. Publication uses a no-clobber hard link bound to verified identity, rehashes
the linked bytes, and only then removes its owned partial name. Mutation or
publication races preserve suspect names and emit no success.

## Genuine source and archive contract

The public source is the original NLST NBIA `getImage` HTTPS endpoint with the
frozen `SeriesInstanceUID` query parameter. The plan's `expected_bytes` is the
sum of uncompressed DICOM-member bytes, excluding the exact `LICENSE` member.
The response bound is the original `2 * expected_bytes + 1,000,000` bytes.

Verification checks full SHA-256, safe ZIP paths, all-member CRC, exact DICOM
count/bytes, bounded LICENSE bytes, every DICOM PatientID/StudyInstanceUID/
SeriesInstanceUID, and nonempty unique SOPInstanceUIDs. Headers are read with
`stop_before_pixels=True`; no files are extracted and no pixel arrays decoded.
Unrecognized non-DICOM member names are refused.

HTTP requests do not follow redirects. Transport failures and HTTP
500/502/503/504 receive at most three attempts. HTTP 401/403/429 immediately stop
admission globally and are never retried. Verification and other HTTP errors are
terminal for their row and stop further queue admission globally. Exhausted
transport retries and newly appearing conflicting bytes also stop admission.
Terminal stop is latched before filesystem-stat and journal writes can block.
A reconciliation mismatch also stops admission before its journal is written.
Already-owned partials remain preserved. A session with unresolved row errors or abnormal resource
stops exits nonzero; its remaining queue and failure receipts remain visible.

## Resource and timing controls

The guard verifies mounted THEIA and Research UUID/device identities and the
actual backing-image association. It requires AC power and checks external free
space against explicit reserves plus both active transfer reservations. Internal
free space is checked against its separate small-journal reserve. Deadline and
resource failures latch a global stop. Admission cutoff only stops new work;
already admitted work may finish before the absolute deadline. Resource probes
recheck time and stop state before returning; socket shutdown also bounds peers
that continuously drip response headers. Regular file I/O is cooperative and
cannot guarantee immediate cancellation of an unresponsive kernel/device call.

`RUN.json`, `INVENTORY.json`, durable append-only transfer/reconciliation JSONL,
`STATE.json`, and `SUMMARY.json` preserve the session's input/source identities,
resources, outcomes, and exact remaining queue. Full private inventory stays
external. Journal writes and finalization attempt the internal mirror even when
external persistence fails. No acquisition result grants molecular eligibility
or scientific acceptance; all summaries explicitly report zero new evaluable
EGFR patients and no pixel decoding.

## Focused verification

From the repository root, with pinned pydicom already available:

```sh
python3 -m unittest discover -s tests -p 'test_nlst_*.py' -v
```

The 82 focused cases use tiny synthetic DICOM archives, loopback HTTP, and mocked
resource observations. They cover receipt identity, publication mutation windows,
original partial forms, real-schema binding, study identity and LICENSE CRC/byte
scope, simultaneous missing-row admission and legacy hashing, mirrored failure
evidence, deadlines, retries, resource stops and the two-worker bound. Missing
pydicom explicitly skips DICOM fixture classes; such a run is not complete
verification. Fixture success alone does not authorize production acquisition;
actual plan/source and current host resource preflight are separately recorded.
