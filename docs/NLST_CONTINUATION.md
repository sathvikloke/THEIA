# NLST continuation components (D01, version 2)

Status on 2026-10-20: receipt, resource, transport, and archive-verification
components are implemented and fixture-tested. Production acquisition is
**not yet integrated or enabled**. Reading the sealed original downloader and
frozen plan is still blocked by external-volume availability. No actual plan
schema, endpoint construction, or PatientID mapping has been guessed.

The original downloader remains sealed. D01 is partial; D02 has not run through
these components.

## Implemented

`scripts/nlst_resume_core.py` provides:

- A nonblocking exclusive `flock`, held on a stable lock inode.
- Strict JSONL parsing and provenance hashes for input receipt journals.
- Exact agreement between a normalized frozen-plan row and every successful
  receipt for that series; conflicting successes require review.
- A full archive SHA-256 on every reuse, plus size, canonical pathname, and
  stable file identity checks before and after reading. Successor receipts also
  bind the full raw plan row digest and full plan digest.
- Refusal of symlinks, unknown archives, mismatches, missing files associated
  with successes, malformed journals, and unresolved partials. Bytes are
  preserved in place; `review_required` is a logical quarantine decision.
- Atomic, no-clobber publication using a same-volume hard link. It binds the
  previously verified identity, rehashes the linked destination, and keeps the
  partial until validation succeeds; suspect bytes and names remain preserved.
  Preexisting destinations are never replaced.
- Exact partial-name matching for `<UID>.part[.*]` and `<UID>.zip.part[.*]`,
  without confusing a longer UID with a shorter prefix.
- Exclusive JSON creation and durable, append-only JSONL journals.

Legacy receipts do not contain a file-stat identity or an explicit unique-SOP
flag. They are refused by default. The future caller must inspect and verify
the sealed original validator before passing its known digest to
`ReceiptIndex(legacy_verifier_sha256=...)`. Only then may successful legacy
verification evidence be carried forward. The current file identity is bound
at reconciliation, and that fact is explicit in the successor receipt. This
does not claim a historical inode was recorded when it was not.

`scripts/nlst_resources.py` provides:

- Explicit UTC cutoff for starting work and a later deadline for active work.
- Required measured reserve bytes for both THEIA and its Research backing
  volume, accounting for simultaneous transfer reservations.
- Actual mounted external-volume UUID/device checks, and association of THEIA
  with the supplied backing image through `hdiutil info -plist`.
- AC-power checks and termination on deadline, backing-capacity exhaustion,
  device changes, power loss, or a global terminal stop.
- Stop/time rechecks after resource probes, with power-probe timeouts bounded by
  the remaining session/admission window. Deadline and resource failures latch
  a terminal session stop. Admission cutoff alone leaves active transfers free
  to finish before the deadline.

Resource fixtures replace host probes with synthetic observations. They do
not certify current host storage or power. The production resource probe is
macOS-specific and has not been exercised against the currently unavailable
external drives.

`scripts/nlst_transfer.py` provides:

- An explicit `ArchiveExpectation` for source PatientID, SeriesInstanceUID,
  image count and total uncompressed member bytes. Any non-DICOM file must
  appear in an exact member-name allowlist supplied by the eventual adapter.
- Full archive SHA-256, stable file identity, all-file ZIP CRC checks, exact
  file/image and uncompressed-byte checks, all-DICOM PatientID/SeriesInstanceUID
  checks, and nonempty unique SOPInstanceUID checks. It reads headers with
  `stop_before_pixels=True` and never extracts files or decodes pixel arrays.
- Explicit HTTPS URLs supplied by the adapter; no redirect following,
  credentials, endpoint guessing, or hidden library retries.
- At most three attempts for transport failures and HTTP 500/502/503/504.
  HTTP 401/403/429 stop admission globally and are never retried. Other HTTP
  failures and verification failures are terminal for their row.
- Exclusive UUID-named partials, compressed-byte bounds, retention of failed
  partials, and publication only after successful verification.
- Two transfer worker slots. Rows enter those slots in supplied order; the two
  simultaneous requests may reach the server in either order. No later rows are
  admitted after cutoff or global stop. Socket deadlines also interrupt
  continuously arriving partial HTTP headers.

The adapter must validate what the original plan's byte count represents before
constructing this explicit expectation. Runtime URL/identity records remain
local. The transport accepts plaintext HTTP only through an explicit loopback
fixture option, never for public hosts.

`scripts/nlst_continue.py check-bindings` is an executable **refusal interface**.
It checks explicit session timestamps and the recorded original/plan hashes,
then still exits 2 with `acquisition_enabled: false` until the actual source
adapter has been integrated. It does not parse plan rows or issue network
requests. This is intentionally not a download launch command.

## Remaining integration

1. Read and verify the actual `download_nlst.py` and `NLST_DOWNLOAD_PLAN.json`
   against their recorded sealed digests. Inspect the actual plan schema,
   nested subset membership, source URL construction, identity mapping,
   archive layout, and original full-header/CRC/image-count/unique-SOP checks.
2. Extend `scripts/nlst_continue.py` with actual schema bindings and the explicit
   session/resource arguments. Preserve frozen roster order and nested
   P1k/P3k/P10k membership. Do not relabel acquisition as decoding or molecular
   eligibility.
3. Bind the original lock location, archive directory, and all previous terminal
   receipt journals. Save a complete reconciliation manifest, including
   unrecognized artifacts and review-required rows, outside the source tree.
4. Compare the actual original archive/header implementation with the tested
   primitives and preserve any additional source-specific identity checks.
   Connect journals, the two-worker scheduler, resource guard and reconciliation.
5. Verify actual host UUIDs, backing image, free bytes, power, and dependencies;
   exercise the complete integrated CLI once its genuine input contract exists.
6. Produce a concrete launch command only after those bindings and checks
   exist. Reconciliation of the legacy 118.4 GB archive set must actually read
   and hash its bytes; the receipt mirror alone cannot prove present-day files.

## Focused checks

Run from the repository root:

```sh
python3 -m unittest discover -s tests -p test_nlst_resume.py -v
```

The command above checks receipt and resource primitives with standard-library
fixtures. For the complete suite, use an environment with the repository's
pinned `pydicom==3.0.2`:

```sh
python3 -m unittest discover -s tests -p 'test_nlst_*.py' -v
```

The complete suite currently has 60 cases. It uses tiny synthetic DICOM
archives, a loopback HTTP server, and mocked resource probes. It does not
download public imaging, access patient records, or decode pixels. If pydicom
is absent, the real DICOM/transfer fixture classes are explicitly skipped;
such a run must not be reported as complete verification. Raw receipts,
archives, participant mappings, and real plan rows remain outside public source.
