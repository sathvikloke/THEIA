# CPU execution on 20 October 2026

Checkpoint: 11:52 UTC. The authorized session ends at 07:30 America/Chicago (12:30 UTC), with lengthy-job admission closing at 12:10 UTC. GPU and HPC work are excluded. The task register remains [the October 15 backlog](../../planning/execution_backlog_2026-10-15/TASKS.md).

| Task | Actual result | Remaining work |
|---|---|---|
| P01 | Historical cohort-flow SVG/PNG and source tables regenerated and mechanically/visually checked. | Author adoption; this is not a new corrected replication. |
| E01/E03 | Actual trainer import closure, acceptance helpers and frozen matrix exported; CLI import passed in the existing runtime. | Clean-environment reproduction and private runtime/data inputs. |
| E05 | Bounded attempt runner implemented and reviewed; a real invocation recorded processes and stopped its training group on a storage-reserve failure. | Successful real training and acceptance integration. |
| M01 | Reconciled and reused the October 16 real acceptance: 38/38 checks passed, fixtures false; retained A15 memberships unchanged. Current small-file bindings match. | No duplicate acceptance run was needed; no fit follows merely from source acceptance. |
| M02 | Direct CPU smoke/fit driver published; 15 focused tests passed. Full fits require a matching accepted real smoke. | Native fit-receipt verification and a versioned readout entry have been reviewed and published. The readout CLI still refuses a scientific endpoint pending maps integration; the native five-fold maps adapter has passed scoped independent review after reserve and launch-cutoff repairs. No real maps have run. No historical release receipts are fabricated. |
| M03 | Real invocation verified all 158 bound NPZ files, then the resource monitor stopped it at 09:24:14 UTC after internal free storage fell below reserve. Logs and partials retained. | No accepted two-epoch smoke; another execution requires restored resources and explicit review of the interruption. |
| M04/M05 | No newly accepted corrected scientific fit. | Accepted smoke, then the fixed 50-fit matrix. |
| D01/D02 | Actual sealed downloader and 10,000-row frozen plan bound to receipt-aware continuation. Independent review passed after terminal-stop repairs; 82 checks plus unchanged adversarial probes passed. One historical archive was fully reverified separately; a later genuine acquisition verified eight new archives (435,749,716 compressed bytes; 1,468 DICOM objects), then stopped at the internal storage reserve. | Remaining acquisition and historical archive verification. One received partial is retained; a second transfer stopped before receiving bytes. No automatic restart. |
| D05 | Official CT/XML inventory reconciled: 1,010 people, 1,018 CT series; 883 with included polygons and 135 without. Duplicates/source correction and remaining source exceptions recorded. | Remaining CT acquisition and source exceptions; inventory is not an adopted cohort. |
| D06 | One real 133-object CT/XML pilot passed native consensus export, row construction and actual Torch dataset/collate. Technical crop/ROI QA passed. | Cohort-scale preparation and source coverage. One unlabeled row is not an EGFR-evaluable patient. |
| D09/D16 | Exact 19-case source/attrition assessment completed; one missing reference CT archive restored with 274 verified DICOM objects. Header geometry checks passed. | The guarded attempt extracted and verified all 274 objects, then stopped at the internal storage reserve after 16.4 seconds of builder execution. No candidate NPZ or technical gate outcome was produced; no frozen M1 membership change or clinical acceptance. |
| D08 | Reconciled 421 local imports and 420 mask-row records; all 420 referenced NPZ filenames exist without reading payloads. | The advertised 422-person roster remains unbound; historical transform, partition and checkpoint lineage remain unresolved. |
| D13 | One bounded metadata comparison worker timed out without a durable comparison ledger. | All 15 new cross-collection overlap cells remain unknown; no new person-disjointness or exposure result. |
| D10/D11 | Existing CT and molecular-file availability reconciled; 36 prior public WXS files are present at expected sizes. | Protected molecular decoding/linkage remains separate; availability is not a verified EGFR outcome. |

Research and THEIA mounted and passed identity/write/readback checks at 08:53 UTC. Internal free storage fluctuates below 1 GB. Both the real training attempt and later NLST acquisition stopped when the 500 MB reserve was unavailable. No further reserve reduction is planned. The later candidate builder also stopped at that reserve. Small metadata work and source implementation continue; fresh model admission requires actual headroom. The candidate job deadline has expired and is not extended.

Actual new outputs at this checkpoint: **eight NLST CT archives, one LIDC CT archive, one restored reference CT archive, one loadable unlabeled LIDC row, zero accepted corrected smokes, zero accepted corrected full fits, and zero newly independently EGFR-evaluable patients**. The eight NLST archives correspond to eight frozen roster people for imaging use, without verified EGFR outcomes. The separately reverified historical archive is not a new download. Prior valid outputs remain separate and are reused when compatible. There is no defensible whole-project completion percentage from these task counts.

## Reviewable artifacts

- [Restored reference CT and technical limits](source_restoration/README.md)
- [Internal source inventory](internal_case_inventory/README.md)
- [LIDC real pilot and aggregate inventory](lidc_pilot/README.md)
- [Corrected Figure 1](figure1/README.md)
- [Native map receipt adapter and limits](../CORRECTED_CPU_MAPS.md)
- [Native readout entry and limits](../CORRECTED_CPU_READOUT.md)
- [Corrected CPU execution](../CORRECTED_CPU.md)
- [Standalone attempt commands](../CPU_ATTEMPTS.md)
- [NLST continuation implementation and limits](../NLST_CONTINUATION.md)
- [LIDC mask-to-row adapter](../LIDC_ROWS.md)

Private receipts, source identities, patient data, images and correspondence remain outside GitHub. Tests and adapter implementation are reported separately from real data outputs and accepted scientific fits.
