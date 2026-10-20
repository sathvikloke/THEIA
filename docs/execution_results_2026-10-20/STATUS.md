# CPU execution on 20 October 2026

Checkpoint: 09:32 UTC. The authorized session ends at 07:30 America/Chicago (12:30 UTC), with lengthy-job admission closing at 12:10 UTC. GPU and HPC work are excluded. The task register remains [the October 15 backlog](../../planning/execution_backlog_2026-10-15/TASKS.md).

| Task | Actual result | Remaining work |
|---|---|---|
| P01 | Historical cohort-flow SVG/PNG and source tables regenerated and mechanically/visually checked. | Author adoption; this is not a new corrected replication. |
| E01/E03 | Actual trainer import closure, acceptance helpers and frozen matrix exported; CLI import passed in the existing runtime. | Clean-environment reproduction and private runtime/data inputs. |
| E05 | Bounded attempt runner implemented and reviewed; a real invocation recorded processes and stopped its training group on a storage-reserve failure. | Successful real training and acceptance integration. |
| M01 | Reconciled and reused the October 16 real acceptance: 38/38 checks passed, fixtures false; retained A15 memberships unchanged. Current small-file bindings match. | No duplicate acceptance run was needed; no fit follows merely from source acceptance. |
| M02 | Direct CPU smoke/fit driver published; 15 focused tests passed. Full fits require a matching accepted real smoke. | Complete compatibility with readout consumers that expect historical controller receipts. No historical release receipts are fabricated. |
| M03 | Real invocation verified all 158 bound NPZ files, then the resource monitor stopped it at 09:24:14 UTC after internal free storage fell below reserve. Logs and partials retained. | No accepted two-epoch smoke; another execution requires restored resources and explicit review of the interruption. |
| M04/M05 | No newly accepted corrected scientific fit. | Accepted smoke, then the fixed 50-fit matrix. |
| D01/D02 | Actual sealed downloader and 10,000-row frozen plan bound to receipt-aware continuation. Independent review passed after terminal-stop repairs; 82 checks plus unchanged adversarial probes passed. Actual resource preflight refused admission. | Actual archive reconciliation and new acquisition; no new NLST archive this session. |
| D05 | Official CT/XML inventory reconciled: 1,010 people, 1,018 CT series; 883 with included polygons and 135 without. Duplicates/source correction and remaining source exceptions recorded. | Remaining CT acquisition and source exceptions; inventory is not an adopted cohort. |
| D06 | One real 133-object CT/XML pilot passed native consensus export, row construction and actual Torch dataset/collate. Technical crop/ROI QA passed. | Cohort-scale preparation and source coverage. One unlabeled row is not an EGFR-evaluable patient. |

Research and THEIA mounted and passed identity/write/readback checks at 08:53 UTC. Internal free storage later fell to approximately 330 MB; the real training monitor stopped its owned work and NLST preflight refused admission. No further reserve reduction is planned. Small metadata reconciliation continues; new training/download admission awaits adequate actual headroom.

Actual new outputs at this checkpoint: **one LIDC CT archive, one loadable unlabeled row, zero new NLST archives, zero accepted corrected smokes, zero accepted corrected full fits, and zero newly independently EGFR-evaluable patients**. Prior valid outputs remain separate and are reused when compatible. There is no defensible whole-project completion percentage from these task counts.

## Reviewable artifacts

- [LIDC real pilot and aggregate inventory](lidc_pilot/README.md)
- [Corrected Figure 1](figure1/README.md)
- [Corrected CPU execution](../CORRECTED_CPU.md)
- [Standalone attempt commands](../CPU_ATTEMPTS.md)
- [NLST continuation implementation and limits](../NLST_CONTINUATION.md)
- [LIDC mask-to-row adapter](../LIDC_ROWS.md)

Private receipts, source identities, patient data, images and correspondence remain outside GitHub. Tests and adapter implementation are reported separately from real data outputs and accepted scientific fits.
