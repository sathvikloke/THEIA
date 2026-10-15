# Detailed execution task cards

Owner for every task: **Sathvik + Claude**. No teammate allocations. Status starts `pending_reconciliation`: newer compatible evidence can close an item by reuse. Core means the documented corrected study/output path, not a promise that every item is mandatory for every possible journal claim. `claim_conditional`, `data_expansion`, `alternative_route`, `external_access` and `optional_research` are explicit scope boundaries.

Dependencies list task IDs. Input requirements can release partial work earlier (for example maps after a cell/seed closes; acquisition/preprocessing by closed tranche). Human dependencies refer to [HUMAN_REVIEW.md](HUMAN_REVIEW.md); an existing valid decision satisfies them. They never require another Codex controller.

Commands use bound local paths, not literal placeholder filenames. Every implementation task says when no ready CLI exists. Validated existing flag shapes and source aliases are in [COMPUTE_SOURCE_AUDIT.md](COMPUTE_SOURCE_AUDIT.md). Source-specific acquisition details are in [DATASETS.md](DATASETS.md).

## Index

| ID | Task | Scope | Dependencies |
|---|---|---|---|
| [E01](#e01) | Publish the portable execution source, without importing the dirty checkout wholesale | core | None |
| [E02](#e02) | Reconcile all selected results and newer Claude returns before assigning unfinished work | core | None |
| [E03](#e03) | Make a fresh environment and one targeted workflow smoke reproducible | core | E01 |
| [E04](#e04) | Measure storage, memory and runtime capacity for the next real units | core | None |
| [E05](#e05) | Provide standalone checkpoint/return handling for nightly sessions | core | E01, E03 |
| [E06](#e06) | Prepare the same run matrix for an available HPC route | alternative_route | E01, E03, E04 |
| [D01](#d01) | Implement receipt-aware NLST continuation | core | E01 |
| [D02](#d02) | Finish the remaining frozen P10k imaging archive acquisition | data_expansion | D01, E04 |
| [D03](#d03) | Profile NLST decoding and preprocessing on a bounded tranche | data_expansion | E01, E03, E04 |
| [D04](#d04) | Process acquired NLST tranches for the selected pretraining use | data_expansion | D03 |
| [D05](#d05) | Complete the LIDC-IDRI source inventory and permitted missing acquisition | data_expansion | E01, E04 |
| [D06](#d06) | Connect LIDC consensus masks to usable training rows | data_expansion | E01, D05 |
| [D07](#d07) | Build the LIDC pretraining cohort under the selected consensus method | data_expansion | D06, E04 |
| [D08](#d08) | Reconcile and preprocess NSCLC-Radiomics for its documented role | data_expansion | E01, E04 |
| [D09](#d09) | Close the corrected NSCLC-Radiogenomics source/attrition inventory | core | E02 |
| [D10](#d10) | Reconcile TCGA-LUAD external CT and existing label metadata | core | E01, E02 |
| [D11](#d11) | Complete CPTAC-LUAD CT and molecular file availability inventory | core | E01, E02 |
| [D12](#d12) | Decode and link molecular outcomes only in the recorded exposure sequence | core | D11 |
| [D13](#d13) | Establish person-level overlap and split/exposure accounting across cohorts | core | D09, D10, D11 |
| [D14](#d14) | Build the acquisition funnel for 1,000 paired CT–EGFR persons and the separate 10,000 aspiration | core | E02 |
| [D15](#d15) | Prepare and run incoming institutional-cohort ingestion when access exists | external_access | E01, D14 |
| [D16](#d16) | Assess the 19 labelled but unbuilt internal development cases | data_expansion | E02, D09 |
| [D17](#d17) | Build genuinely reader-resolved external input arrays and the final analysis join | core | R03, D10, D11, D13 |
| [D18](#d18) | Materialize a selected human-annotation robustness subset | optional_research | E01, E02, D13 |
| [D19](#d19) | Resolve every optional public candidate into an executable include/defer/exclude record | data_expansion | E02 |
| [D20](#d20) | Build the separate 250-person unlabeled NLST development cohort | data_expansion | D03, E04 |
| [D21](#d21) | Acquire and verify the held-out NLST same-acquisition reconstruction pairs | data_expansion | D03, E04 |
| [M01](#m01) | Execute real corrected-input S1-a acceptance | core | E01, E03, D09 |
| [M02](#m02) | Bind a direct corrected-fit driver to the existing acceptance/readout contract | core | E01, E03 |
| [M03](#m03) | Run the real corrected loader/model/loss smoke on target hardware | core | M01, M02, E04 |
| [M04](#m04) | Complete the first corrected scientific fit | core | M03 |
| [M05](#m05) | Complete the remaining fixed corrected 50-fit matrix | core | M04 |
| [M06](#m06) | Produce corrected best/last maps and reload controls | core | M03 |
| [M07](#m07) | Materialize the prespecified corrected replication readout | core | M05 |
| [M08](#m08) | Execute the prepared evaluation-only fragment and repaired-case sensitivities | core | M06 |
| [M09](#m09) | Build matched corrected native-feature and non-THEIA comparators | claim_conditional | M01, E03 |
| [M10](#m10) | Compute corrected calibration, proper losses and incremental-value tables | claim_conditional | M07, M09 |
| [M11](#m11) | Run locked-model external prediction and prespecified external endpoints | core | E03, D10, D13 |
| [N01](#n01) | Close the remaining NLST adaptation settings from the existing design | expanded_study | E02 |
| [N02](#n02) | Implement the scientific NLST adaptation runner using existing primitives | expanded_study | E01, E03, N01 |
| [N03](#n03) | Integrate the adapted-vision transfer hook into downstream training | expanded_study | E01, E03, N01 |
| [N04](#n04) | Execute the six independent NLST scale/seed adaptation runs | expanded_study | N02, E04, N09 |
| [N05](#n05) | Execute the 30 matched downstream adaptation-transfer fits | expanded_study | N03, M01 |
| [N06](#n06) | Implement pair-aware sampling and the held-out reconstruction evaluator | expanded_study | E01, E03, N01 |
| [N07](#n07) | Run frozen-encoder and adapted-encoder reconstruction robustness | expanded_study | N06, D21 |
| [N08](#n08) | Compute the declared scale/transfer readout and close the expanded program | expanded_study | N05 |
| [N09](#n09) | Run real-derivative adaptation smoke and measure actual update cost | expanded_study | N02, D03 |
| [A01](#a01) | Reproduce only missing or changed historical all-fold evidence outputs | core | E02, E03 |
| [A02](#a02) | Evaluate accepted corrected models on retained RIDER test–retest inputs | claim_conditional | M03, E02 |
| [R01](#r01) | Materialize the prepared reader packet and perform technical viewer QA | core | E01, E02 |
| [R02](#r02) | Finish the native-compatible custodian inference package | core | E01, E03 |
| [R03](#r03) | Ingest genuine reader returns with provenance | external_access | R01 |
| [R04](#r04) | Compute the prespecified reader endpoints from genuine completed reads | external_access | R03 |
| [R05](#r05) | Ingest custodian model-output returns independently of reader packets | external_access | R02 |
| [P01](#p01) | Regenerate Figure1 image from the corrected accepted flow | core | E02 |
| [P02](#p02) | Finalize mechanical C06 layout and labeling without recomputing it | core | E02 |
| [P03](#p03) | Assemble the final selected table/figure data and value traceability | core | E02, A01 |
| [P04](#p04) | Run a source-preserving reproduction of selected publication outputs | core | E01, E03, P03 |
| [P05](#p05) | Typeset and mechanically inspect the human-approved submission files | core | P03 |
| [P06](#p06) | Create the final technical release/reviewer bundle | core | P04, P05, P07 |
| [P07](#p07) | Populate technical reporting and IJAS evidence attachments | core | E02, P03 |
| [X01](#x01) | Validate predictive/spatial null procedures if retaining the familywise claim | optional_research | E01, E02, E03 |
| [X02](#x02) | Transfer selected mechanism/intervention analyses to corrected models | optional_research | E01, E02, E03, M07 |
| [X03](#x03) | Implement and run wrong-mask training controls if selected | optional_research | E01, E02, E03, M01 |
| [X04](#x04) | Implement annotation-efficiency subsets if selected | optional_research | E01, E02, E03, M01 |
| [X05](#x05) | Run clipped grounding-pretraining transfer comparisons if needed | optional_research | E01, E02, E03 |
| [X06](#x06) | Run additional architecture/frozen-versus-finetuned comparator if selected | optional_research | E01, E02, E03, M01 |
| [X07](#x07) | Implement within-site training-label permutation if selected | optional_research | E01, E02, E03, M01 |
| [X08](#x08) | Develop geometry-aware3D or CT–pathology branch if selected | optional_research | E01, E02, E03 |
| [X09](#x09) | Run Tier2 retraining or input-versus-initialization separation if selected | optional_research | E01, E02, E03, M07 |
| [X10](#x10) | Stabilize and evaluate generation/reader-benefit branch if retained | optional_research | E01, E02, E03 |
| [X11](#x11) | Run additional selection/stability study only if scientifically retained | optional_research | E01, E02, E03 |
| [X12](#x12) | Implement retained multigene endpoints and the declared discovery branch | optional_research | E01, E02, D13 |

<a id="e01"></a>

## E01 — Publish the portable execution source, without importing the dirty checkout wholesale

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** None. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Public main d70c091a9fa63d276d8c0fd5d6336136d4517273
- Local HEAD 4fc73abb2cf1466d8830de202076e13f855292c7 and staged/unstaged/untracked source
- COMPUTE_SOURCE_AUDIT.md paths BW, NR, PX, PF and I3

**Work**

1. Inventory the exact trainer, loader, acceptance, split, map/readout, acquisition and reader modules used by this backlog; record local file hashes and their imports.
2. Export those code/config/protocol files and safe fixtures to an execution branch; include the actual 50-row fit matrix and its source-defined split/protocol files. Exclude datasets, labels, private linkage keys, credentials, email drafts and bulky campaign scratch directories.
3. Use explicit staged paths and review the diff. Preserve every original local change. Record any file that cannot be publicly distributed as an access dependency with a local lookup.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- SOURCE_EXPORT_MANIFEST.json
- Portable source/config commits and LOCAL_INPUTS.example.json
- README mapping current entrypoints to their versions

**Technical completion**

- A fresh clone resolves every required import/config/CLI for the selected task.
- No missing local-only dependency is described as already published.
- No patient data or private material appears in staged additions.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="e02"></a>

## E02 — Reconcile all selected results and newer Claude returns before assigning unfinished work

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** None. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- AUDIT.md evidence locator
- Historical N08/C22/N03/RIDER/TCGA results
- Any post-October-14 standalone Claude returns

**Work**

1. Build one task/run/fold/output inventory with input and method versions, actual exits, accepted-unit status and evidence path.
2. For each task below, mark reused_complete, ready, waiting_dependency, waiting_external or needs_method_decision. Separate implemented, smoke-tested, executed and technically accepted.
3. Reconcile source task IDs (I/C/R/D, CMP, WP, acquisition) against this register; add only demonstrated uncovered deliverables.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- EXECUTION_STATUS.json
- RESULT_INDEX.csv
- DATED_MILESTONES.json

**Technical completion**

- All 24 closed light tasks retained as completed within scope.
- Accepted fits counted by unique planned identity; downloads/readers/evaluable people reported separately.
- Newer work overrides old status only with compatible evidence.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="e03"></a>

## E03 — Make a fresh environment and one targeted workflow smoke reproducible

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- requirements.txt
- requirements-lock.txt
- setup.py
- configs/default.yaml
- Selected model/cache identities

**Work**

1. Record Python/platform/dependency versions and model file identities; resolve installed versus pinned differences.
2. Build an isolated environment and exercise the selected data/model interface with a small permitted or synthetic fixture.
3. Run only tests needed for edited paths and genuine previously observed failures; save logs.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- environment.lock or equivalent exact dependency record
- ENVIRONMENT_RECEIPT.json
- Targeted smoke/test logs

**Technical completion**

- Import and CLI resolution from fresh clone succeed.
- Unavailable model files are explicit; no silent online model/version switch.
- Smoke is not credited as a scientific fit.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="e04"></a>

## E04 — Measure storage, memory and runtime capacity for the next real units

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** None. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Actual local mounted volumes/power/device
- Frozen NLST plan and corrected fit storage requirements

**Work**

1. Check current power, writable real paths, backing-volume free bytes, RAM and chosen device.
2. Budget retained checkpoints, staged downloads, uncompressed scans, scratch and safety headroom separately.
3. Record capacity-limited tranche sizes and local/HPC route; refresh estimates after D03 and M04. Keep other-project and active sources untouched.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- RESOURCE_PLAN.json
- Measured pilot/per-fit resource updates

**Technical completion**

- Plan uses current observations, not October-14 free-space values.
- No arbitrary historical internal reserve becomes a permanent blocker; projected writes fit real available capacity.
- Transfers stop before backing storage reserve exhaustion.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="e05"></a>

## E05 — Provide standalone checkpoint/return handling for nightly sessions

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- NIGHTLY_RUNBOOK.md
- Existing trainer resume/deadline facilities
- Existing transfer receipts

**Work**

1. Implement only missing adapters for exact session timing, attempt logs and output collection; reuse existing resume logic.
2. Give each task one output namespace and each local process a recorded identity.
3. Exercise a small interruption/resume case and a failing exit; preserve both attempts.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Standalone session command or documented command batch
- Session/attempt/output schemas and one fixture receipt

**Technical completion**

- No dependency on Codex controller, expired window file or hourly reviewer.
- Interrupted unit resumes only with matching fingerprints; failed exit remains failed.
- Cutoff/deadline do not extend on waits.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="e06"></a>

## E06 — Prepare the same run matrix for an available HPC route

**Owner:** Sathvik + Claude. **Scope:** alternative_route. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, E04. **Human inputs:** H09.

**Inputs**

- revamp/campaign/execution_path_integration_2026-10-07_124017-2d0aea/compute/hpc/HPC_PLAN.md
- n08_fit_hpc4.sbatch and submit_n08_hpc4.sh
- Actual authorized cluster/account/storage details

**Work**

1. Adapt existing scheduler templates to the real cluster, environment and M02 argument-array driver.
2. Transfer only permitted inputs with checksum validation; execute a real target-machine smoke before full jobs.
3. Capture scheduler IDs, exit/accounting and checkpoint manifests; collect outputs into the same acceptance interface as local fits.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Cluster-specific job template and transfer manifest
- Target-device smoke/parity record
- Per-fit scheduler/return receipts

**Technical completion**

- No invented host/account/allocation.
- Scheduler success alone does not count as accepted fit.
- Local work remains runnable when no HPC route exists.

**Resources:** Cluster route unmeasured; profile actual CPU/GPU memory, walltime and threads.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d01"></a>

## D01 — Implement receipt-aware NLST continuation

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- planning/outreach_acquisition_2026-10-14/download_nlst.py
- NLST_DOWNLOAD_PLAN.json
- Append-only terminal transfer receipts and partials

**Work**

1. Create a versioned successor; preserve the sealed original. Replace hardcoded expired window constants with explicit session parameters.
2. Skip existing archives only when canonical success receipt, frozen-plan membership, size/hash and file identity agree. The old script only recognizes its first existing archive and cannot simply be rerun.
3. Reconcile incomplete files, bounded transport retries and terminal access errors. Keep frozen roster order and nested P1k/P3k/P10k selections.

**Command / implementation entry**

Current download_nlst.py has no argparse CLI. Implement and document the successor interface; do not use the expired script as a ready command.

**Outputs**

- Versioned resumable downloader
- Resume reconciliation manifest and bounded fixture checks

**Technical completion**

- Already verified archives are not downloaded twice.
- Unknown/mismatched existing bytes are quarantined for review, never silently accepted.
- Cutoff and backing-volume capacity checks operate on the actual session.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d02"></a>

## D02 — Finish the remaining frozen P10k imaging archive acquisition

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** D01, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Frozen P10k plan: 10000 identifiers
- 2391 terminal verified archive receipts; refresh before resume

**Work**

1. Continue remaining plan entries in order with the existing two-transfer design unless measured capacity justifies a change.
2. Checkpoint P1k, P3k and P10k membership separately; verify archive CRC, file/image count, header identity and unique SOP within each archive.
3. Keep terminal attempts and partials; seal aggregate counts/bytes and remaining queue each session.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Verified archive receipts and archive inventory
- P1k/P3k/P10k acquisition progress and remaining queue

**Technical completion**

- Unique plan IDs and receipts reconcile exactly.
- No whole-collection dump, implicit EGFR eligibility or pixel-processing credit.
- No retry loop on access/quota errors.

**Resources:** Network + external storage; frozen selected uncompressed total about 893 GB, compressed total still unknown.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d03"></a>

## D03 — Profile NLST decoding and preprocessing on a bounded tranche

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Designated10-personengineeringpilotrole,13series in HX/public_ct/p04/local/pilot_plan_v2_local.json
- Existing pilot allowance inventory and ingester; P10k people are a different role

**Work**

1. Execute the designated engineering pilot, preserving its cumulative person/series inventory and using the current session resource plan. Do not borrow P1k/P10k people as undocumented pilot subjects.
2. Keep raw archive identity, CT-series/SOP association, duplicate reconstructions and missing/corrupt items visible.
3. Measure peak RAM, decoded/scratch bytes and per-series time; choose stream/chunk layout from actual measurements.

**Command / implementation entry**

DATASETS.md NLST pilot_cli.py plan/release-template/verify-release/run/report interfaces. Replace retired controller coupling truthfully if needed; preserve cohort role and input checks.

**Outputs**

- Pilot processed rows/arrays and geometry sidecars
- DECODE_PROFILE.json
- Failed/excluded-series ledger

**Technical completion**

- Output rows trace to archive/source series.
- No silent axis/spacing/sample-order changes.
- Pilot images do not become accepted molecular subjects.

**Resources:** CPU/I/O; bound pilot size before decoding.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d04"></a>

## D04 — Process acquired NLST tranches for the selected pretraining use

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** D03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Verified archives as each D02 tranche closes; entire P10k not required to start
- Pilot-verified processing configuration and frozen role manifest

**Work**

1. Bridge already verified ZIP receipts to the scientific ingester through an explicit offline/archive adapter if absent; bind source ZIP/hash/config and avoid re-download.
2. Process only checksum-verified closed tranches, retaining person-level nested subsets and source identity.
3. Exclude alternate reconstructions from cross-split leakage; do not count multiple series as independent persons.
4. Record train/validation/test exposure and actual usable rows; retain rejected series and reasons.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Versioned pretraining rows/arrays
- NLST preprocessing manifest and role/exposure ledger

**Technical completion**

- Every processed output has a verified source receipt and geometry result.
- Actual usable counts, exclusions and expanded disk use reconcile.
- Pretraining has no access to downstream protected test persons.

**Resources:** CPU/I/O; full processed footprint determined by D03, stream to real external storage.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d05"></a>

## D05 — Complete the LIDC-IDRI source inventory and permitted missing acquisition

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** E01, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- LIDC-IDRI collection CT/XML inventory
- theia/data/lidc.py and existing local LIDC receipts

**Work**

1. Reconcile actual CT series, annotation XML, subjects and metadata against source inventory.
2. Download only missing permitted objects; record version, license/access and hashes.
3. Resolve XML-to-series/SOP coverage and enumerate unmatchable annotations without inventing labels.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- LIDC source manifest
- CT/XML pairing coverage and missing-source queue

**Technical completion**

- Source counts distinguish persons/series/nodules/readers.
- No claimed downstream row count from XML availability alone.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d06"></a>

## D06 — Connect LIDC consensus masks to usable training rows

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** E01, D05. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- theia/data/lidc.py
- theia/data/preprocess_masks.py
- Existing consensus NPZ outputs

**Work**

1. Implement explicit consensus-NPZ-to-row adapter: the advertised preprocess_masks path currently expects DICOM SEG.
2. Preserve SOP/slice correspondence, geometry and existing min-agreement definition; do not invent a new clinical nodule eligibility threshold.
3. Validate with a tiny known geometry fixture and real matched-source pilot.

**Command / implementation entry**

Existing source mask export: python -m theia.data.lidc --lidc_root LIDC_ROOT --min_agreement 3 --masks_out MASKS --out SUMMARY; implement downstream adapter before invoking preprocessing.

**Outputs**

- LIDC mask/row adapter
- Geometry and membership checks
- Documented actual command

**Technical completion**

- No SEG-parser misuse on NPZ.
- Consensus mask lands on the correct CT pixels and original source identity.
- Empty/unmatched sources fail explicitly.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d07"></a>

## D07 — Build the LIDC pretraining cohort under the selected consensus method

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** D06, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- LIDC CT/XML and corrected adapter
- Selected consensus and patient-level split definition

**Work**

1. Generate versioned masks and processed CT rows using the declared method.
2. Audit person-level partitions, missing annotations and source coverage; keep reader/consensus provenance.
3. Produce label-free grounding/pretraining input manifest without molecular claims.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- LIDC processed rows/masks
- Cohort flow and split manifest

**Technical completion**

- All usable rows are loadable and source-bound.
- No patient appears in both train and held-out role.

**Resources:** CPU/I/O; measure on pilot before full preprocessing.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d08"></a>

## D08 — Reconcile and preprocess NSCLC-Radiomics for its documented role

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** E01, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- NSCLC-Radiomics CT and tumor SEG sources
- theia/data/import_nbia.py
- theia/data/preprocess_masks.py

**Work**

1. Bind legacy LUNG1 grounding exposure: reconcile422source vs420processed vs stale421docstring, pretraining partitions and checkpoint lineage. Reuse unchanged historical inputs.
2. Inventory actual CT/SEG series and missing source objects.
3. Build source-linked tumor-mask rows with frame/geometry and empty-mask checks.
4. Keep this segmentation/pretraining role distinct from NSCLC-Radiogenomics molecular endpoints; assess overlap before pooled use.

**Command / implementation entry**

python -m theia.data.import_nbia --nbia_root NBIA_ROOT --out IMPORT_OUT; python -m theia.data.preprocess_masks --config CFG --raw_dir RAW --out_dir PROCESSED

**Outputs**

- NSCLC-Radiomics source/processed manifests
- Geometry/coverage failures and cohort-role ledger

**Technical completion**

- No presumed EGFR labels from this collection name.
- Rows trace to tumor-only SEG and selected CT series.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d09"></a>

## D09 — Close the corrected NSCLC-Radiogenomics source/attrition inventory

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Reviewed 211→188→158→153 cohort ledger
- Selected CT/SEG/AIM source receipts
- PF reviewed registry and content manifest

**Work**

1. Reuse the completed attrition reconstruction and 158-row manifest verification.
2. Resolve only source identity/availability gaps needed for M01; keep the three unknown import-selection reasons explicitly unknown unless new records exist.
3. Record AIM-only series mismatches, reconstructed air exclusions and reference-semantics limitations.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Versioned source-location and attrition addendum
- Exact input list for M01

**Technical completion**

- No duplicate reconstruction of unchanged historical flow.
- Current reference alignment is not claimed to certify historical M0 masks.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d10"></a>

## D10 — Reconcile TCGA-LUAD external CT and existing label metadata

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E02. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- TCIA TCGA-LUAD local metadata; TCGA-LUSC is unverified prospective scope under D19/X08.
- Historical external TCGA arrays/results and existing label records

**Work**

1. Inventory imaging series, molecular availability, time/role exposure and protected-vs-already-scored cases.
2. Resolve person/series crosswalk in local controlled storage and record usable/error reasons. Do not promote the unverified TCGA-LUSC roadmap into an existing external cohort.
3. Reuse historical small-TCGA predictions as historical evidence; prepare only genuinely new eligible or changed-input cases for M11.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Local external cohort manifest and exposure ledger
- Aggregate eligible/unresolved/excluded counts

**Technical completion**

- Previously scored patients never become fresh holdout by relabeling.
- Starter labelled ceiling is not inflated by unlabeled CPTAC cases.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d11"></a>

## D11 — Complete CPTAC-LUAD CT and molecular file availability inventory

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E02. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- TCIA CPTAC-LUAD CT subject inventory
- GDC_DOWNLOAD_SUMMARY_FINAL.json
- 36 verified public WXS MAFs

**Work**

1. Reconcile CT availability and exact GDC case/file mappings without reading quarantined molecular content.
2. Verify retrieval receipts and request/download only legitimately missing permitted files.
3. Keep the excluded controlled targeted-sequencing file and all missing modalities explicit; prepare the blinded linkage interface.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- CPTAC imaging/file-availability manifest
- Blinded linkage plan and aggregate coverage

**Technical completion**

- No repeat download of verified files.
- No claim of newly evaluable patients or decoded EGFR from file count.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d12"></a>

## D12 — Decode and link molecular outcomes only in the recorded exposure sequence

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** D11. **Human inputs:** H03.

**Inputs**

- Genuine H03 outcome definition and exposure/access record
- Frozen model/prediction hashes when prospective evaluation requires them
- Verified compressed MAFs and controlled crosswalk

**Work**

1. Implement parser/join on synthetic fixtures first; encode assay coverage, callable outcome, variant class and missingness.
2. After the required freeze/exposure record exists, decode permitted real files and link only the intended cases.
3. Distinguish verified negative from no variant row, incomplete panel and unknown result. Record mutation-positive count separately.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Versioned local CT–molecular endpoint table
- Assay/outcome dictionary and join/exclusion ledger
- Aggregate positive/negative/unknown counts

**Technical completion**

- Exactly one supported person/outcome record per evaluable endpoint or explicit longitudinal rule.
- No absent MAF row automatically treated as clinical-assay negative.
- Exposure timestamps and input hashes preserved.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


**Ordered execution stages**

- `D12.label_join` requires D11, M11.prediction_freeze: Qualified CPTAC labels linked after the selected model/prediction freeze and H03 exposure record.

<a id="d13"></a>

## D13 — Establish person-level overlap and split/exposure accounting across cohorts

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** D09, D10, D11. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Local crosswalks and all source/role manifests
- Retained R03 sampled-pixel overlap result

**Work**

1. Use permissible person/study/series identifiers and source lineage to detect cross-collection duplicates and repeat acquisitions.
2. Record remaining uncertainty where private identifiers cannot be compared; use image evidence only within its stated sampling/transform limits.
3. Freeze cohort roles and exclusion/overlap decisions before downstream fitting or protected evaluation.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Controlled overlap matrix and role manifest
- Public aggregate overlap/unknown-count summary

**Technical completion**

- Zero sampled pixel matches is not person-disjointness proof.
- Alternate reconstruction/repeated visit does not count as new independent person.
- No protected test-person exposure through pretraining or tuning.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d14"></a>

## D14 — Build the acquisition funnel for 1,000 paired CT–EGFR persons and the separate 10,000 aspiration

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02. **Human inputs:** H04.

**Inputs**

- DATASETS.md current/candidate register
- Existing cohort feasibility/outreach records
- User paired-cohort target and separate imaging target

**Work**

1. Record per source: available persons, CT coverage, assay coverage, linkage rights, timing, unique overlap, positives, usable pairs and reason unknown.
2. Keep institutional cohorts request-only until actual approved data/access arrives; reuse eight unsent drafts without sending automatically.
3. Report an honest sum/union range and gaps. Plan acquisition batches from supported availability; do not multiply published cohort totals into usable local participants.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- COHORT_FUNNEL.csv
- Unique paired-person/positive counts and target shortfall
- Next data-access requirements

**Technical completion**

- Imaging totals and molecular-evaluable totals remain distinct.
- 10,000 target is not presented as worldwide ceiling or journal minimum.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d15"></a>

## D15 — Prepare and run incoming institutional-cohort ingestion when access exists

**Owner:** Sathvik + Claude. **Scope:** external_access. **Status:** pending_reconciliation.

**Dependencies:** E01, D14. **Human inputs:** H03, H04.

**Inputs**

- Approved institutional export/data dictionary
- Actual data-use/access conditions
- Frozen cohort/endpoint/time-alignment method

**Work**

1. Prepare reusable metadata validation, de-identification checks, CT inventory, assay mapping and source-to-eligible flow before data arrives.
2. Once access exists, import into controlled storage and apply the declared method; record missingness, overlap and label exposure.
3. Return aggregate coverage and explicit unresolved mapping questions; do not fabricate local copies from published study descriptions.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Cohort-specific validated importer and local manifest
- Data dictionary, exclusions and eligible-pair flow

**Technical completion**

- Access and identity provenance traceable.
- No assumptions that a published CT cohort includes usable EGFR labels.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d16"></a>

## D16 — Assess the 19 labelled but unbuilt internal development cases

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** E02, D09. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Existing private list and 211→188→158→153 attrition evidence

**Work**

1. Resolve CT/annotation availability, reference UID and actual exclusion reason for each candidate.
2. Produce repairability/eligibility/bytes estimates without changing frozen158-person M1 membership.
3. If adopted later, create a separate cohort version with development/exposure role retained.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Aggregate repairability ledger
- Separate future-cohort proposal and source requirements

**Technical completion**

- No independent-external-person credit.
- No silent expansion of the current M1 matrix.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d17"></a>

## D17 — Build genuinely reader-resolved external input arrays and the final analysis join

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** R03, D10, D11, D13. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Accepted reader localizations/series identity
- Existing external adapter and frozen timing/air rules
- Declared endpoint/exposure record; D12 for included CPTAC labels

**Work**

1. Convert accepted centers/series into label-free versioned arrays with source/geometry/error ledger.
2. Lock model/predictions before any required protected label join; then align exactly one person to a supported endpoint.
3. Report stage/event counts and explicit exclusions. Current 13 TCGA + 28 CPTAC is conditional capacity, not achieved patients.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Accepted external input manifest and private aligned analysis table
- Source→reader→input→prediction→endpoint flow

**Technical completion**

- Development identity-check cases excluded from new evaluation count.
- Never-scored TCGA outcomes already exposed remain outcome-exposed.
- Newly evaluable count advances only after all actual requirements.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


**Ordered execution stages**

- `D17.input_arrays` requires R03, D10, D11, D13: Accepted label-free reader-resolved input arrays; this stage can close before prediction or label linkage.
- `D17.analysis_join` requires D17.input_arrays, M11.prediction_freeze: Aligned prediction/endpoint table after appropriate exposure and label evidence; genuine new-evaluable counts only here.
  - For CPTAC: D12.label_join.

<a id="d18"></a>

## D18 — Materialize a selected human-annotation robustness subset

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, D13. **Human inputs:** H10.

**Inputs**

- NLST 1,092-person reserve,LIDC,Interobserver1,CrowdsCure2018 source registers
- Declared single reference-comparison question

**Work**

1. Resolve exact annotation release,reference type,person overlap and CT mapping.
2. Build only the selected bounded subset with appropriate source-linked labels.
3. Keep reserve out of pretraining and do not force nodule/primary-tumor/line annotations into one interchangeable truth label.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Versioned human-reference subset
- Annotation-role/overlap/provenance ledger

**Technical completion**

- AI-derived masks not presented as human truth.
- Derivative releases never add new people.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d19"></a>

## D19 — Resolve every optional public candidate into an executable include/defer/exclude record

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** E02. **Human inputs:** H04, H10.

**Inputs**

- DATASETS.json: all 37 resource records
- Recorded official source links and source registers

**Work**

1. Refresh selected source release/access/license before use; reconcile CT-RATE,RadGenome,LNDb,LUNA25,QIN,Lung-PET-CT-Dx,Radiomics-Genomics,I3LUNG,PhysioNet,TCGA-LUSC/WSI and other named candidates.
2. For each, record role,format,person counting,endpoint,overlap and local availability; specify adapter/acquisition work only for selected experiments.
3. Keep hidden/gated data unavailable until genuine access, and extracted features distinct from raw CT.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Candidate decision matrix and source access records
- Selected-source adapter specifications

**Technical completion**

- Every candidate has an explicit state and reason.
- No unsupported universal downloader or implied local access.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d20"></a>

## D20 — Build the separate 250-person unlabeled NLST development cohort

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** D03, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Frozen 250-person development role
- Verified scientific ingester and role manifest

**Work**

1. Acquire/process selected development series separately from pretraining and held-out roles.
2. Use source-bound HU/geometry/sampling checks and retain failures.
3. Return real usable derivatives for preprocessing/augmentation choices.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Development derivative manifest
- Measured storage/QC/yield report

**Technical completion**

- No molecular labels or held-out robustness outcomes used for development choices.

**Resources:** Recorded expected source~22.47GB;actual compressed/derivative budget from pilot.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="d21"></a>

## D21 — Acquire and verify the held-out NLST same-acquisition reconstruction pairs

**Owner:** Sathvik + Claude. **Scope:** data_expansion. **Status:** pending_reconciliation.

**Dependencies:** D03, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Frozen1,000-person held-out role and1,052 technique-equal candidate pairs
- Declared pair eligibility rules

**Work**

1. Select exact index/alternate series and capture acquisition/frame/orientation/coverage/spacing/kernel/time evidence.
2. Exclude unresolved shared-study identity anomaly and record all pair failures without silent substitution.
3. Materialize source-bound pair derivatives; use patient-level weighting for multiple pairs.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Private verified pair ledger
- E0/E1/E2/E3/usable aggregate counts
- Paired derivative manifests

**Technical completion**

- Metadata candidate pair is not automatically a same-acquisition pair.
- No longitudinal substitution or double-counted persons.
- Final paired derivative manifest uses N06 aligned sampling; old independent16-slice arrays are not promoted as accepted paired derivatives.

**Resources:** Recorded planned source~187.68GB;profile actual ZIP/extraction/derivative cost.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


**Ordered execution stages**

- `D21.acquisition_headers` requires D03, E04: Verified source archives and same-acquisition header evidence; can proceed before sampler implementation.
- `D21.paired_derivatives` requires D21.acquisition_headers, N06: Final paired arrays use the N06 aligned sampler; retain sources until this step succeeds.

<a id="m01"></a>

## M01 — Execute real corrected-input S1-a acceptance

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, D09. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- PF/review/REPAIR_REGISTRY_S1_v1_optA_REVIEWED.json
- PF/source/manifest_S1_v1/CONTENT_MANIFEST.json
- Real bound CT/SEG files and existing corrected arrays

**Work**

1. Reuse manifest verification; execute the real rebuilt-array/source/loader comparison with the reviewed registry.
2. Write new acceptance and A15 membership outputs, preserving historical rejected receipts.
3. Repair a demonstrated failure narrowly and repeat only affected checks. Keep archive-byte equivalence unverified when not established.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md template A: build_m1_manifest.py verify; m1_acceptance.py --registry REG --registry-sha256 REGSHA --manifest MAN --out NEW_ACCEPTANCE --rebuild-arrays; a15_population.py --rows ROWS --out NEW_A15

**Outputs**

- Real M1 acceptance JSON
- A15 membership audit
- Source/rebuild/loader comparison logs

**Technical completion**

- Fixtures false, all required technical comparisons pass and decision accepted.
- Biological tumor truth remains a claim limitation where unadjudicated; it is not a blanket prohibition on the disclosed reference benchmark.

**Resources:** CPU/I/O; historical ~1.5h and ~7.43GB read, actual peak RAM unmeasured.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m02"></a>

## M02 — Bind a direct corrected-fit driver to the existing acceptance/readout contract

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- BW/n08/M1/FIT_MATRIX_M1.csv
- NR/run_n08_replication.py, accept_fit.py, data_manifest.py, fit_witness.py
- I3 final first-fit proposal

**Work**

1. Build a thin argument-array batch driver or documented equivalent around existing trainer/acceptance logic; retire only controller-specific admission coupling.
2. Inject paths.processed_dir, data.content_manifest and data.dataset_version explicitly: the matrix alone omits these overrides.
3. Preserve real fit/checkpoint/summary identity, fold membership, epoch-selection replay and acceptance receipt schema; adapt consumers consistently if schema changes.

**Command / implementation entry**

Use actual trainer flags and data overrides in COMPUTE_SOURCE_AUDIT.md template B; inspect accept_fit caller/schema before implementation.

**Outputs**

- Standalone matrix driver and command documentation
- Receipt compatibility fixture and acceptance integration

**Technical completion**

- No fake release/allow/procedure-validation receipt or test hook in real runs.
- Final executable/protocol/config bindings are acyclic; records do not hash themselves or future mutable outputs.
- No dependency on old Codex review or unrelated custodian promotion.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m03"></a>

## M03 — Run the real corrected loader/model/loss smoke on target hardware

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** M01, M02, E04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Accepted corrected manifest/rows
- B/query-union cell init42 fold0
- Actual target device and model cache

**Work**

1. Use a distinct smoke run with the existing two-epoch stop override.
2. Check finite loss, expected ground gradients, data/checkpoint identity and loader membership.
3. If hardware/precision differs from the fixed route, measure and record parity/tolerance before combining results.

**Command / implementation entry**

Retained run_n08_replication.py smoke after M02 adaptation, or template B with B-cell overrides, separate run_id and train.stop_after_epochs=2.

**Outputs**

- Two-epoch smoke outputs and hardware profile
- Technical acceptance/failure receipt

**Technical completion**

- Real data/model path succeeds; no synthetic success substituted.
- Smoke excluded from scientific 50-fit count.

**Resources:** Historical CPU smoke135.5s solo/585.3s loaded; refresh on real machine.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m04"></a>

## M04 — Complete the first corrected scientific fit

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** M03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Matrix row n08m1-cellA-init42-part1337-fold0
- Accepted data/splits and driver

**Work**

1. Execute the exact first row with fixed split1337 and init42; zero automatic failed-fit retries.
2. Retain best/last weights, full logs, fold summary and OOF predictions.
3. Run existing acceptance replay and measure wall/device/peak memory/output bytes for remaining scheduling.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md template B through M02 receipt-producing driver.

**Outputs**

- First planned fit outputs
- Accepted-fit or explicit failure receipt
- Measured fit resource profile

**Technical completion**

- Selected/stop epochs replay correctly and OOF members equal the frozen split.
- Exit0 alone is insufficient; negative/imprecise predictive result is not grounds for rerun.

**Resources:** Historical estimate0.2–3.3h; ~0.70–0.72GB checkpoints per fit, not RAM.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m05"></a>

## M05 — Complete the remaining fixed corrected 50-fit matrix

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** M04. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Existing A–E × init42/7 × five-fold matrix
- Frozen partition1337
- Closed accepted-unit inventory

**Work**

1. Execute genuinely missing rows, one fit at a time or safely isolated parallel jobs after measurement.
2. A/B:mean with ground0/0.5; C/D:query0 with union ground0/0.5; E:query0 with query0 ground0.5. Keep the exact full source config.
3. Resume only matching partials; keep failed attempts and both checkpoints. Do not change split seed with initialization or prune weights before downstream maps.

**Command / implementation entry**

Generate exact argument arrays from the checked-in matrix plus explicit M1 overrides, using M02. Continue across nightly windows, never silently extend one.

**Outputs**

- 50 planned fit identities accounted for
- Best/last checkpoints and fold/OOF/log outputs
- Per-unit technical acceptance inventory

**Technical completion**

- No missing/duplicate run-fold IDs.
- Accepted count is explicit; an unsuccessful unit is retained as unsuccessful.
- No favourable-outcome reruns or unplanned extra seeds.

**Resources:** Historical entire two-seed matrix12.7–20.5h, central13.4h; ~35.1GB retained checkpoints; measure, not a promise.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m06"></a>

## M06 — Produce corrected best/last maps and reload controls

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** M03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Five accepted folds for each ready cell/seed from M04/M05
- Corrected rows/splits/manifest/acceptance and exact checkpoints

**Work**

1. Start each cell/seed as soon as its five folds are accepted; do not wait for every other fit.
2. Generate readout-consistent mean/query0 and union spatial maps for best and last weights.
3. Verify baseline prediction reload, membership and checkpoint bindings; preserve per-patient metrics and arrays locally.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md template C; n08_maps_param.py --cell ... --run ... --ckpt-root ... --out-dir ... --which best|last --splits ... --rows ... --data-manifest ... --dataset-version ... --data-acceptance ...

**Outputs**

- 10 cell/seed map sets × best/last as units become ready
- Map receipts and reload-control metrics

**Technical completion**

- All planned folds are included.
- Best/last identities and inner-validation reload control are correct.

**Resources:** Forward inference; runtime unmeasured on corrected inputs.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m07"></a>

## M07 — Materialize the prespecified corrected replication readout

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** M05. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Accepted all-fold fits/maps/receipts
- M1 matrix, source-defined procedure and splits
- M06 is required for spatial quantities only; produce predictive readout as soon as its fit inputs are closed.

**Work**

1. Compute seed-specific (D−C)−(B−A) AUC interactions and their specified mean; preserve proper losses and per-fold/per-seed outputs.
2. Run paired uncertainty with the same resampled people across cells/seeds; produce all34 secondary estimates, MC error, recurrence and best/last sensitivity.
3. Retain family-validation qualifications and distinguish M0 discovery from M1 whose input and initialization changes are combined.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md template D; pass M1 matrix explicitly, --verify-files full and --which best/last as specified.

**Outputs**

- Machine-readable replication estimates/uncertainty
- Table/figure input files
- Membership and full-checkpoint validation report

**Technical completion**

- No pooling duplicate patients across folds/seeds.
- No naive final-fold-only result.json aggregation.
- Unavailable procedure validation downgrades claims, not fabricated pass.

**Resources:** CPU saved-result analysis; full checkpoint verification reads ~35GB.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.

**Conditional dependencies (required when that route/content is selected)**

- spatial readout: M06

<a id="m08"></a>

## M08 — Execute the prepared evaluation-only fragment and repaired-case sensitivities

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** M06. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Existing C5 worklist: eight affected and two companion cases,stream20261006
- Seven repaired-AMC evaluation-only exclusion specification
- Accepted M1 predictions/maps for ready units

**Work**

1. Locate the prepared implementation or add a narrow adapter to the existing worklist; no verified ready CLI is established by this audit.
2. Compute specified paired estimates and resampling for the fixed cases.
3. Keep seven-case exclusions separate; no silent changes to training population and no substitution of M0 models.

**Command / implementation entry**

Implementation/locate task: see I3 predecessor I05 packet and PF/manuscript/P07_PAPER_PRIORITY_QUEUE.md; bind exact adapter before execution.

**Outputs**

- C5 affected/companion tables and plot data
- Seven-case evaluation-only sensitivity table
- Exact model/input/method bindings

**Technical completion**

- Results describe evaluation-only effects, not causal training influence.
- No Tier2 retraining introduced implicitly.

**Resources:** Saved-result CPU plus any source-specified forward variants.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m09"></a>

## M09 — Build matched corrected native-feature and non-THEIA comparators

**Owner:** Sathvik + Claude. **Scope:** claim_conditional. **Status:** pending_reconciliation.

**Dependencies:** M01, E03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Corrected cohort/splits1337 and training covariates
- theia.analysis.frozen_features,baselines,site_decomposition,peritumoral_by_site,campaign_prediction

**Work**

1. Extract native encoder features with provenance, not a legacy randomly projected cache.
2. Fit the prespecified clinical/radiomic/site/geometry comparators with training-only fitting/imputation and fixed matched partitions.
3. Produce OOF probabilities on identical eligible people and explicit annotation/site tiers; adapter must preserve separate initialization and partition seeds.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md analysis catalog: frozen_features --config CFG --out TOKENS --prov PROV --device cpu; baselines --config CFG --theia_run RUN --seed 1337 --out OUT. Adapt manifest binding where needed.

**Outputs**

- Native feature cache/provenance
- Matched comparator OOF outputs and split audit

**Technical completion**

- No outer-test fitting or mask-derived feature portrayed as unassisted prediction.
- baselines seed1337 is partition seed; do not feed init42/7 as a substitute.

**Resources:** Encoder pass plus CPU fitting; current-input runtime unmeasured.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m10"></a>

## M10 — Compute corrected calibration, proper losses and incremental-value tables

**Owner:** Sathvik + Claude. **Scope:** claim_conditional. **Status:** pending_reconciliation.

**Dependencies:** M07, M09. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Corrected matched OOF predictions
- Approved calibration/fusion methods and training-side data

**Work**

1. Calculate AUC,log loss,Brier and calibration with paired clinical/site comparisons.
2. Fit any stacking/recalibration only inside the declared training boundary; keep raw and within-fold-rank estimators distinct.
3. Retain small-event instability, uncertainty and denominators; do not select a better method on outer-test results.

**Command / implementation entry**

Actual modules calibration --runs ... --out ...; fusion --theia ... --baselines ... --out ...; incremental --canonical ... --out ... . Bind corrected runset and data through M02/E01.

**Outputs**

- Calibration/proper-loss/incremental-value tables
- Training-boundary and matched-identity audit

**Technical completion**

- Logistic CITL is not raw mean-risk minus prevalence.
- No outer-test calibrator,threshold or fusion tuning.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="m11"></a>

## M11 — Run locked-model external prediction and prespecified external endpoints

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E03, D10, D13. **Human inputs:** H03, H06.

**Inputs**

- Human-selected locked model/version and declared external protocol
- Eligible external CT/labels with H03 exposure sequence
- CPTAC D12 or institutional D15 outputs only if included
- For new reader-resolved cases require D17.input_arrays; for custodian-run route require R02. Neither blocks permitted local inference on already accepted historical external inputs.
- For adapted-model external claims,lock scale/model before outcomes and route inference through the accepted N03 transfer path; exposed TCGA40 remains descriptive.

**Work**

1. Validate input adapter against native inference on a permitted fixture; freeze prediction/model identities before protected label opening.
2. Run external cases in resumable batches, retaining all input failures/exclusions and historical-vs-new exposure categories.
3. Execute the existing discrimination/calibration/grounding endpoints and comparator calculations on actual eligible persons; keep prediction and reference-localization outcomes distinct.

**Command / implementation entry**

Use existing evaluate/external_grounding/native library interface after binding actual CLI. frozen_external.py is a library, not an assumed argparse command. R02 supplies the verified packaged/native interface.

**Outputs**

- Local frozen external predictions and technical inference receipts
- Source-to-evaluable flow and endpoint tables
- Aggregate external result package

**Technical completion**

- Genuinely eligible person count, positive count and prior exposure are explicit.
- No clinical-assay-negative assumption from missing labels.
- No model tuning on this evaluation cohort.

**Resources:** Inference CPU/GPU after profiling; actual cohort size unknown.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.

**Conditional dependencies (required when that route/content is selected)**

- CPTAC endpoint scoring: D17.analysis_join
- new reader-resolved TCGA inputs: D17.input_arrays
- institutional endpoint scoring: D15
- custodian-return scoring: R05
- adapted-model external evaluation: N05

**Stage readiness**

- Native adapter plus accepted label-free inputs permits prediction freeze without outcome access.
- Score a selected cohort only after its conditional endpoint/join dependencies and H03 record are satisfied.

**Ordered execution stages**

- `M11.prediction_freeze` requires E03, D10, D13: Exact selected model/input-bound predictions frozen before protected label access; historical already-accepted inputs may proceed independently.
  - For new reader-resolved inputs: D17.input_arrays.
  - For adapted model: N05.
- `M11.endpoint_scoring` requires M11.prediction_freeze: Endpoint tables only for the selected cohort with actual supported outcomes.
  - For new reader-resolved cohort: D17.analysis_join.
  - For institutional cohort: D15.
  - For custodian route: R05.

<a id="n01"></a>

## N01 — Close the remaining NLST adaptation settings from the existing design

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** E02. **Human inputs:** H10.

**Inputs**

- planning/parallel_campaign_handoff_2026-10-02/PUBLIC_CT_ANALYSIS_SPEC.md
- PCT/code/adaptation primitives
- Latest HX public_ct return and draft settings
- P6/ADAPTATION_RUNNER_INPUTS.json contains21 draft/unresolved values; frozen memberships do not mean frozen runtime/training protocol.

**Work**

1. Extract the exact existing objective,optimizer,view,update-budget and transfer choices into a concrete configuration.
2. Identify unresolved values such as U with source-backed alternatives and computational consequences for human selection; do not invent final science settings.
3. Freeze six independent runs P1k/P3k/P10k × seeds42,7 and their matched downstream comparison before examining held-out outcomes.
4. Resolve U,views,augmentation,slice selection,negatives,schedule,device/precision and remaining fields without molecular/held-out tuning. Ordinary gradient accumulation does not create batch 32 / 62 simultaneous negatives semantics.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Final adaptation/transfer configuration or precise unresolved-setting record
- Six-run matrix and method/version binding

**Technical completion**

- Draft settings are not represented as preregistered or already executed.
- No held-out or molecular outcome tuning.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n02"></a>

## N02 — Implement the scientific NLST adaptation runner using existing primitives

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, N01. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- PCT/code/adaptation/ntxent.py,model.py,train_step.py,views.py,transfer.py
- Final N01 config
- Source-bound derivative manifests
- BW/public_ct/scripts/exposure_sampler.py; resume must restore optimizer,projector,RNG,sampler progress.

**Work**

1. Implement missing dataset batching,optimizer/checkpoint loop,resume and exact model/data/config identity using existing objective/model/train_step primitives.
2. Record distinct people actually contributing to applied finite updates,views/update counts and failure/replacement provenance.
3. Verify tiny interruption/resume and transfer-export behavior without rerunning the already completed synthetic benchmark merely for counts.
4. Pass finalized temperature explicitly to production NT-Xent rather than relying on a prototype constant; person_key,not seriesID,defines distinct batch members.

**Command / implementation entry**

Implementation required: no ready scientific adaptation-runner CLI was verified. Reuse PCT/code/adaptation primitives.

**Outputs**

- Runnable adaptation CLI and exact documentation
- Versioned checkpoint/resume/exposure schemas
- Focused real-path smoke outputs

**Technical completion**

- Scientific runner existence is demonstrated; primitives alone do not count.
- Skipped/failed updates do not inflate training exposure.
- No reset of a historical allowance by changing run names.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n03"></a>

## N03 — Integrate the adapted-vision transfer hook into downstream training

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, N01. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- PCT/code/adaptation/transfer.py
- Current theia/engine/train.py and model construction
- Declared transferred parameter set and baseline identity
- P6/P06_TRANSFER_REBASE.diff and theia_engine_train_py_rebased.diff candidate; live trainer lacks adapted_vision_payload.

**Work**

1. Implement the unmerged transfer hook in a versioned training path.
2. Load only the intended pretrained parameters,record exact checkpoint/tensor identities,and initialize remaining components according to the fixed seed.
3. Validate unchanged baseline behavior when hook disabled and declared transfer behavior when enabled.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Transfer hook and compatible configuration
- Loaded/missing/unexpected parameter report
- Baseline/transfer smoke and identity checks

**Technical completion**

- No silent partial load or random fallback.
- Transfer arm differs only as declared; gene labels did not enter adaptation.
- Only26 allowlisted encoder tensors transfer; projector,projection,GroundingHead and classifier do not. Validate payload scale/seed/membership/U/exposure and bind digest in trainer fingerprint.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n04"></a>

## N04 — Execute the six independent NLST scale/seed adaptation runs

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** N02, E04, N09. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Closed QC-accepted P1k/P3k/P10k derivatives from D04 as each tranche becomes available
- Frozen six-run matrix and exposure ledger

**Work**

1. Start P1k after its derivative tranche closes; do not wait for all P10k downloads.
2. Run each scale×seed independently under final method,budget and exact role membership.
3. Preserve best/final as method specifies,resume partials with fingerprints,and report distinct actually used people and finite applied updates.

**Command / implementation entry**

N02 implemented runner; record exact generated argv/config, not an invented currently available command.

**Outputs**

- Six adaptation checkpoints and full logs
- Scale/seed technical acceptance and exposure receipts
- Actual runtime/memory/bytes measurements

**Technical completion**

- No growing one checkpoint through scales and calling it independent runs.
- P10k training claim requires the specified people actually used,not only downloaded.

**Resources:** Historical synthetic scenarios for all six: ~0.95/3.02/11.97h at U313/1000/3000; excludes real I/O and is not a selected U. Six fp32 transfer payloads ~0.34GB..

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n05"></a>

## N05 — Execute the 30 matched downstream adaptation-transfer fits

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** N03, M01. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Closed, accepted N04 checkpoints as available
- Fixed cell-B downstream protocol and partition1337
- Existing N08 B baseline identities

**Work**

1. Run each of six adaptation runs across five fixed downstream folds with matching fit/selection procedure.
2. Reuse the10 N08 B baseline fits only if actual code/data/init/partition/selection identities match. Otherwise record mismatch and execute exactly the needed10 matched baseline fits in separate identities.
3. Retain full checkpoints,OOF outputs and same technical acceptance criteria; release analyses per complete scale/seed.

**Command / implementation entry**

Current trainer with N03 hook and exact N01 matrix; preserve M02 fit-receipt contract.

**Outputs**

- 30 planned downstream transfer fits
- Baseline reuse/mismatch manifest and any required baseline fits
- OOF predictions,checkpoints and acceptance receipts

**Technical completion**

- No stale baseline reused merely because cell name is B.
- No additional baseline rerun if compatible units already complete.
- Baseline reuse also matches downstream projection/head/classifier starting-weight schedule, optimizer and selection procedure; not just the patient split.

**Resources:** Historical 30 fits ~7.82–11.42h; conditional 10 baseline fits ~2.67–3.81h; ~21.1GB best+last storage. Remeasure on M1..

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n06"></a>

## N06 — Implement pair-aware sampling and the held-out reconstruction evaluator

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, N01. **Human inputs:** H10.

**Inputs**

- BW/public_ct/scripts/recon_feature_metrics.py
- D21 same-acquisition pair plan
- Frozen slice/pair eligibility definition

**Work**

1. Resolve pair-aware aligned slice selection: independent16-slice sampling has yielded8/16 matches against the existing≥10 criterion. Preserve method semantics; do not silently lower the threshold.
2. Wrap the existing metric library with an explicit input/checkpoint/pair interface and source-bound outputs.
3. Validate pairing,coverage,patient-averaged weights and identity controls on bounded development/fixture inputs.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Pair-aware sampler/evaluator CLI
- Alignment/coverage and patient-weight checks

**Technical completion**

- No pretending recon_feature_metrics.py has a ready CLI.
- Sampling correction is declared before held-out outcome evaluation.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n07"></a>

## N07 — Run frozen-encoder and adapted-encoder reconstruction robustness

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** N06, D21. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Six closed adaptation checkpoints from N04 as ready plus original frozen encoder
- Verified held-out pairs and finalized paired sampler

**Work**

1. Extract matched representations without altering the encoder on held-out pairs.
2. Compute source-specified patient-averaged cosine/norm contrasts and uncertainty with exact pair exclusions.
3. Retain frozen-versus-scale/seed outputs and method/coverage qualifiers.
4. Use native pre-projection pooled slice features,not the disposable projector or molecular 512D features. Use the exact declared E3 header rule and unchanged matched slices across encoders.

**Command / implementation entry**

N06 wrapper calls pair_metrics,patient_means,paired_summary in BW/public_ct/scripts/recon_feature_metrics.py; existing rule minimum 10 matched slices,z tolerance0.5×maximum thickness,10,000 bootstrap draws,seed20261003. Do not silently replace acquisition-number equality with a new time rule.

**Outputs**

- Per-pair local feature metrics
- Aggregate paired robustness tables and checkpoint bindings

**Technical completion**

- Same-acquisition verified pairs only for primary contrast.
- Technical representation stability is not EGFR validity or clinical benefit.

**Resources:** Forward encoder passes plus CPU aggregation;profilepaircount/coverage.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="n08"></a>

## N08 — Compute the declared scale/transfer readout and close the expanded program

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** N05. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Completed/reused baseline and transfer OOF fits
- Adaptation exposure counts and reconstruction outputs
- Frozen analysis specification

**Work**

1. Implement a small validated OOF-loader/CLI around BW/public_ct/scripts/primary_logloss.py:analyse(oof,labels,fold,B=10000,seed=20261003); check nonempty intended cohort,probability range,alignment and explicit exclusions before invoking.
2. Produce prescribed scale×seed downstream contrasts,paired uncertainty and robustness results.
3. Report actual used-person/update/runtime/storage quantities alongside data availability and unusable pairs.
4. Return machine-readable findings for human interpretation; no performance-improvement requirement or posthoc best-scale promotion.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Scale/transfer and robustness readout
- Complete program inventory and uncertainty/coverage tables

**Technical completion**

- All six adaptation / 30 downstream planned units reconciled; failures explicit.
- No molecular patient count from label-free pretraining exposure.
- Primary is P10k minus matched baseline patient EGFR log loss,mean of two seed-specific contrasts,clip1e-6,10,000 fold-stratified patient bootstrap draws; lower favours adaptation. P1k/P3k never replace missing P10k primary.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.

**Conditional dependencies (required when that route/content is selected)**

- reconstruction portion: N07

<a id="n09"></a>

## N09 — Run real-derivative adaptation smoke and measure actual update cost

**Owner:** Sathvik + Claude. **Scope:** expanded_study. **Status:** pending_reconciliation.

**Dependencies:** N02, D03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Properly role-bound engineering derivatives
- Original encoder cache and finalized adaptation config

**Work**

1. Exercise real loader,augmentations,finite update,exposure log,checkpoint,payload export and reload end to end.
2. Use the actual ten distinct pilot people for a small input-path check; never duplicate them to simulate batch32 distinct-person negatives.
3. Measure RAM,I/O,negatives per anchor,checkpoint cost and update throughput. Repeat old synthetic benchmarking only for a changed path or unresolved measurement.

**Command / implementation entry**

N02 implemented runner in an explicit smoke mode; actual scientific batch/objective semantics must remain separately declared.

**Outputs**

- Real adaptation smoke receipt
- Measured loader/update/checkpoint resource record

**Technical completion**

- Engineering weights excluded from six scientific adaptations.
- Only allowlisted tensors update and payload reload agrees.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="a01"></a>

## A01 — Reproduce only missing or changed historical all-fold evidence outputs

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02, E03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- SC/n08/n08_consolidate.py and discovery n08_readout.py
- SC/c22/C22_INTERPRETATION.json
- SC/n03/N03_REPLAY_ANALYSIS.json
- SC/rider/RIDER_CORRECTED.json
- Latest source pins and all-fold summaries

**Work**

1. Map retained N08,C22,N03,RIDER,TCGA/stability outputs to claimed table fields. Close by reuse when matching outputs already exist.
2. For a real missing/changed output, run the fixed saved-array consolidation in a fresh directory, preserving source values and denominators.
3. Reconcile exact family and simulation qualifications and missing units; leave scientific interpretation to humans.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md CMP-09 and existing fixed consolidation entrypoints; record exact selected command per missing output.

**Outputs**

- Historical selected-output index
- Only necessary regenerated aggregate tables and comparisons

**Technical completion**

- Never substitute final separately run fold4 result.json for all folds.
- No repeated unchanged simulation,20-seed sweep or bounded overlap run.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="a02"></a>

## A02 — Evaluate accepted corrected models on retained RIDER test–retest inputs

**Owner:** Sathvik + Claude. **Scope:** claim_conditional. **Status:** pending_reconciliation.

**Dependencies:** M03, E02. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- RIDER 28 paired people / 56 scan arrays and manual contour sources
- Accepted corrected model/checkpoints needed for selected comparison

**Work**

1. Reuse unchanged accepted RIDER arrays; rebuild only if a changed model input contract requires it.
2. Execute paired test/retest inference and the existing fixed repeatability/reference-alignment calculations.
3. Preserve per-scan contour semantics and person-level pairing; retain baseline M0 comparison.

**Command / implementation entry**

ACQ/rider/run_rider.py and existing RIDER_PROTOCOL.md; bind corrected model/config/rows explicitly, implementing a narrow adapter if needed.

**Outputs**

- Corrected-model paired inference and repeatability tables
- Input/contour/model provenance and exclusions

**Technical completion**

- No 56-independent-person denominator.
- No voxelwise registered-map or EGFR claim.

**Resources:** Forward inference plus CPU aggregation; current model runtime unmeasured.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="r01"></a>

## R01 — Materialize the prepared reader packet and perform technical viewer QA

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E02. **Human inputs:** H05.

**Inputs**

- RUN/readiness/attempt_03/R04/starter_copy_candidate.py
- Existing eight-case/680-image copy plan and exact file membership
- Actual authorized destination and secret handling

**Work**

1. Reuse the completed correction candidate and preflight; apply only source-compatible changes.
2. Create the planned packet in controlled storage, with exact manifest/file membership and missing-source refusal.
3. Open in the intended viewer and verify technical loading/orientation/presentation; route clinical content questions to human readers.

**Command / implementation entry**

Use prepared candidate interface after E01 help/contract verification; do not run an invented copy CLI. Sending requires the separately specified recipient/delivery action.

**Outputs**

- Actual packet and checksummed manifest
- Technical viewer QA record
- Delivery-ready local inventory

**Technical completion**

- No read/clinical-adjudication count from packet preparation.
- No secret in argv/log/GitHub; no extra images or missing cases.

**Resources:** ~358.55MB source images in existing eight-case proposal; measure copies.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="r02"></a>

## R02 — Finish the native-compatible custodian inference package

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- SC/custodian_package
- PF/provenance/custodian_v0.1.6 candidate
- Actual selected model,schemas and existing producer/validator fixtures

**Work**

1. Re-run synthetic producers, not just update their manifest hashes. Resolve demonstrated compatibility failures narrowly.
2. Build a versioned environment/container and inference entrypoint; compare packaged versus native outputs on a fixture and allowed actual input.
3. Require model/input/exposure provenance in returns and explicit refusal of malformed or unknown-exposure records.

**Command / implementation entry**

Existing package producers/validator harnesses; document exact built entrypoint and image/environment identity, not a speculative ready Docker tag.

**Outputs**

- Versioned runnable custodian package
- Native/package comparison and schema compatibility results
- Documented aggregate return schema

**Technical completion**

- Old 51/56 and41/42 results are not relabeled all-pass.
- No need to promote this package before local N08 fits.

**Resources:** CPU/container build; model load memory measured.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="r03"></a>

## R03 — Ingest genuine reader returns with provenance

**Owner:** Sathvik + Claude. **Scope:** external_access. **Status:** pending_reconciliation.

**Dependencies:** R01. **Human inputs:** H05.

**Inputs**

- Actual reader returned files and reading records
- Fixed case/reader identities and versioned packet
- Declared reading/adjudication method

**Work**

1. Validate returned schema,case completeness,reader identity,duplicate/partial status and packet version.
2. Keep original returns immutable and separate raw reads from adjudicated outcomes.
3. Prepare discrepancy tables for human adjudication and update actual return counts.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Controlled return ledger and validated files
- Aggregate completeness/discrepancy report

**Technical completion**

- No fabricated approvals,returns or clinical adjudication.
- Duplicate return does not create a new independent case/reader.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="r04"></a>

## R04 — Compute the prespecified reader endpoints from genuine completed reads

**Owner:** Sathvik + Claude. **Scope:** external_access. **Status:** pending_reconciliation.

**Dependencies:** R03. **Human inputs:** H05, H06.

**Inputs**

- Accepted real read/return data
- Existing approved reader endpoint/analysis specification

**Work**

1. Execute fixed paired reader/case calculations and uncertainty after the relevant real data exists.
2. Respect repeated cases/readers and missing-return rules; retain noninterpretable/partial strata.
3. Produce tables/data for human interpretation, without invented clinical utility conclusions.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Reader endpoint tables and underlying aggregate data
- Missingness and repeated-measure accounting

**Technical completion**

- Real returned count and denominator match ledger.
- Method changes go to H06 before inferential claims.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="r05"></a>

## R05 — Ingest custodian model-output returns independently of reader packets

**Owner:** Sathvik + Claude. **Scope:** external_access. **Status:** pending_reconciliation.

**Dependencies:** R02. **Human inputs:** H03, H04.

**Inputs**

- Actual custodian return bundle
- Locked model/package/schema and input/exposure evidence

**Work**

1. Validate model/package identity, person-level uniqueness, completeness and exposure/schema fields.
2. Keep failed/partial returns immutable and report exact incompatibilities.
3. Pass accepted predictions and separately qualified endpoint joins to M11; do not count a prediction return as a reader assessment.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Custodian return receipt and aggregate coverage
- Accepted local prediction/endpoint interface

**Technical completion**

- No R01 reader-packet dependency for a pure custodian inference route.
- Unknown labels/exposure remain unresolved,not accepted by missing fields.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="p01"></a>

## P01 — Regenerate Figure1 image from the corrected accepted flow

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02. **Human inputs:** H01.

**Inputs**

- Selected corrected attrition ledger and approved flow definition
- Existing already-corrected legend

**Work**

1. Create vector/raster image and source table from the selected flow.
2. Preserve unknown/reconstructed exclusions and separate internal/external populations.
3. Compare labels,arrows,counts and caption to the source data.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Figure1 source data,SVG/PNG and caption link

**Technical completion**

- Image itself matches the corrected legend; a text-only fix does not close this task.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="p02"></a>

## P02 — Finalize mechanical C06 layout and labeling without recomputing it

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02. **Human inputs:** H01.

**Inputs**

- Existing package_v2 SVG/PNG and panel JSON/CSV
- Approved main/supplement placement and F/G label choices

**Work**

1. Apply selected relabeling/layout in a new version while retaining exact panel data.
2. Preserve posthoc null/binormal and calibration qualifications in the caption.
3. Check panel legibility,source references and consistency across paper/supplement.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Final C06 assets and data-identity link
- Layout/caption comparison record

**Technical completion**

- No recalculated statistics solely for caption/layout changes.
- Human placement decision recorded.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="p03"></a>

## P03 — Assemble the final selected table/figure data and value traceability

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02, A01. **Human inputs:** H01, H06.

**Inputs**

- Human-selected paper claims and latest accepted outputs
- Current journal_set_v3 review package
- M07/M10/M11/R04 outputs only for claims actually selected

**Work**

1. Create one inventory mapping every displayed estimate,uncertainty,denominator and figure panel to source fields.
2. Regenerate only assets whose data or approved labels changed; include scientific missing values honestly.
3. Check all numerical/text references and cross-document counts against the chosen version.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Table CSV/JSON and figure-data inventory
- Value/denominator/source crosswalk
- Cross-document consistency report

**Technical completion**

- No mixed M0/M1 or historical/new external label.
- Missing scientific output cannot be filled with a surrogate number.
- All assets and scientific outputs selected in the final content manifest have their applicable conditional dependency completed; unselected extensions do not block packaging.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.

**Conditional dependencies (required when that route/content is selected)**

- selected Figure1: P01
- selected C06: P02
- corrected primary claims: M07
- corrected calibration/fusion claims: M10
- new external claims: M11
- reader-outcome claims: R04

<a id="p04"></a>

## P04 — Run a source-preserving reproduction of selected publication outputs

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E01, E03, P03. **Human inputs:** None beyond existing valid study/session scope.

**Inputs**

- Selected output inventory and permitted inputs
- scripts/reproduce.sh and newer revamp builders

**Work**

1. Extend only missing coverage so selected newer outputs have explicit reproduction commands.
2. Run in a fresh output directory/checkout, preserving originals. Separate saved-output reproduction from expensive refitting and data-access-required steps.
3. Compare fieldwise outputs using method-appropriate tolerances and save every exit/log/difference.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Reproduction commands and environment
- Fresh outputs/manifest
- Fieldwise comparison and unresolved-difference report

**Technical completion**

- No absolute mount/local-only import dependency hidden in published instructions.
- Scientific acceptance is not inferred from a successful reproduction script.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="p05"></a>

## P05 — Typeset and mechanically inspect the human-approved submission files

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** P03. **Human inputs:** H01, H07.

**Inputs**

- Human-approved main/supplement/title/cover text
- Final selected figures/tables/references
- Confirmed venue/article type profile

**Work**

1. Update the selected source format and compile/export all required documents.
2. Render and inspect pages for missing figures,clipping,numbering,reference and metadata errors.
3. Count against current chosen journal requirements; report remaining OPEN fields rather than invent authors/ethics statements.

**Command / implementation entry**

Use actual chosen document toolchain after source selection; older paper/ PDFs are not a build of the October review set.

**Outputs**

- Submission source files and compiled exports
- Build logs and layout/count report

**Technical completion**

- No ghostwritten scientific manuscript or initial IJAS prose.
- Inherited Methods800 rule is not enforced as official unless confirmed.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="p06"></a>

## P06 — Create the final technical release/reviewer bundle

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** P04, P05, P07. **Human inputs:** H07.

**Inputs**

- Frozen selected code/results/assets
- Approved licenses and release/access inventory

**Work**

1. Bundle reproducible safe code,aggregate outputs,dependency/model identities and acquisition instructions.
2. Produce blinded review copy when required and inspect metadata,text,paths and figures for identity/private data leakage.
3. Record excluded controlled inputs and how an authorized reviewer can obtain them; hash the final package.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Technical release archive/manifest
- Reviewer package and disclosure/access inventory

**Technical completion**

- No clinical images/subject predictions/private crosswalks/unsent drafts in public package.
- Public deposition/submission and author attestations remain explicit human actions.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="p07"></a>

## P07 — Populate technical reporting and IJAS evidence attachments

**Owner:** Sathvik + Claude. **Scope:** core. **Status:** pending_reconciliation.

**Dependencies:** E02, P03. **Human inputs:** H07, H08.

**Inputs**

- Frozen results and actual approvals/contribution records supplied by humans
- CLAIM/TRIPOD+AI templates and current IJAS route

**Work**

1. Map reporting items to real methods/output locations,flagging missing human statements.
2. Compile chronological code/AI contribution log and student-provided evidence files.
3. Check file inventory,dates,links and required formats without writing the initial student plan/abstract/paper/poster.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Technical reporting evidence map
- AI/code chronology and reproducibility annex
- IJAS technical evidence inventory

**Technical completion**

- No invented approvals,contributions,assay labels or deadline.
- Submission readiness awaits human scientific and institutional decisions.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x01"></a>

## X01 — Validate predictive/spatial null procedures if retaining the familywise claim

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03. **Human inputs:** H01, H10.

**Inputs**

- Existing null_size_study.py
- Production replication procedure

**Work**

1. Run fixed predictive null regime/replicates using production readout.
2. Implement the genuinely missing spatial generator only under the declared null design; report type-I and coverage uncertainty.
3. Failure/absence leaves secondary reporting exploratory and does not block M01–M07.

**Command / implementation entry**

python PX/n08/readout/null_size_study.py --regime moderate --replicates 1000 --start 0 --out-dir OUT

**Outputs**

- Null-size and coverage tables
- Known-answer/alignment results

**Technical completion**

- Apply existing V rule without claiming it validates .05/34 tail.

**Resources:** Historical~6.35CPU-s/replicate,~1.8CPU-h per predictive regime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x02"></a>

## X02 — Transfer selected mechanism/intervention analyses to corrected models

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03, M07. **Human inputs:** H01, H10.

**Inputs**

- Accepted M1 checkpoints/rows and training-only reference pools
- Existing M0 mechanism modules/receipts

**Work**

1. Select only mechanism claims retained by researchers.
2. Adapt corrected manifest/matrix bindings for attention/value donors,coalitions,context,attribution/randomization and stress; existing modules and exact CLI shapes are cataloged.
3. Shard by fit/fold and preserve identity/reload/donor-role controls; do not relabel old G50/G0 results as N08 B/A.

**Command / implementation entry**

COMPUTE_SOURCE_AUDIT.md existing analysis catalog; add only required corrected-data adapters.

**Outputs**

- Versioned per-unit mechanism shards
- Paired effect/uncertainty tables

**Technical completion**

- No unapproved method or test-patient donor use.
- Unchanged M0 analyses close by reuse.

**Resources:** Forward/backward jobs; profile, old backward-heavy jobs often CPU faster.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x03"></a>

## X03 — Implement and run wrong-mask training controls if selected

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03, M01. **Human inputs:** H01, H10.

**Inputs**

- Accepted rows/partitions
- Declared within-training permutation/displacement method

**Work**

1. Reuse A/B absent/correct controls.
2. Implement training-only wrong-mask permutation and geometry-matched displacement; audit masks/gradients.
3. Execute declared new arms with matched budget and separate run identities.

**Command / implementation entry**

No ready wrong-mask training CLI verified; implement narrow control builder and explicit new matrix.

**Outputs**

- Control-generation code/manifest
- New fit/OOF outputs and contrast tables

**Technical completion**

- No held-out mask donors.
- New controls not silently part of frozen50.

**Resources:** Historical ~10 new fits for two one-seed arms.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x04"></a>

## X04 — Implement annotation-efficiency subsets if selected

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03, M01. **Human inputs:** H01, H10.

**Inputs**

- Accepted rows
- Declared 0/25/50/100% training-annotation design

**Work**

1. Create fixed training-only patient subsets and preserve test labels/membership.
2. Run the declared subset draws/fit matrix.
3. Report achieved annotation/event counts and uncertainty.

**Command / implementation entry**

New subset adapter/matrix required; not an existing ready flag.

**Outputs**

- Subset manifests
- Fit and efficiency tables

**Technical completion**

- No tuning subset selection for favourable outcome.

**Resources:** Historical proposal ~20 fits; actual chosen design governs.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x05"></a>

## X05 — Run clipped grounding-pretraining transfer comparisons if needed

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03. **Human inputs:** H01, H10.

**Inputs**

- LIDC/NSCLC-Radiomics masks or chosen accepted mask cohort
- Existing warm-start results and default.yaml warnings

**Work**

1. First check for newer successful clipped reruns.
2. Pretrain on disjoint permitted masks; compare fixed no-warm,head-only and vision+head arms under declared clipping.
3. Retain NaN/dead folds and actual method differences.

**Command / implementation entry**

python -m theia.engine.pretrain --config CFG --rows ROWS --out CKPT --epochs N --val_frac .15; use actual model.grounding_pretrain_ckpt/model.warm_start_vision/train.backbone_lr_mult/train.grad_clip overrides.

**Outputs**

- Pretraining checkpoint/provenance
- Matched transfer fit matrix and results

**Technical completion**

- No stalled-fold omission or protected-test pretraining.

**Resources:** Pretraining plus new matched fits; profile actual cohort.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x06"></a>

## X06 — Run additional architecture/frozen-versus-finetuned comparator if selected

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03, M01. **Human inputs:** H01, H10.

**Inputs**

- Native features
- Declared architecture/tuning budget and fixed cohort

**Work**

1. Reuse existing frozen BiomedCLIP probe where compatible.
2. Implement alternative encoder/adapter where absent; do not invent a supported config value.
3. Run training-only tuning and matched OOF evaluation.

**Command / implementation entry**

Existing frozen-feature path plus explicit new architecture implementation.

**Outputs**

- Architecture implementation and smoke
- OOF comparator results

**Technical completion**

- Common eligible people and partitions; no outer-test selection.

**Resources:** Encoder/training capacity unmeasured.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x07"></a>

## X07 — Implement within-site training-label permutation if selected

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03, M01. **Human inputs:** H01, H10.

**Inputs**

- Fixed cohorts/site labels/splits
- Declared null-training design

**Work**

1. Permute training labels only within site.
2. Keep outer truth and budget unchanged; create separate run IDs.
3. Execute fixed null arms and return actual statistics.

**Command / implementation entry**

New training-label adapter/matrix; do not substitute randomization.py.

**Outputs**

- Permutation audit
- Null fits/OOF outputs

**Technical completion**

- Existing parameter randomization is not label randomization.

**Resources:** Historical 5 fits per one-seed arm.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x08"></a>

## X08 — Develop geometry-aware3D or CT–pathology branch if selected

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03. **Human inputs:** H01, H10.

**Inputs**

- Verified geometry/raw sources
- Paired pathology availability and patient overlap ledger

**Work**

1. Produce feasibility and person-level modality intersection before training.
2. Implement loaders/backbones/fusion for chosen physical-mm/slice-aware/3D or multimodal design.
3. Run actual geometry/interface smoke,then declared fixed split experiment; missing modality stays missing.

**Command / implementation entry**

New implementation scoped by H10; existing code entrypoints identified in source audit only.

**Outputs**

- Feasibility/cohort ledger
- Loader/model implementation
- Versioned experiments

**Technical completion**

- Roadmap/config comment is not runnable implementation.
- No paired pathology claim from CT-only cohort.

**Resources:** Large preprocessing/training; unknown capacity.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x09"></a>

## X09 — Run Tier2 retraining or input-versus-initialization separation if selected

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03, M07. **Human inputs:** H01, H10.

**Inputs**

- Accepted M1 matrix
- Human-declared new sensitivity design

**Work**

1. Specify exact additional cells and case-exclusion policy.
2. Run all declared units with immutable data versions.
3. Produce matched contrasts separating the chosen factors.

**Command / implementation entry**

Design then concrete matrix; no ready CLI/matrix invented.

**Outputs**

- New versioned matrix
- Fit outputs and contrast tables

**Technical completion**

- Evaluation-only M08 does not estimate training influence.
- No extra 40+ fits without a declared design.

**Resources:** Additional full fits; size from selected matrix.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x10"></a>

## X10 — Stabilize and evaluate generation/reader-benefit branch if retained

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03. **Human inputs:** H01, H10.

**Inputs**

- Existing generation/reader modules
- Declared rationale endpoints and genuine human reads

**Work**

1. Resolve actual generation gradient/runtime failures on a bounded test.
2. Run fixed training/inference and existing factuality/confabulation calculations.
3. Build technical reader cases; genuine benefit measurement follows real reader data.

**Command / implementation entry**

Actual build_cases/serve/build_demo interfaces; bind generation config explicitly,default loss0 is not a trained-generation result.

**Outputs**

- Generation stability and fit outputs
- Factuality tables and reader packets

**Technical completion**

- Fluent text metrics are not clinical correctness.
- run_eval.sh builds cases,not quantitative model evaluation.

**Resources:** Training, inference and reader work; profile.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x11"></a>

## X11 — Run additional selection/stability study only if scientifically retained

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, E03. **Human inputs:** H01, H10.

**Inputs**

- Existing20-seed stability and mass/pointing-gated3-seed results
- Declared new selection/stability question

**Work**

1. Reconcile/reuse completed sweeps first.
2. If required,execute only genuinely missing declared units with independent output paths.
3. Keep selection-rule versions separate; retain weights needed downstream.

**Command / implementation entry**

Use current explicit trainer matrix; legacy stability.sh may prune checkpoints.

**Outputs**

- Missing-unit results or reuse receipt
- Selection/stability tables

**Technical completion**

- Do not use legacy pruning script on needed checkpoints.
- No automatic canonical headline promotion.

**Resources:** Potentially many fits; avoid unchanged repeats.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.


<a id="x12"></a>

## X12 — Implement retained multigene endpoints and the declared discovery branch

**Owner:** Sathvik + Claude. **Scope:** optional_research. **Status:** pending_reconciliation.

**Dependencies:** E01, E02, D13. **Human inputs:** H03, H10.

**Inputs**

- Existing EGFR/KRAS source labels and any separately verified additional endpoints
- Human-declared target,hypothesis,multiplicity and validation design

**Work**

1. Inventory real assay coverage and target-specific eligible/positive counts; do not use expression as mutation status.
2. Implement selected target or paired-modality adapters and frozen experiment matrix.
3. Execute only prespecified model/discovery computations; return features,hypothesis outputs and validation evidence without inventing biological interpretations.

**Command / implementation entry**

Implementation/entry-point binding task; do not invent a ready CLI.

**Outputs**

- Endpoint-specific cohort dictionary and exposure manifest
- Explicit experiment matrix/code
- Raw discovery/evaluation outputs for human review

**Technical completion**

- No broad multigene capability claimed from EGFR-only execution.
- Exploratory discovery and confirmatory validation use appropriately separated data.

**Resources:** CPU / small metadata; measure actual runtime.

**Reuse:** Consume any newer compatible result first. Reuse if code, data, method and output criteria match; otherwise retain prior output and record the exact changed input or missing unit.
