# THEIA dataset inventory and remaining execution work

Audit date: 2026-10-15. Owner of every implementation, acquisition and computation card below: **Sathvik + Claude**. No teammate or collaborator is assigned. Scientific design decisions, independent verification, interpretation and manuscript writing belong in the separate human/Codex workstream. Author names are not execution assignments.

This is an audit of existing source code, safe catalogue metadata, saved summary receipts and reviewed manuscript evidence. No images were decoded, molecular quarantine files opened, downloads launched, training run, accounts accessed or messages sent. Current official websites were not re-queried; URLs below are the official sources already recorded by the project. Provider counts and licenses retain the dates/versions of those records until refreshed. Historical instructions are evidence of design and implementation, not fresh execution authority.

## Path key and source hierarchy

Repository root `ROOT` = `the local research checkout (ROOT)`.

| Key | Repository-relative location |
|---|---|
| `PCT` | `revamp/campaign/parallel_expansion_2026-10-02/public_ct` |
| `BW` | `revamp/campaign/nine_hour_corrected_inputs_2026-10-02` |
| `HX` | `revamp/campaign/nine_hour_corrected_inputs_2026-10-02/external_nine_hour_continuation` |
| `ACQ` | `revamp/campaign/acquisition_completion_2026-09-21` |
| `OUT` | `planning/outreach_acquisition_2026-10-14` |
| `CLOSEOUT` | `revamp/claude_runs/nine_hour_2026-10-13_210100` |

The fast local mirror of closeout receipts is `the retained local campaign mirror (CAMP)`. Repository reads of `OUT/NLST_DOWNLOAD_STATE.json`, `OUT/NLST_DOWNLOAD_PLAN.json`, `OUT/NLST_TRANSFER_RECEIPTS.jsonl` and `OUT/GDC_DOWNLOAD_SUMMARY_FINAL.json` were also completed. They still report the October 14 terminal state. No newer acquisition receipt was found in that canonical acquisition folder. This is not a claim that every possible work directory was searched.

Important source registers:

- `planning/rigor_revamp_2026-09-16/07_DATA_RESOURCE_REGISTER.csv`: historical broad resource list, with published populations rather than acquired totals.
- `planning/january_submission_2026-10-04/PUBLIC_COHORT_FEASIBILITY.md`: provider/source-scope distinctions; dated October 4–5.
- `planning/parallel_campaign_handoff_2026-10-02/PUBLIC_CT_ANALYSIS_SPEC.md`: public CT adaptation design and denominator definitions.
- `PCT/sources/SOURCE_REGISTER.json`, `PCT/manifests/NLST_CANDIDATE_SUMMARY_v1.json`, `PCT/ROLES_PROTOCOL.json`, `PCT/OVERLAP_LEDGER.json`.
- `BW/public_ct/RETURN.md`, `HX/public_ct/RETURN.md`, `HX/public_ct/STAGE_COUNTS.json`, `HX/public_ct/P04_PILOT_PLAN.json`, `HX/public_ct/P08_SCALE_PROPOSAL.md`: successive implementation/status evidence. Earlier zero-download statements are superseded only for the October 14 P10k archive acquisition, not for pixel processing or training.
- `ACQ/AUXILIARY_DATA_REGISTER.md`: useful source/role record, but its old molecular ceiling, internal raw-data absence, and acquisition statuses are superseded by later evidence. Do not copy those stale statements into GitHub issues.
- `CLOSEOUT/codex_closeout_review/journal_set_v3/manuscript/SUPPLEMENT_reviewed_copy.md`, sections S2, S7, S8; `CLOSEOUT/codex_closeout_review/D05_reviewed_copy.json` and `D06_reviewed_copy.json`: latest aggregate attrition, reader and exposure findings inspected here.

## Counts that must remain separate

Each dataset status should contain: advertised population; enumerated unique people; metadata-selected; downloaded/archive-verified; pixel-decoded; preprocessing/QC accepted; model-input arrays available; actually used in an applied training update; assay-linked; localization-eligible; evaluated; and previously exposed versus genuinely protected outcomes. Unknown fields remain unknown, not zero unless a receipt explicitly supports zero.

One person can have multiple studies, series, reconstructions, annotation objects and slices. Those are not additional people. Model-input availability does not establish molecular eligibility. An absent qualifying variant in an open filtered MAF is not a clinically verified negative. Mirrors and annotation derivatives do not create additional people.

## Established or explicitly planned dataset inventory

| Dataset | Existing role and verified status | Remaining dataset operations |
|---|---|---|
| NSCLC-Radiogenomics | Internal development. 211 source CT-listed subjects; 188 imported; 158 historical input arrays; 153 EGFR two-class (40 mutant), 5 EGFR unknown. 119 supplied extent masks and 39 point-tier inputs. Corrected M1 candidate has 158 rows but no accepted real M1 input release or corrected fits in latest closeout. | Finish source-preserving corrected input acceptance, repair/linkage/geometry checks and versioned arrays; separately assess 19 labeled-but-never-built development subjects without altering frozen M1 membership. |
| NSCLC-Radiomics / LUNG1 | Prior grounding development exposure; source 422, project evidence records 420 processed crops. Not a fresh external molecular cohort. | Bind legacy pretraining membership/weights/transform provenance; any rebuild must be a new version; resolve annotation provenance and tumor-only segments. No blanket re-download indicated. |
| LIDC-IDRI / LUNA16 | Auxiliary nodule/reader-uncertainty/localization route. Source register 1,010 people/1,018 scans; LUNA16 is a derivative subset of 888 scans. XML converter exists; no verified current acquired, processed or model-used count established by this audit. | Reconcile actual files/receipts; verify access/version, XML↔CT SOP linkage, reader consensus policy, explicit mask-to-row adapter, patient grouping and overlap before use. |
| NLST P10k | Label-free representation development. Frozen plan 10,000 people, one index series each. 2,391 archives verified, 118,384,085,731 compressed bytes; 7,609 remain. All P1k archives and 2,391 of P3k are verified. The October 14 transfer decoded no pixels. | Receipt-aware transfer successor; real ingestion pilot; archive-to-derivative adapter; QC and versioned derivatives; complete P3k/P10k; separate development and held-out reconstruction-pair acquisition; exposure ledger. |
| NLST human annotation derivatives | 1,092-person reserved union in frozen roles. Includes LIDC-annot-NLST501, NLSTseg, Sybil-related lesion references and new-lesion longitudinal annotations. These are not extra people. | Resolve annotation release/version/access and exact reference types; keep reserve out of pretraining; only build localization inputs after a specified evaluation question. |
| NLST AI-only derivatives | BAMF/AIMI, nnU-Net-BPR and TotalSegmentator membership flags recorded. Not independent human tumor references. | Preserve provenance and derivative overlap flags; use only for an explicitly defined engineering/segmentation comparison, not molecular or human-reference credit. |
| TCGA-LUAD + Crowds Cure 2017 | Existing external research-endpoint evaluation: 40 patients, 4 positives/36 research negatives; outcomes exposed. 67 CT subjects in local source enumeration; 59 two-class endpoint records, 8 unknown. Four adapter-development subjects are separate. | Reader-enable remaining candidates; maintain timing/assay exclusions; build and lock label-free inputs/predictions before further permitted joins. Existing 40 can be reused descriptively for corrected-model comparison. |
| CPTAC-LUAD | Provider radiology listing 38 vs local CT enumeration 37; source aggregate EGFR publication 36 imaged. 29 selected index CTs acquired; 28 eligible before reader localization; 0 evaluated. 36 public WXS MAFs acquired, compressed/quarantined, no label decoding/join. | Reconcile version/count differences; reader localization, input construction, model/prediction lock, then authorized molecular decode/linkage and endpoint qualification; labels/exposure disclosure. |
| RIDER Lung CT + contour release | Test–retest/reference alignment only. 28 paired people, 56 CT series, 13,488 CT images and 56 manual contour files used historically. Source collection has 32 people. No EGFR endpoint. | Reuse accepted pairs/arrays; only rebuild if corrected input contract actually requires it; preserve manual ROI/source linkage and per-scan coordinate semantics; re-score with accepted corrected models as its own computation. |
| Crowds Cure 2018 | Candidate alternate internal reference/annotation-uncertainty analysis, 88 internal-cohort people measured in historical register. Not additional people or fresh molecular validation. | Reassess now-restored raw CT availability, inspect annotation release and joins; implement only a frozen reference-comparison question. |
| BiomedCLIP / PMC-15M | Existing upstream encoder exposure. Public CT participant overlap with figure-caption pretraining corpus is unknown. | Pin original weights/version/license and exposure declaration; do not claim patient-level exclusion from incomplete corpus metadata. No need to download PMC-15M for this backlog. |

## Dataset details and runnable interfaces

### 1. NSCLC-Radiogenomics: clinical table, CT, SEG and AIM

Official source: <https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/>. Use the collection-specific citation/license record; public availability does not make every derivative freely redistributable. `theia/data/download.py` points to `https://services.cancerimagingarchive.net/nbia-api/services/v1`. Its old account/token prose is not a current provider-access determination; the code accepts an optional local token. Do not expose credentials in docs.

Code: `theia/data/download.py`, `import_nbia.py`, `preprocess.py`, `preprocess_masks.py`, `aim.py`, `dataset.py`; historical config `configs/default.yaml`. Historical arrays: `data/processed/rows.jsonl` and adjacent `.npz`; provenance `results/dataset_provenance.json` (use only aggregate fields publicly). Raw-layout contract: `data/raw/nsclc_radiogenomics/clinical/clinical.csv`, `aim/`, `dicom/<subject>/CT`, `SEG`; `series_manifest.csv`, `series_index.json` and importer logs preserve mapping locally.

Actual module interfaces:

```sh
.venv/bin/python -m theia.data.download --raw_dir <new-raw-dir> [--no_extract]
.venv/bin/python -m theia.data.import_nbia --nbia_root <retriever-tree> --out <new-raw-dir>
```

The generic downloader lists/downloads **every** CT/SEG/RTSTRUCT series. It is not the right command for silently rebuilding the frozen corrected cohort. The importer chooses an axial, non-localizer CT, prefers the SEG reference when usable, then ranks non-fusion/non-PET and non-derived series, then image count. The old downloader flattens each subject's CT into one directory whereas the importer selects a series: audit which route generated the actual input and never assume these are interchangeable.

The latest reconstructed attrition is 211 → 188 → 158 → 153. The 23 not imported comprise 20 without the relevant local SEG/AIM download and 3 AIM entries whose CT reference was not found in the catalogue, with original selection reason unrecorded. The 30 imported-but-unbuilt comprise 26 without either known target-gene label plus 4 air-rule failures; 3 of those failures followed the wrong-series frame fallback. Nineteen two-class EGFR subjects were never built: 16 non-imported plus 3 air exclusions. They remain from development institutions and have exposed clinical records; they are not independent external capacity.

Known historical defects: 7 point-tier inputs centered from an annotation frame on another series; 2 SEG patients with misplaced masks; 10 with sampled planes lacking the supplied extent or selecting only a minor component, with overlap between categories. M0 crop-jitter used Python's salted hash with lost salt, so the cached historical arrays cannot be regenerated identically. Preserve their hashes; corrected deterministic arrays are M1, not a replacement of M0.

Corrected-source locations: `BW/input_repair/audit/SEVEN_REFERENCE_SERIES.json`, `BW/input_repair/SEG_DEFECTS.json`; `HX/input/M1_S1_candidate_v1` and provisional raw-tree/registry lineage are part of the candidate. The training/release audit owns exact successor acceptance commands. The October 14 decision says raw NSCLC CT already resolves to `/Volumes/Research`; the earlier assertion that all raw CT was absent is stale. Refresh symlink resolution and metadata first, not a fresh whole-cohort download.

### 2. NSCLC-Radiomics / LUNG1 and its SEG representation

Official source: <https://www.cancerimagingarchive.net/collection/nsclc-radiomics/>. Historical project source record states CC BY-NC 3.0; preserve noncommercial conditions and refresh the exact selected annotation release's license. Source 422 people vs historical 420 processed crops; module docstring says 421, which is not a verified present denominator.

Prior roles are grounding-head pretraining and spatial exposure. This is not CT–EGFR evaluation and cannot be relabeled as unseen external localization. `theia/data/preprocess_masks.py` walks imported per-patient CT/SEG and emits `.npz` images/ROI plus `rows.jsonl` with `labels={}`. `RadiogenomicsDataset` maps absent gene labels to −1 and classification loss masks them. `segment_mask` selects tumor-labeled segments from multistructure planning SEGs rather than thresholding lungs, heart and other structures together. Source original delineations and a SEG conversion are representations of the same people.

```sh
.venv/bin/python -m theia.data.import_nbia --nbia_root <radiomics-tree> --out <new-raw-dir>
.venv/bin/python -m theia.data.preprocess_masks --config <config> --raw_dir <new-raw-dir> --out_dir <new-derived-dir>
.venv/bin/python -m theia.engine.pretrain --config <config> --rows <rows.jsonl> --out <checkpoint.pt> [--epochs N] [--val_frac F]
```

Default preprocessing output is `data/processed_pretrain`; do not overwrite an existing historical cache. Actual downloaded/source/derivative byte totals were not established here. Need a manifest-only inventory and recorded exclusions for the difference between source and processed counts, plus pretraining split and checkpoint linkage. Any legacy pretraining rerun is a separately justified model computation, not a required download task.

### 3. LIDC-IDRI / LUNA16

Official source/derivative routes already recorded: <https://www.cancerimagingarchive.net/collection/lidc-idri/> and <https://luna16.grand-challenge.org/>. Dataset register's LIDC documentation source is <https://www.nature.com/articles/sdata2017124>. Treat release-specific access/license as unresolved until recorded rather than inheriting one license across LIDC, LUNA and LNDb. No verified EGFR table is established.

Actual parser `theia/data/lidc.py`: XML reading sessions, per-slice contours keyed by SOP UID; default `min_agreement=3`; exclude contourless/small non-nodules; mismatch with zero matching annotated SOPs raises. It writes per-scan `.npz` masks.

```sh
.venv/bin/python -m theia.data.lidc --lidc_root <tree> --min_agreement 3 --summarise_only --out <agreement.json>
.venv/bin/python -m theia.data.lidc --lidc_root <tree> --min_agreement 3 --masks_out <mask-dir> --out <agreement.json>
```

**Concrete integration gap:** `preprocess_masks.py` requires DICOM SEG and does not consume the `.npz` masks emitted by `lidc.py`. The converter's advertised next command is insufficient. Implement an explicit UID-bound mask/CT-to-row adapter or documented SEG conversion. Validate exclusion polygons, per-reader union and consensus semantics, missing readers and cross-scan grouping. `summarise()` counts how many readers marked anything on a slice; it is not a measurement of voxelwise consensus or usable patient yield. The docstring's 875 is not a supported current acquired/processed count.

### 4. NLST archive roster, preprocessing and held-out reconstruction pairs

Source: <https://www.cancerimagingarchive.net/collection/nlst/>, DOI <https://doi.org/10.7937/TCIA.HMQ8-J677>, source API `https://nlst.cancerimagingarchive.net/nbia-api/services/v1`; IDC mirror <https://portal.imaging.datacommons.cancer.gov/collections/nlst/>. Saved image license `OUT/TCIA_NLST_LICENSE.txt`: CC BY 4.0 plus TCIA use/citation conditions. Clinical outcomes are a separate CDAS access/DUA route at <https://cdas.cancer.gov/nlst/> and are not required for this label-free experiment.

`PCT/manifests/NLST_CANDIDATE_SUMMARY_v1.json` pins TCIA v3 (2021-09-24) and IDC v24 (idc-index-data 24.2.2). It records 26,254 CT people, 203,087 IDC series, 73,113 IDC studies; TCIA has 12 extra series. Metadata selects 26,185 people; 69 excluded. A later IDC page's 26,410 all-modality listing is a source-scope/version discrepancy to explain, not an acquired-cohort increment.

Canonical local metadata: `PCT/manifests/local/nlst_series_v1.parquet`, `nlst_persons_v1.parquet`, `nlst_roles_v1.parquet`; source selection `PCT/manifests/SELECTION_HIERARCHY_v1.json`; role/overlap `PCT/ROLES_PROTOCOL.json`, `PCT/OVERLAP_LEDGER.json`, `BW/public_ct/results/OVERLAP_UNCERTAINTY_v2.json`. Do not commit local IDs.

Roles: 10 engineering pilot, 250 unlabeled development, 1,000 held-out robustness, nested 1k/3k/10k pretraining, 2,000 replacement reserve, 1,092 human-annotation reserve; unassigned eligible 11,845 and ineligible role 57. Twelve non-selected people are in the human annotation reserve, explaining 69 versus 57; these counts must not be added as disjoint selected people blindly. Fixed salted ordering and replacement-after-QC-failure preserve nesting. All reconstructions of a person remain in one role. One shared study UID under two patient IDs is a recorded linkage uncertainty, not a proved duplicate; it is excluded from reconstruction pairing.

`OUT/NLST_DOWNLOAD_PLAN.json` source role SHA-256 `d1906de181c8ca0c6d17c0ee3fdddb922290110422901b0901e71bae5082e809`, series SHA-256 `65f348d0aa35b0ca66b3fc4318a8deb7ec64da34662ca2c51980f8fb81a072c6`; plan SHA-256 `746c7f24c8fb5cb39ee8a5d29130a3847d181082386562fa53b0b5cffe36971b`.

Fresh receipt/plan aggregation on October 15, with no archive reopening:

| Quantity | Value / meaning |
|---|---|
| Frozen P10k queue | 10,000 distinct plan people / 10,000 index series |
| Expected full-plan DICOM images/bytes | 1,695,648 images / 893,062,564,138 uncompressed bytes |
| Verified terminal receipt membership | 2,391 unique archives/person keys/series; 1 existing verified + 2,390 newly verified in that transfer |
| Actual retained verified ZIP bytes | 118,384,085,731 |
| Expected DICOM images/bytes represented by those archives | 402,029 / 211,743,268,760 |
| Remaining queue | 7,609 archives; expected 1,293,619 images / 681,319,295,378 uncompressed bytes |
| Nested subset archive coverage | P1k 1,000/1,000; P3k 2,391/3,000; P10k 2,391/10,000 |
| Transfer terminal state | stopped, no active archives, zero recorded failures, ended 2026-10-14T11:06:40Z |
| Pixel/derivative/training status | Transfer explicitly records no extraction or pixel decode and zero new evaluable EGFR patients. No accepted NLST derivative/training execution receipt found in inspected summaries. |

The archive verification checks CRC, exact image count/uncompressed size, all DICOM header identities, distinct SOPs and SHA-256. This audit re-counted their metadata receipts, not archive bytes. Full collection ≈11.14 TB and the one-series P10k plan ≈893 GB uncompressed are different quantities.

**Restart defect:** `OUT/download_nlst.py` has no argparse interface, hardcoded October 14 cutoff/deadline, and only permits an existing archive when it equals `NLST_FIRST_ARCHIVE_RECEIPT.json`. A naïve rerun after changing dates encounters `unbound_existing_archive` on already acquired files. Preserve the original script/receipts and implement a versioned receipt-aware successor. Its future CLI must not be invented in issues as an existing runnable command.

Archive destination: `data/acquisition_staging/2026-10-14_ct_egfr/nlst_archives`. Existing scientific ingester is separate: `PCT/code/ingest/{ingester.py,dicom_validate.py,plan.py,pilot_cli.py}`. Metadata tools: `build_nlst_manifest.py`, `assign_roles.py`, `build_overlap_ledger.py`, `sources_fetch.py`, `build_source_register.py`, `hf_tree_list.py`, `build_resource_envelope.py`.

Current parser-verified pilot interface:

```sh
.venv/bin/python PCT/code/ingest/pilot_cli.py plan --plan-out <local-plan.json> --inventory <allowance-inventory.json>
.venv/bin/python PCT/code/ingest/pilot_cli.py release-template --plan <local-plan.json> --inventory <allowance-inventory.json> --out <new-pilot-dir>
.venv/bin/python PCT/code/ingest/pilot_cli.py verify-release --release-file <release.json> --plan <local-plan.json> --inventory <allowance-inventory.json> --out <new-pilot-dir>
.venv/bin/python PCT/code/ingest/pilot_cli.py run --release-file <release.json> --plan <local-plan.json> --inventory <allowance-inventory.json> --out <new-pilot-dir>
.venv/bin/python PCT/code/ingest/pilot_cli.py report --out <new-pilot-dir>
```

Expand `PCT` to its path key before use. The selected newer plan is `HX/public_ct/p04/local/pilot_plan_v2_local.json`; allowance inventory `HX/public_ct/p04/PILOT_ALLOWANCE_INVENTORY.json`; historical output proposal `HX/public_ct/p04/pilot_run_v1`. `PCT/pilot_run` is not an output to overwrite; old bare-boolean release commands are stale. Refresh authority/resources and manifests rather than reusing expired release facts. Engineering pilot role is separate from P10k: do not borrow P1k people as undocumented pilot subjects. The v2 pilot plan has 10 people and 13 series (10 index plus 3 alternate reconstructions), expected uncompressed source bytes 1,132,427,904; saved allowance inventory has zero consumed people/transfer bytes and a cumulative 10-person/2,000,000,000-byte transfer and footprint cap, subject to refreshed evidence. The two-GB cap does not mean all raw plus extracted data can fit simultaneously.

Known intended preprocessing: 16 float16 224×224 slices per person, label-free thoracic views with HU/orientation/geometry validation and no fabricated tumor masks. Fixture-based derivative size is 1,606,584 bytes/person (P10k ≈16.07 GB; upper allowance ≈16.39 GB), not a real NLST yield measurement. Development metadata bytes ≈22.47 GB. Held-out planned index+chosen alternate bytes ≈187.68 GB, 356,357 slices; fixture derivatives ≈3.21 GB. Compressed transfer ratio, real decode throughput, peak extraction/storage and real retained yield need fresh measurement. Historic forecasts assuming deletion of new raw staging are proposals, not permission to delete the archives now retained.

Reconstruction design: 1,000 people / 1,052 technique-equal candidate pairs in BW's more detailed pairing table; metadata evidence does not prove same acquisition. Header capture must resolve FrameOfReferenceUID, acquisition number/time, orientation, coverage, spacing, kernel and temporal consistency. Pair failure is a failure/coverage record, not a silent replacement. `BW/public_ct/scripts/recon_feature_metrics.py` consumes same-acquisition checks and computes patient-averaged cosine/norm contrasts. The existing synthetic benchmark's 200-step allowance was already consumed; do not classify re-running it as an unfinished data task. Training runner work belongs to the training audit.

### 5. TCGA-LUAD with Crowds Cure 2017 and GDC/cBioPortal endpoint provenance

Sources: <https://www.cancerimagingarchive.net/collection/tcga-luad/>, <https://www.cancerimagingarchive.net/wp-content/uploads/TCGA-LUAD-Clinical-Data-1516.zip>, <https://api.gdc.cancer.gov/>, <https://www.cbioportal.org/>. Public measurement release uses TCIA upload `CrowdsCureCancer2017Annotations.csv` and `CrowdsCureCancer-dicomsrfiles_20180830.zip`; `ACQ/pilot/acquire_tcga_luad.py` records the concrete support URLs and CC BY 3.0 basis.

Code: `theia/data/acquire.py`, `external_adapter.py`, `lesion_consensus.py`; `ACQ/pilot/acquire_tcga_luad.py`, `pilot_qc.py`, `build_external_inputs.py`; `ACQ/molecular/build_label_manifest.py`, `scripts/gdc_retrieve.py`, `cbioportal_retrieve.py`, `build_per_subject_evidence.py`, `aggregate_counts.py`; metadata reproducibility `ACQ/metadata_repro/verify_public_metadata.py`.

Local raw/metadata/arrays: `data/external/tcga_luad/`, `data/external/crowds_cure_2017/`; `ACQUISITION_LEDGER.jsonl`, `DEVELOPMENT_SAMPLE.json`, `qc/`, `processed/<recipe>/rows.jsonl` and `.npz`. Separate later reader inputs remain under external reader worklist storage, not copied into public Git.

Actual positional-command interfaces:

```sh
.venv/bin/python ACQ/pilot/acquire_tcga_luad.py support
.venv/bin/python ACQ/pilot/acquire_tcga_luad.py dev
.venv/bin/python ACQ/pilot/acquire_tcga_luad.py annotated
.venv/bin/python ACQ/pilot/build_external_inputs.py dev [--overlays]
.venv/bin/python ACQ/pilot/build_external_inputs.py all
```

These acquire/build existing routes; a new reader-derived input version needs its reviewed consumer rather than silently changing the old `all` output. The external build reads geometry/measurement/QC and writes label-free rows, resolution table and failure ledger. It must not read EGFR to choose a lesion.

67 CT subjects partition into 8 research positives (2 also clinical-assay-positive) + 51 research negatives + 8 unknown. The 40 evaluated are 4 positive/36 research negative; 4 additional adapter-development subjects are excluded from evaluation. Cohort flow: 11 without resolvable lesion, 2 timing-ineligible, 2 input/air failures, 4 development and 8 endpoint unknown. This is a conditional research endpoint, not clinical assay equivalence. The misleading legacy `egfr_mutation_status` header maps to whether testing was performed; YES/NO there must never become mutation labels. Whole-genome amplification and open-MAF callability limits remain in endpoint provenance.

Crowds Cure provides line measurements rather than contours. Existing rule: cluster midpoint measurements; finding with most annotators (minimum 2); medoid measurement. Do not present algorithmic consensus as an adjudicated tumor contour. The 40 evaluated use 264 measurements (five to seven per patient). No independent contour Dice can be computed from these lines without a new reference.

Current candidate capacity from latest closeout: at most 13 never-scored TCGA candidates after accepted reader/input work, with 2 known positives across that candidate set; not a protected blind cohort because outcomes are known. Eight-case starter could yield at most 3 newly label-evaluable TCGA cases (1 positive). Acquisition/storage totals for all current TCGA arrays were not re-inventoried; retrieve aggregate ledger bytes before cost estimates. Do not use whole TCGA molecular enrollment as an imaging cohort size.

### 6. CPTAC-LUAD images, research sequencing and deferred label join

Sources: <https://www.cancerimagingarchive.net/collection/cptac-luad/>, <https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac>. Image acquisition code records CC BY 4.0, public TCIA REST. Molecular access is file-specific: open released WXS masked MAFs were selected; one controlled targeted-sequencing file was excluded. Public pathology counts are not radiology counts.

`ACQ/cptac/acquire_cptac_luad.py` stages `series`, `headers`, `select`, `fetch`, `qc`, `summary` or `all`; positional parser uses `sys.argv`, not argparse. It surveys CT series headers, selects one candidate index CT by label-free metadata/timing rule, then selectively downloads/QCs. Outputs: `data/external/cptac_luad/metadata/getSeries_CPTAC-LUAD_CT.json`, `headers/header_table.csv`, `index_selection.csv`, `series_eligibility.csv`, image tree and `ACQUISITION_LEDGER.jsonl`; aggregate `ACQ/cptac/cptac_index_series.json`.

```sh
.venv/bin/python ACQ/cptac/acquire_cptac_luad.py summary
# Acquisition stages exist; running `all` would execute data operations:
.venv/bin/python ACQ/cptac/acquire_cptac_luad.py all
```

29 index CTs were acquired (older register ≈8.0 GB under 12 GB acquisition cap); 28 remain pre-reader eligible. This does not mean 28 usable labeled arrays. The imaged catalogue/aggregate discrepancy 38 / 37 / 36 must remain explicit until source-release reconciliation.

Molecular downloader `OUT/download_open_cptac.py` has no parameterized CLI and consumes `OUT/GDC_CPTAC_IMAGED_FILES_CURRENT.json`; it writes only compressed files to `data/acquisition_staging/2026-10-14_ct_egfr/gdc_open_maf_quarantine`, per-file receipts under `OUT/gdc_receipts/`, and summaries. The final later-retry summary is `OUT/GDC_DOWNLOAD_SUMMARY_FINAL.json`. Status: 36/36 selected public files verified, 3,335,010 bytes; metadata SHA-256 `1ddc979a98429331e2af9a7e78c4be56f58cc98a499b72492b43ca09400c3871`; three initial connection resets were retained and later retried once. No content decoded, no EGFR join, no scientific acceptance. Do not rerun successful acquisition or open quarantine as a discovery task.

Label-bearing files have been local since September 21, before selection lock; analyst blindness is not established. Required sequence is reader-derived location/input acceptance, declared model/prediction lock, then the agreed L5 label decode/join and endpoint qualification. Report CPTAC as not demonstrably blind; no retrospective hash repairs historical exposure. Keep positives, research negatives, unknown/unassessable and conflicts separate and preserve specimen/timing/assay evidence.

### 7. RIDER Lung CT and separate annotation release

Sources: <https://www.cancerimagingarchive.net/collection/rider-lung-ct/>, CT DOI <https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR>, annotation DOI <https://doi.org/10.7937/tcia.2020.jit9grk8>. Saved acquisition code says CC BY 4.0/open; retain both source citations and check annotation-specific terms.

Code: `ACQ/rider/acquire_rider.py`, `build_rider_inputs.py`, `run_rider.py`, `theia/data/rtstruct.py`, `theia/data/external_adapter.py`. Raw/manifest: `data/external/rider_lung_ct/metadata/getSeries_all.json`, `pairs.json`, imaging and `ACQUISITION_LEDGER.jsonl`; derived `processed/rows_TEST.jsonl`, `rows_RETEST.jsonl`; aggregate `ACQ/rider/RIDER_BUILD.json`, protocol `RIDER_PROTOCOL.md`. Historical analysis `revamp/campaign/scientific_closure_2026-09-23/rider/RIDER_CORRECTED.json`.

```sh
.venv/bin/python ACQ/rider/acquire_rider.py rtstruct
.venv/bin/python ACQ/rider/acquire_rider.py ct
.venv/bin/python ACQ/rider/build_rider_inputs.py
```

Acquisition takes the CT series explicitly referenced by each scan's RTSTRUCT; builder rasterizes its manually named `GTVp_<scan>_man` ROI on that CT and emits label-free arrays. Test and retest remain paired within the same person. Historical arrays/analysis are already present: 28 pairs/56 scans. Repeated reconstructions or the collection's many SEG/SR objects do not increase this count. No registered common frame was established, so do not imply voxelwise map correspondence across scans. No EGFR labels exist for the present study role. Full current storage bytes were not verified here; use existing ledger inventory rather than download again.

### 8. Other genuine public candidates and related resources

These are discovery/access/preprocessing tasks conditional on a defined experiment; none is a mandatory blanket download. Counts are recorded source descriptions, not new THEIA participants.

| Resource / official recorded URL | Recorded scale/access and role | Concrete prerequisite before any data use |
|---|---|---|
| CT-RATE — <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE> | 21,304 people; 25,692 scans; 50,188 reconstructed volumes; gated own-account terms; historical register CC BY-NC-SA 4.0. Alternative representation/report resource, no EGFR. | Record own access; preserve publisher validation split; enumerate original vs reconstruction groups, metadata/corrections and source checksum; benchmark selected source volumes. It is an alternative/factor, not silently mixed into NLST scale points. |
| RadGenome-Chest CT — <https://www.nature.com/articles/s41597-025-05922-9> | README advertises 665K grounded reports, not 665K people; no genomic endpoint. No verified acquired or usable count established in this audit. | Resolve source dataset relation/overlap, actual access/license, annotation type and compatible loader. README/config intent is not proof of executed pretraining. |
| TCGA-LUSC and TCGA H&E whole-slide images — GDC <https://portal.gdc.cancer.gov/> and TCIA catalogue <https://www.cancerimagingarchive.net/collections/> | Named only in the older README's future multimodal program; no acquired/usable CT–WSI–molecular join or executed pathology branch established here. TCGA-LUAD is used as external CT evaluation in the current study despite the old README calling it internal. | Treat as a legacy prospective scope item requiring an explicit scientific decision. If retained, enumerate actual CT/WSI/specimen/endpoint intersection, resolve slide/sample/person hierarchy, access/license and train/test exposure, and build a separate modality loader. Do not silently repurpose the exposed TCGA external cohort for training or count a whole genomic project as paired imaging. |
| LNDb — <https://zenodo.org/records/8348419> | 294 scans, patient uniqueness unverified; historical licensing conflict/CC BY-NC-ND conditions flagged; auxiliary nodule annotations. | Pin release, confirm permitted transformations/redistribution, deduplicate persons, build annotation/CT adapter only for declared localization question. |
| LUNA25 — <https://luna25.grand-challenge.org/> | 4,000+ examinations across challenge description, public development vs restricted hidden tests; no EGFR. | Acquire permitted manifest only, map NLST/other source overlap, preserve hidden/publisher splits and independent evaluation restrictions. |
| NSCLC-Radiomics-Interobserver1 — <https://www.cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/> | 22 people; repeated contour information for 21; historical corrected v3. | Resolve overlap with LUNG1, select corrected release, map readers/ROIs to CT; annotation uncertainty analysis, not new molecular count. |
| NSCLC-Radiomics-Genomics — <https://www.cancerimagingarchive.net/collection/nsclc-radiomics-genomics/> | 89 subjects; expression arrays rather than verified mutation calls; historical CC BY-NC 3.0. | Confirm any actual assay endpoint and linkage; otherwise only expression/biology feasibility; no expression-to-EGFR relabeling. |
| Lung-PET-CT-Dx — TCIA collection catalogue <https://www.cancerimagingarchive.net/browse-collections/> | 355 source subjects; no verified EGFR table. | Resolve exact release and terms, distinguish diagnostic chest CT from attenuation-correction CT, eligibility and overlap. |
| QIN LUNG CT — TCIA catalogue <https://www.cancerimagingarchive.net/collections/> | 47 people; historical CC BY 3.0; auxiliary acquisition variability. | Pin exact source release and pair/group definitions; distinct question beyond already completed RIDER before acquisition. |
| Crowds Cure 2018 — TCIA annotation release via <https://www.cancerimagingarchive.net/collections/> | 88 already-internal people measured; historical CC BY 3.0. | Resolve actual release URL, CT/annotation UID linkage and extent/line semantics; no additional persons. |
| BAMF / AIMI — TCIA/IDC analysis-result catalogues | AI masks, including some TCGA/NLST derivatives; historical CC BY 4.0. | Keep model provenance/exposure and human-vs-AI reference separation; no detector route is established just by downloading masks. |
| I3LUNG — <https://www.nature.com/articles/s41591-026-04488-2> | Public extracted features; original CT not released in inspected record. | Source access recheck; features alone cannot support THEIA spatial image experiment; CT–EGFR intersection remains unknown. |
| PhysioNet sepsis chest CT — <https://physionet.org/content/chest-ct-sepsis-er/1.0.0/> | 836 scans/327 admissions in inspected page; restricted own-account and contributor review; not a lung-cancer EGFR source. | Only a justified domain-shift study; resolve people vs admissions, actual CT release/access and endpoint. Credentialing alone supplies no data. |
| IDC — <https://datacommons.cancer.gov/repository/imaging-data-commons> | Discovery/mirror infrastructure, not another cohort. | Map accession/release/UID aliases to TCIA; do not count duplicates twice. |
| AACR GENIE | Genomic consortium mentioned as a possible custodian route; no paired CT release or accessible patient count established here. | Imaging linkage and permissions must be established by custodian; no ready downloader or eligibility count. |

### 9. Request-only or restricted molecular cohorts

Canonical provenance: `planning/rigor_revamp_2026-09-16/07_DATA_RESOURCE_REGISTER.csv`, `planning/ct_egfr_outreach_2026-10-04/CONTACT_REGISTER.csv`, `SOURCE_EVIDENCE.json`, `OUT/OUTREACH_AND_ACQUISITION.md`. Published cohorts overlap; do not sum this table. All accessible/eligible/committed counts are unverified or zero acquired in inspected records. Existing inquiry drafts are unsent; publication of a backlog does not authorize outreach. Keep contacts and message IDs out of the dataset catalogue unless needed and explicitly approved.

| Candidate | Recorded published scale / source | Remaining operation before compute |
|---|---|---|
| FAIS network | 18,232 across 9 published cohorts; <https://pubmed.ncbi.nlm.nih.gov/35341713/> | Identify actual custodian/site subsets and overlap; negotiate permitted data transfer or custodian-run evaluation; confirm eligible pretreatment CT/assay-negative definitions. |
| West China FCPNet | 3,433 published; <https://papers.miccai.org/miccai-2025/paper/2645_paper.pdf> | Access not verified; check overlap with FAIS and other West China papers, raw volumes vs features, eligibility and assay metadata. |
| Ajou–Chungnam–Inha | 1,280 published; <https://www.nature.com/articles/s41598-024-51630-6> | Reasonable request route, no access obtained; histology/timing/volume availability and split reconciliation. |
| AI-Hub catalogue | 10,000-patient catalogue cited; <https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71394> | Restricted eligibility/access and actual CT–EGFR intersection; not 10,000 verified usable pairs. |
| Shanghai Chest | 1,010 published; <https://pmc.ncbi.nlm.nih.gov/articles/PMC7500487/> | Current custodian, sharing/compute route, overlap and temporal/molecular eligibility. |
| IEO | Relevant published molecular study; <https://pmc.ncbi.nlm.nih.gov/articles/PMC10527057/> | Current custodian feasibility; independent person/site counts, raw images, tested-negative definition and permissions. No verified accessible count. |
| Guangzhou–Shengjing | 914 (709+205) published; <https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2020.598721/full> | Both hospitals' ethics/access decision; compatible images and molecular/clinical completeness. |
| Suqian–Suzhou | 826 published; <https://pubmed.ncbi.nlm.nih.gov/40328536/> | Availability of raw volumes rather than features, source sites, endpoint and terms. |
| Soochow–Changzhou PET-CT | 516 published; <https://link.springer.com/article/10.1186/s12880-024-01232-5> | Conflicting access statements; clarify CT type, specimen/timing and permissions. |
| Shanghai Pulmonary–Tianjin | 844 published; <https://pmc.ncbi.nlm.nih.gov/articles/PMC6437603/> | Custodian/contact current status, source overlap and compatible eligible raw data. |
| Huadong | 579 local published; <https://pmc.ncbi.nlm.nih.gov/articles/PMC6601587/> | Separate local people from reused public TCIA external cases; no access assumed. |

If a custodian permits evaluation, reuse `revamp/campaign/parallel_expansion_2026-10-02/molecular/custodian_package/SPEC.md` and `validate_custodian_package.py` as starting code. Need a real agreed schema, identity/assay/time fields, person-level roles and exposure lock; synthetic fixtures do not validate the real incoming cohort. No worldwide cohort ceiling is inferred from these inspected candidates.

## Task mapping

The canonical queue is [TASKS.md](TASKS.md) / [TASKS.json](TASKS.json). Dataset-specific details above supplement those cards; they do not create a second competing queue. Existing local pilot/release interfaces carry historical orchestration fields. E01/M02/E05 may replace retired controller coupling while preserving real role, identity, resource and input checks; never fabricate an old approval receipt.

## Audit limits and unresolved items

- The mounted filesystem was slow. Reads were targeted and later completed; no exhaustive data-directory recursion or full file-integrity rehash was attempted.
- Current acquisition state was verified from canonical summary/plan/receipt metadata. Existing CRC/header/hash assertions were not independently replayed on archives.
- Exact present LIDC acquisition/usable count, complete per-source storage totals outside NLST/CPTAC receipts, and any newer work outside the inspected canonical folders remain unknown.
- Historical public-source license facts and URLs need a dated official refresh before new use. The saved NLST license was directly inspected; other licenses are attributed to saved project records/code rather than represented as fresh legal determinations.
- No molecular quarantine content was parsed. CPTAC event counts remain unknown for the eligible/readable future subset.
- Old prompts contain obsolete access, raw-data absence, timing, authority and worldwide-scale statements. This audit preserves their useful source/code pointers and explicitly avoids carrying those obsolete assertions into current counts.
- A successfully downloaded archive, passing test, generated manifest or readable array is an execution milestone; none alone establishes a supported scientific conclusion.
