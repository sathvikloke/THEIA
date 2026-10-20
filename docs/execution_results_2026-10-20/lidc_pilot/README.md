# LIDC source inventory and real adapter pilot

Date: 2026-10-20. This record contains aggregate engineering evidence only.
Source images, XML identities, patient-level inventories, and processed rows
remain in the private external workspace.

The reviewed v2 LIDC adapter passed a real, matched-source pilot from XML and
native CT through consensus export, row transformation, and the actual Torch
dataset/collator. The original research checkout was not overwritten. No model
was trained and no clinical or molecular label was assigned.

## Public source and bounded acquisition

The [official LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)
provides the source under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
Attribution: Armato III, S. G., McLennan, G., Bidaut, L., et al. (2015).
*Data From LIDC-IDRI* [Data set]. The Cancer Imaging Archive.
[DOI: 10.7937/K9/TCIA.2015.LO9QL9SX](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX).
The collection access table is Version 4, updated 2020-09-21.

Acquisition used the documented
[public NBIA REST API](https://wiki.cancerimagingarchive.net/display/Public/NBIA%2BSearch%2BREST%2BAPI%2BGuide).
The XML-only archive was 9,034,311 bytes. One CT series transferred as a
34,609,083-byte ZIP, containing 133 DICOM objects totaling 70,018,838 bytes.
All 133 objects matched TCIA's supplied MD5 values. The public XML/CT
StudyInstanceUID, SeriesInstanceUID, and every annotated SOP were matched
before decoding. No other CT series or clinical spreadsheet was acquired.

## D06 real-source result

| Stage | Exit | Wall seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Native consensus export | 0 | 507.682 | 685,408,256 |
| Row conversion | 0 | 45.058 | 620,380,160 |
| Actual Torch dataset and collator | 0 | 2.953 | 354,811,904 |

The native grid was 133 × 512 × 512 with spacing
0.703125 × 0.703125 × 2.5 mm. The unchanged threshold of at least three distinct
readers produced 4,812 positive voxels across eight native slices.
The source had four local reader sessions and nine annotated SOPs.
These are different units, not counts of unique lesions or evaluable patients.

One source-bound row contained float32 images and binary ROI arrays of
16 × 224 × 224, with 16,054 positive ROI pixels after the existing transform.
The sampler used eight distinct native slices and padded the last slice to
16 tensor slices. Exact selected SOPs and crop boxes were retained.
The actual CPU dataset item was 16 × 1 × 224 × 224 and the collated batch was
1 × 16 × 1 × 224 × 224. Both configured molecular targets remained unknown
(`-1`); the row's label dictionary and report were empty.

The pilot used the existing Python 3.13.9 / Torch 2.10.0 environment, with
one thread and external temporary/output paths. No packages or models were
installed. The header-derived peak estimate was 965,738,496 bytes; each stage
remained below it. A process sample confirmed substantial SimpleITK/GDCM read
and seek waits during export. The faster second pass does not establish a
stable cohort-wide throughput estimate.

A private, nonclinical QA figure was generated from saved arrays. Its recorded
native-mask crop/resize exactly reproduced the saved ROI; hashes of both
input artifacts were unchanged. No source CT reread or clinical adjudication
was performed for that figure.

## D05 source inventory

The current official CT metadata contains 1,010 subjects, 1,018 studies,
1,018 CT series, and 243,958 CT images, totaling 128,405,916,114 DICOM bytes.
These are source inventory counts, not an adopted training cohort.

The archive contains 1,036 CT XML members spanning those 1,018 CT series,
plus 283 explicitly marked chest-radiograph XML files. The CXR records are
kept separate from CT polygon annotations.

Seven CT series have multiple XML byte representations. Six have identical
method-relevant contour inputs after normalizing only reader/nodule ordering.
Their raw hashes are retained and a deterministic representative is bound.
For the remaining series, the official 2012-03-21 source correction identifies
the superseded version; the downloaded official audit-only old XML is
byte-identical to the old archive member. The corrected member is selected
with that evidence, without choosing by lesion size or downstream performance.

After this source reconciliation, 883 CT series have matched XML with included
polygon ROIs and 135 have matched XML without included polygon ROIs.
Included polygons do not establish a nonempty three-reader mask or a usable
training row. One selected XML has eight reading sessions and is flagged for
source resolution before the exporter’s at-most-four-reader contract.
Eight people have two CT timepoints; both remain in the source inventory.

Only the one pilot series has actual CT/SOP/grid validation. The other
1,017 series, totaling 128,335,897,276 official DICOM bytes, were not present
in the bounded local inventory. This is an explicit acquisition-planning
queue, not a claim that all mounted storage was searched. No further CT
download was launched.

## Method, commands, and limits

The archive schema is `theia.lidc.consensus.v2`; the required method identifier is
`distinct-reader-vote/nodule-inclusion-minus-exclusion/strict-polygon-interior/v2`.
Nodule inclusions and exclusions are resolved before reader union and voting.
Strict polygon interior excludes included-outline pixels. The adapter checks
source identity and physical geometry, refuses empty or unmatched sources,
and never sends NPZ masks through the DICOM SEG parser.

Portable implementation and command documentation:
[LIDC_ROWS.md](../../LIDC_ROWS.md).
The private execution driver ran these stages sequentially:

```text
python -u run_real_pilot.py export
python -u run_real_pilot.py rows
python -u run_real_pilot.py dataset
```

Each stage asserted the reviewed code paths/hashes and recorded its actual PID,
argv, timestamps, input hashes, environment, exit code, and peak RSS.
The implementation hashes used were:

```text
theia/data/lidc.py
b574e557b8e23f52cb7c5e7c1b191823530d036d5ef4cfcf6619b0d79d28ac3f
theia/data/lidc_rows.py
d71a7284e8f7614370ba660111baf2297a5be973fac9a750959f343b8aa39e3c
```

The pilot closes the real matched-source and actual Torch-loading gaps of D06.
It contains no exclusion polygon, so exclusion-hole behavior on real sources
remains untested beyond the earlier synthetic XML fixtures and independent
geometry oracles. One prepared person does not add an EGFR-evaluable case.
D07 cohort adoption still requires source coverage, person-level series and
partition decisions, and recorded source/geometry and manifest checks. No split or
clinical eligibility threshold was invented here.
