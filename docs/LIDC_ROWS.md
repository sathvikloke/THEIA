# LIDC consensus masks to training rows

`theia.data.lidc_rows` consumes the versioned NPZ archives exported by
`theia.data.lidc`. It writes the existing `rows.jsonl` / `images` / `roi` NPZ
contract consumed by `RadiogenomicsDataset`. Rows have `labels={}` and
`report=""`: these are grounding inputs, with no molecular or report labels.

The consensus method remains a **per-voxel count of distinct readers**, default
`min_agreement=3`. No nodule diameter or minimum voxel threshold is added.
In particular, the clinical pipeline's `data.min_tumor_voxels` does not apply.
This is a description of the existing mask rule, not a claim that it reproduces
any other dataset's clinical nodule eligibility protocol.

## Source binding

The old export stored only `mask`. Such archives are rejected because neither
their file names nor their dimensions prove which CT slices they belong to.
Re-export their original XML against the matched CT; do not fabricate metadata
for legacy masks.

New archives contain a binary `[z,y,x]` `mask` and a scalar Unicode
`metadata_json` (loaded with `allow_pickle=False`) with:

| Field | Meaning |
|---|---|
| `schema`, `mask_axes` | `theia.lidc.consensus.v2`, `zyx` |
| `patient_id` | DICOM PatientID, retained for patient-level membership |
| `study_instance_uid`, `series_instance_uid`, `frame_of_reference_uid` | Original CT identity |
| `sop_instance_uids` | All CT SOPInstanceUIDs in the exact volume slice order |
| `shape_zyx`, `spacing_xyz`, `origin_xyz`, `direction` | Native CT grid; direction is the row-major SimpleITK 3×3 matrix |
| `annotated_sop_instance_uids`, `reader_indices` | Included/excluded polygon source membership and XML session indices |
| `min_agreement`, `consensus_method` | Reader threshold and exact versioned method ID |
| `source_xml`, `source_xml_sha256`, `source_ct_dir` | XML provenance and original source paths |

`consensus_method` must equal
`distinct-reader-vote/nodule-inclusion-minus-exclusion/strict-polygon-interior/v2`.
Missing or different method IDs fail even when the numeric reader threshold
matches. Both mask-only exports and the initial v1 implementation require
re-export; v1 did not bind the method and inherited two XML interpretation bugs.

Version 2 preserves each contour's reader, nodule and inclusion flag. For each
reader's nodule, it unions included polygon interiors and subtracts excluded
polygons, including their boundary. It then unions that reader's nodules and
counts distinct readers per voxel. A hole in one nodule cannot erase an included
pixel from another nodule. The XML exclusion semantics are documented by the
[pylidc Contour API](https://pylidc.github.io/contour.html) and implemented per
annotation in [pylidc's mask builder](https://github.com/notmatthancock/pylidc/blob/master/pylidc/Annotation.py).

LIDC contour points are outside the nodule, as described in the annotation
methods and Figure 2 of [Armato et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3041807/).
The rasterizer therefore uses deterministic even-odd ray crossing at integer
pixel centers with explicit on-segment detection: included regions omit their
border and excluded regions remove both interior and border. The result is
independent of contour winding and repeated first/last vertices. This corrects
the old `matplotlib.Path(..., radius=0.5)` border inclusion. The rasterizer uses
NumPy only, on the clipped polygon bounding box. These source-interpretation
corrections change masks; they do not change the `>=3` reader threshold.

The exporter selects exactly one series matching **all** polygon SOPs.
It uses one GDCM-ordered file list for both headers and pixel loading, checks
patient/study/series/frame consistency, and compares every DICOM position,
orientation, and pixel spacing to the loaded physical grid (absolute tolerance
`1e-4`, in each field's units). Missing/duplicate SOPs, partial XML matches,
nonuniform grids, and multi-frame CT fail explicitly. There is no nearest-slice
fallback. The adapter requires that the source still has the same ordered SOP
list, dimensions, and geometry.

## Bounded pilot

Use an environment with the repository dependencies, and output paths with
sufficient space. Run from the repository root:

```sh
python -m theia.data.lidc \
  --lidc_root /absolute/path/to/verified/LIDC-series-directory \
  --min_agreement 3 --limit 1 \
  --masks_out /absolute/path/to/new/lidc-pilot-masks \
  --out /absolute/path/to/new/lidc-pilot-agreement.json

python -m theia.data.lidc_rows \
  --config configs/default.yaml \
  --masks_dir /absolute/path/to/new/lidc-pilot-masks \
  --out_dir /absolute/path/to/new/lidc-pilot-rows \
  --expected_min_agreement 3 --limit 1
```

The export command assumes the CT series is beside its XML, as before. For an
XML stored separately from its source CT, call the explicit API:

```python
from theia.data.lidc import export_mask_for_scan
export_mask_for_scan("/verified/annotation.xml", "/verified/CT",
                     "/new/pilot-masks", min_agreement=3)
```

If CT directories moved, pass `--source_map /absolute/path/to/source_map.json`,
where the JSON maps each original SeriesInstanceUID to its current CT directory.
The map must cover every selected archive. New paths do not bypass identity or
geometry verification. `--limit N` selects the first N sorted archives/XMLs;
it establishes a bounded engineering pilot, not representative cohort sampling.

The adapter refuses a nonempty output directory and never overwrites artifacts.
It publishes `rows.jsonl` only when all selected inputs succeed. A failure after
earlier rows have been written may leave NPZs without an index; inspect those
outputs and rerun into a fresh output directory. A nonzero exporter exit also
means at least one selected XML failed, even if others produced valid archives.

## Transform and membership checks

Images and masks use the actual `preprocess.window_hu`, `tumor_slices`,
`roi_box`, `crop_box`, and `_resize` helpers. Each pair shares its crop box,
padding, and resize; masks retain the existing linear interpolation followed by
`>0.5` threshold. As in the existing preprocessor, the pixel margin is computed
from x spacing. The adapter records all native spacing values rather than
pretending to resample anisotropic CT. Crop jitter uses a SHA-256-derived patient
seed, stable between Python processes. An empty target after resizing is an
explicit failure.

Each row retains the archive SHA-256, source metadata, selected CT indices and
SOPs, crop boxes, and transform parameters. Multiple selected archives for one
PatientID fail before any output is written: resolve patient/series membership
explicitly before cohort construction. This avoids feeding duplicate patients
to the existing one-row-per-person split helpers. No train/validation/test split
is chosen by this adapter. The existing `max_area` slice helper produces the
wrong count for odd `n_slices`; the adapter refuses that case instead of writing
a misleading row count.

## Verification boundary

`tests/test_lidc_rows.py` uses tiny synthetic volumes and DICOM reader/header
doubles to check slice order, an oblique physical grid, edge padding, the real
image/mask transform, manifest schema, provenance, and refusals. Export and
polygon tests exercise the real parser, rasterizer and consensus: holes,
separate nodules, reader counting, diagonal borders, concave contours and
reversed/closed outlines. All existing `tests/test_lidc.py` regressions remain
applicable. The dataset integration test runs only when Torch and scikit-learn
are available.

These checks do not replace a real matched XML/CT pilot. Inspect the native mask
overlay and at least one processed image/mask pair from that pilot before
constructing the cohort. The D06 completion gate remains open until that source
pilot is run and its geometry and identity evidence is recorded.
