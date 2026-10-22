# Figure 1 — corrected historical cohort flow

P01 now has a concrete [vector SVG](FIGURE1.svg) and [3200 × 3280 PNG](FIGURE1.png), generated from the retained closed `journal_set_v3` review evidence. [Caption and source notes](CAPTION.md) accompany the image.

The internal flow is **211 → 188 → 158 → 153**. The image displays the unknown selection reason, reconstructed air-rule exclusions and approximate slice ordering. External molecular evaluation, RIDER test–retest and CPTAC preparation remain separate. This is corrected reporting of historical M0 inputs, not a corrected-input replication result.

**Status:** technical figure generation complete; publication adoption remains pending student/author review. The original paper and historical figure are not replaced. This artifact makes no submission-readiness or student-initial-authorship claim.

## Source and checks

- [Aggregate source JSON](SOURCE_DATA.json) and [CSV](SOURCE_DATA.csv): source anchors, count hierarchy and uncertainty classifications.
- [Exact retained excerpts](SOURCE_EXCERPTS.md): corrected Figure 1 caption, internal Table 1, attrition paragraph, external definitions and preparation status.
- [Verification receipt](VERIFICATION.json): arithmetic, visible count bindings, critical qualifiers, PNG dimensions, local links and retained source hashes.
- [Visual review](VISUAL_REVIEW.md): review of the rendered image and remaining limits.

Source files were read from the retained campaign `theia_campaigns/2026-10-13_nine_hour_210100`. JSON records paths relative to that campaign and full-file SHA-256 hashes. The latest closed `journal_set_v3` is selected as the author-review source. The retained readiness seal and earlier consumed-source index preserve the link to R01, whose raw patient-level package was not available during this work. No attrition is newly inferred from patient records. The original preprocessing logs are unavailable; the four air-rule causes remain reconstructed and three use approximate IPP order. The retained excerpts do not identify GDCM by name, so the image does not assert an exact historical GDCM order.

## Reproduction

From this directory, using Python 3 and an existing Node `sharp` installation:

```sh
python3 build_figure1.py
NODE_PATH=/path/to/existing/node_modules node rasterize.mjs
python3 verify_figure1.py --source-root /path/to/2026-10-13_nine_hour_210100
```

`build_figure1.py` uses the committed JSON without accessing clinical data or the original campaign. `--capture-sources /path/to/2026-10-13_nine_hour_210100` explicitly recaptures the named source excerpts and aggregate transcription; source-content guards fail if the selected text no longer matches. Rasterization uses vector source at 2× resolution, not an enlarged previous raster. No packages were installed.

The generator and count checks use Python's standard library. The local rasterization used Node's bundled `sharp` 0.35.5 / libvips 8.18.7. The PNG is a review image; final publisher-specific size, file format, typography and placement remain production choices.

## Contribution and boundary

On 20 October 2026, Codex transcribed retained aggregate evidence, generated the flow diagram and its reproducible source, linked the existing caption, and checked the resulting artifacts. No new model fit, inferential statistic, molecular label join, clinical adjudication, reader result or initial IJAS prose was produced. Student/author interpretation and publication adoption remain human decisions.
