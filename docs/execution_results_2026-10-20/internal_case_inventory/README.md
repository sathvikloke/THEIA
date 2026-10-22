# Internal development case inventory — October 20, 2026

The metadata assessment accounts for all **19 previously identified unbuilt
development cases**. None is ready for an automatic rebuild on the evidence
reviewed. The current corrected-input cohort remains **158 cases**; all 19
inventory cases are outside that cohort and its 16-case repair registry.
No scientific eligibility decision or independent external-person credit was
made.

The assessment first consumed the newer October 16 technical acceptance:
38 of 38 checks passed with fixture mode disabled. Its current bound checker,
reviewed registry, manifest and row metadata still match the accepted record.
The existing cohort-attrition reconstruction was reused. No acceptance,
attrition calculation or historical air-rule replay was repeated.

| Remaining source requirement | Cases |
|---|---:|
| No local AIM annotation or original CT+SEG download: locate an existing source annotation and bind its referenced CT. | 13 |
| AIM reference absent from the saved CT catalogue: resolve the source reference; the original import-selection reason remains unknown. | 3 |
| Wrong imported series, with the reference CT catalogued but absent from both saved download records: locate or acquire the identified reference series, then technically validate it. | 1 |
| Wrong imported series, with the reference CT absent from the saved catalogue: resolve the source reference before proposing a rebuild. | 1 |
| Direct-point exclusion under the existing air rule: retain the documented exclusion; metadata does not justify another point or threshold. | 1 |
| **Total** | **19** |

Current local observations show **3 accessible imported CT directories,
0 imported SEG directories and 6 AIM files**. Two of those AIM files reference
CT series present in the saved catalogue. These availability observations
match the retained inventory.

The historical cohort flow remains **211 → 188 → 158 → 153**. Three original
import-selection reasons remain unknown. Four historical air exclusions remain
reconstructed, including three approximate slice-order replays. Current
source alignment does not certify historical input masks.

The preprocessing source, AIM parser and default configuration still match
the retained reconstruction's hashes. Any future addition requires a separate
cohort version, with the internal development and prior-exposure role retained.
No source annotation, threshold, crop or frozen membership was changed.

Verification confirmed the private JSON/CSV inventory's count and identity
agreement, its output hash, the aggregate status counts, absence of molecular
outcome fields, no overlap with current corrected-input membership, no automatic
rebuild claim and unchanged source availability. Case identities, source UIDs
and detailed source locations remain in restricted local evidence and are
omitted here.

The metadata run completed in **6.73 seconds**, using **25.6 MB peak RSS** and
reading **1.38 MB** across 26 small metadata/source files. It made no public
network requests and read no CT headers, CT pixels, processed arrays or model
outputs. Directory metadata does not establish DICOM identity or image quality;
saved catalogue records do not establish current remote availability.

This is a completed metadata inventory with outstanding source requirements.
It establishes no accepted training smoke, scientific fit or downstream
clinical result.
