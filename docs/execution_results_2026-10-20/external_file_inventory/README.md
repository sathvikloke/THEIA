# External CT and molecular-file inventory reconciliation

On 2026-10-20, a metadata-only D10/D11 reconciliation confirmed that the existing TCGA-LUAD and CPTAC-LUAD availability inventories can be reused. The October 15/16 execution reports contain no newer external endpoint join or evaluation. No new download, CT processing, molecular decoding, clinical-label linkage, or prediction change was performed.

| Resource | Reconciled availability |
| --- | --- |
| TCGA-LUAD CT | 67 source subjects and matching local subject directories; 551 source CT series |
| Historical TCGA evidence | 52 prior model inputs, including 40 historical primary evaluated cases; 15 other subjects remain never-scored, without a fresh label-blind holdout claim |
| CPTAC-LUAD CT | 37 source subjects; 29 previously selected and acquired index series |
| Existing CPTAC pre-reader eligibility | 28 subjects under the unchanged prior selection |
| Public CPTAC WXS files | 36 files for 36 subjects, totaling 3,335,010 bytes |
| CT/WXS availability overlap | 28 of 29 selected CT subjects; all 28 subjects in the existing pre-reader eligibility set |
| Without open-WXS metadata | One CT subject, outside the existing 28-person eligibility set |
| Controlled targeted-sequencing file | One remains excluded |
| Missing permitted downloads | Zero in the retained source metadata snapshot |
| Newly evaluable patients | Zero |

Five imaging, selection, QC, and eligibility metadata hashes match the prior inventory. The exact public file/case mappings match the retained GDC acquisition receipts. All 36 molecular files remain present with their expected sizes. Their MD5 and SHA-256 verification is reused from **2026-10-14**; molecular payloads were not opened or rehashed on October 20. CT directories and retrieval receipts were checked without reading pixels. This was a reconciliation of saved metadata, not a new remote inventory refresh.

The final metadata process exited **0**, took **4.81 seconds**, and used **22,970,368 bytes** peak RSS. An initial utility parsing error was corrected to use the source selection CSV's yes/no encoding; the failed attempt was retained separately.

Availability does not establish EGFR status, clinical eligibility, or prospective blinding. Historical TCGA exposure remains historical. The existing CPTAC timing selection, localization requirements, and prediction/label sequence remain unchanged. Private case identifiers, file linkage records, and source data are retained outside the public repository.
