# NSCLC-Radiomics metadata reconciliation

The local import log and imported-directory inventory contain 421 unique records. There are 420 processed rows and 420 referenced NPZ filenames, with none missing or extra. The one imported record without a processed row is exactly the one import without SEG. The advertised 422 remains unbound to a complete local source roster; no missing person was inferred.

These are metadata and filename checks. Image, mask and checkpoint payloads were not read. All processed rows declare masks and empty molecular-label dictionaries; this does not verify mask content or establish EGFR outcomes. Historical pretraining exposure remains recorded.

Full D08 is incomplete: source/geometry/tumor linkage, payload acceptance, executed partition and checkpoint lineage, and overlap evidence remain unresolved. The exact default checkpoint/log paths checked were absent; other historical checkpoint locations were not searched. Current code hashes do not retroactively bind old arrays.

Execution exited 0 in 4.735 seconds with 23.07 MB peak RSS. No new patients, derivatives, downloads or scientific fits were produced. Exact private membership remains outside GitHub. See the [safe aggregate](PUBLIC_AGGREGATE.json).
