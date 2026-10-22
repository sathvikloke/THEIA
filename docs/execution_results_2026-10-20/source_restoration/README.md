# One existing CT reference restored

On 2026-10-20, one previously catalogued CT series referenced by an existing development-case annotation was acquired from the official TCIA NSCLC Radiogenomics collection under CC BY 3.0. The imported source available earlier was a different series. The original import and the frozen 158-person M1 dataset were not changed.

The single archive request returned **88,502,169 bytes** containing **274 CT objects**, totaling **144,484,576 uncompressed DICOM bytes**. All 274 objects matched the official MD5 manifest. Header checks confirmed the exact patient, study, series, complete SOP inventory, and referenced annotation SOP. Exact source hashes and terminal receipts are retained privately.

The acquisition and verification exited **0** in **39.90 seconds**, with **74,317,824 bytes peak RSS**. MD5 and header checks streamed inside the ZIP: no CT objects were extracted and no pixels were decoded. An earlier wrapper preflight refused an inconsistent per-job cutoff before any archive request; its corrected session retained the original earlier job deadline. The successful run made one archive request with no retry.

This closes one source-availability gap in the existing 19-case development inventory. That prior assessment is preserved, with the restoration recorded in a separate private addendum. No image array, clinical label, new eligibility decision, new evaluable patient, model fit, or M1 membership change is credited by this restoration. Further technical processing must use the existing localization and pixel checks; source availability alone does not satisfy them.

Official source and license: [TCIA NSCLC Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/). Private identifiers, annotation coordinates, linkage records, and source images remain outside the public repository.
