# Corrected CPU source support

These are exact exports of the actual research checkout's technical acceptance helpers and fixed 50-row M1 matrix. The source mapping and SHA-256 values are in [SOURCE_EXPORT_MANIFEST.json](../../docs/execution_results_2026-10-20/SOURCE_EXPORT_MANIFEST.json). The current trainer and its static internal import closure are also exported into `theia/`.

The old campaign controller is not required by these helper modules. Current execution uses a separate, explicit session binding. These exports do not issue a historical release or certify an experiment.

The original research inputs remain local. [LOCAL_INPUTS.example.json](LOCAL_INPUTS.example.json) identifies the required source acceptance, content manifest, rows, fixed split membership, imaging arrays, and locally cached model. Patient membership and molecular labels are not part of this public export. Supply those genuine inputs before research execution.

The observed external runtime on October 20 was Python 3.13.9, Torch 2.10.0, NumPy 2.1.3, SimpleITK 2.5.5, pydicom 3.0.2, open_clip_torch 3.3.0 and transformers 5.2.0. The requirements files retain their original bytes; this observation is not a claim that an unrelated environment was recreated or that model weights were downloaded.
