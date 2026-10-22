# Native corrected-CPU readout entry

`execution_support/n08/readout_standalone_v1.py` is a versioned fit-consumption and readout-entry component. **It cannot produce a final or spatial scientific readout.** The native map producer/consumer interface is not implemented. This component does not complete M06 or M07 and makes no canonical-adoption claim.

## Current entry points

The command line performs a capability preflight and exits 2 with `native_map_integration_unavailable`. It writes a new `N08_NATIVE_READOUT_REFUSAL_best.json` (or `_last.json`) to the explicitly supplied output directory. It refuses to replace an existing file.

```sh
python execution_support/n08/readout_standalone_v1.py \
  --root /absolute/project/root \
  --output-root /absolute/native/execution/output \
  --out-dir /absolute/new/refusal/directory
```

This preflight does not call fit verification, open model/data/OOF/map inputs, or execute statistics. Its receipt reports `fit_verification_performed: false`, `native_full_fits_verified: 0`, and `scientific_readout_count: 0`. Those zeros describe this invocation; they do not establish whether valid fits exist elsewhere. `--maps-dir` cannot enable an unsupported endpoint, and `--which best|last` does not bypass the refusal. The retained module imports NumPy/SciPy; use an environment providing those dependencies.

For an explicitly authorized prerequisite check, the separate Python API `verify_native_fits(root, output_root)` discovers the exact 50 results in the pinned M1 matrix and invokes the reviewed native verifier with `mode="full"`. Full verification reads scientific artifact bytes through the unchanged verification helpers and requires the bound interpreter/runtime. It can be expensive. It is not run by the currently unavailable CLI endpoint.

The API refuses a missing matrix unit, an unexpected result outside the matrix and designated smoke, structural-only verification, a partial full-verification report, duplicate evidence, conflicting execution-set identities, changed metadata, inconsistent frozen rows, or inconsistent fit/initialization identities. The full verifier checks genuine E05 and smoke dependencies, runtime/source/data bindings and fresh technical acceptance. There is no saved-report acceptance path.

A successful prerequisite return contains the fresh verifier report, matrix rows, per-fit checkpoint and initialization identities, `native_bundle_sha256`, the common `execution_set_sha256`, and `native_input_bundle_sha256`. These names describe native evidence. They do not stand in for historical controller releases, witnesses, campaign fingerprints, or legacy accepted-receipt hashes. The return explicitly has `readout_complete: false` and `canonical_adoption: false`.

## Preserved methods and remaining interface

The retained source is `n08_replication_readout.py`, SHA256:

```text
b5e1ef4ef64f415cda4895477377ce7e1614e41a9526f5c2f588d6e289830e70
```

The versioned copy preserves `DECLARED`, its statistical constants, statistical functions, OOF loading, patient/fold/label alignment, design checks, condition-V checks and map-consistency functions without changing their source bodies. The native entry does not yet execute the OOF/map/statistical path. Legacy fit-receipt loading, controller verification, run-receipt writing and the old main orchestration are removed from this branch. Native uniqueness checks use explicit native identities.

The pinned native verifier is `standalone_receipts.py`, SHA256:

```text
013e165ee000bd7876f89c25eabc6911c98359fed148d12fb729034144319b42
```

The next interface must bind native fit bundles and current checkpoints to map outputs while preserving controls, data identity, patient/fold identity, and map definitions. Existing legacy map receipts expect a legacy fit-receipt digest. This component does not synthesize that digest or consume those receipts as native authority. There is no predictive-only alternative endpoint or implicit bypass.

## Focused checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests -p test_readout_standalone_v1.py -v
```

Tests use tiny synthetic metadata. They call the actual pinned verifier's batch function with explicit substitutions for its structural and scientific work, and stub NumPy/SciPy imports so no statistics run. They cover full-mode forwarding, zero-credit/partial refusals, identity mutations, uniqueness and initialization checks, and unsupported-map preflight. Fixture success is not scientific acceptance, real-data validation, or evidence that any actual fit/readout completed. The native verifier's separate full suite is not repeated here.
