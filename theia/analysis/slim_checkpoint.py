"""Strip a checkpoint to the weights the external endpoint actually reads.

A full checkpoint here is 1.8 GB, most of it optimizer state and the LM. The
external attention evaluation loads only `vision.*` and `grounding.*` -- see
`theia.analysis.external_grounding.main`, which filters to exactly those prefixes
-- so everything else is dead weight once a run has been evaluated.

This matters because the stability experiment trains 20 runs at ~8.9 GB each,
which is 358 GB against 47 GB of disk. Slimming after evaluation keeps the peak
around 15 GB and leaves every run re-evaluable without retraining.

The original is deleted only after the slim file is written AND verified to load,
so an interrupted slim leaves the original intact.

Run: python -m theia.analysis.slim_checkpoint --run stab-s101
"""
from __future__ import annotations

import argparse
import glob
import os

import torch

KEEP_PREFIXES = ("vision.", "grounding.")


def slim_one(path: str, delete_original: bool = True) -> tuple[int, int]:
    """Rewrite `path` keeping only the evaluable weights. Returns (before, after)."""
    before = os.path.getsize(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = blob.get("model") or {}
    kept = {k: v for k, v in model.items() if k.startswith(KEEP_PREFIXES)}
    if not kept:
        raise ValueError(f"{path}: no vision./grounding. tensors; refusing to slim")

    out = {"model": kept, "slimmed": True,
           "fold": blob.get("fold"), "cfg": blob.get("cfg"),
           "epoch": blob.get("epoch"), "metrics": blob.get("metrics")}
    tmp = path + ".slim.tmp"
    torch.save(out, tmp)

    # Verify before destroying anything.
    check = torch.load(tmp, map_location="cpu", weights_only=False)
    assert set(check["model"]) == set(kept), "slim file did not round-trip"

    if delete_original:
        os.replace(tmp, path)
        after = os.path.getsize(path)
    else:
        after = os.path.getsize(tmp)
    return before, after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id under checkpoints/")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    paths = sorted(glob.glob(f"checkpoints/{a.run}/fold*/best.pt"))
    if not paths:
        raise SystemExit(f"[slim] no checkpoints for {a.run}")

    tot_b = tot_a = 0
    for p in paths:
        b, aft = slim_one(p, delete_original=not a.dry_run)
        tot_b += b
        tot_a += aft
        print(f"[slim] {p}: {b/1e9:.2f} GB -> {aft/1e9:.2f} GB")
    print(f"[slim] {a.run}: {tot_b/1e9:.2f} GB -> {tot_a/1e9:.2f} GB "
          f"(freed {(tot_b - tot_a)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
