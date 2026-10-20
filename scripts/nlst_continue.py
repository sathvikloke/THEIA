#!/usr/bin/env python3
"""Check sealed NLST source bindings; acquisition remains disabled until integration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .nlst_resume_core import Refusal, SEALED_LEGACY_SHA256, parse_utc, stable_sha256, utcnow
except ImportError:
    from nlst_resume_core import Refusal, SEALED_LEGACY_SHA256, parse_utc, stable_sha256, utcnow


SEALED_PLAN_SHA256 = "746c7f24c8fb5cb39ee8a5d29130a3847d181082386562fa53b0b5cffe36971b"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check-bindings",
        help="Read-only source/hash check; always refuses acquisition while adapter is unbound",
    )
    check.add_argument("--original-downloader", type=Path, required=True)
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--cutoff-utc", required=True)
    check.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    try:
        cutoff, deadline = parse_utc(args.cutoff_utc), parse_utc(args.deadline_utc)
        if not utcnow() < cutoff < deadline:
            raise Refusal("Require current UTC < cutoff UTC < deadline UTC")
        original_sha, _ = stable_sha256(args.original_downloader)
        if original_sha != SEALED_LEGACY_SHA256:
            raise Refusal("Sealed original downloader SHA-256 mismatch")
        plan_sha, _ = stable_sha256(args.plan)
        if plan_sha != SEALED_PLAN_SHA256:
            raise Refusal("Frozen plan SHA-256 mismatch")
        print(json.dumps({
            "status": "integration_unbound",
            "original_sha256": original_sha, "plan_sha256": plan_sha,
            "acquisition_enabled": False,
            "reason": (
                "Checksums match, but actual plan extraction, PatientID mapping, "
                "endpoint construction and original verifier comparison are not "
                "integrated. No network request or archive write was performed."
            ),
        }, sort_keys=True))
    except (OSError, ValueError, Refusal) as exc:
        print(json.dumps({
            "status": "refused", "acquisition_enabled": False,
            "reason": type(exc).__name__ + ":" + str(exc),
        }, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
