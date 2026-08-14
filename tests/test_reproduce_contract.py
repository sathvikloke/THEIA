"""`scripts/reproduce.sh` is the manuscript's load-bearing claim.

The paper says every number regenerates from archived predictions by a single
script. That claim broke silently when fusion.json's schema changed from
{"results": ...} to {"aggregate": ..., "per_run": ...} during the provenance fix:
the artifact was correct, the manuscript was correct, and the script that is
supposed to prove it crashed with a KeyError.

Running the whole script here would be slow, so instead the embedded reader
blocks are extracted and executed against the real artifacts. That is the part
that goes stale -- the analysis modules have their own tests.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

SCRIPT = "scripts/reproduce.sh"

pytestmark = pytest.mark.skipif(not os.path.exists(SCRIPT),
                                reason="reproduce.sh not present")


def _blocks() -> list[tuple[int, str]]:
    """Every `python - <<'PY' ... PY` heredoc, with the line it starts on."""
    src = open(SCRIPT).read().splitlines()
    out, buf, start = [], None, 0
    for i, line in enumerate(src, 1):
        if re.match(r"\s*python\s+-\s+<<'PY'\s*$", line):
            buf, start = [], i
        elif buf is not None and line.strip() == "PY":
            out.append((start, "\n".join(buf)))
            buf = None
        elif buf is not None:
            buf.append(line)
    return out


def test_script_has_reader_blocks():
    assert _blocks(), "no embedded python blocks found; did the script change?"


@pytest.mark.parametrize("idx", range(len(_blocks())))
def test_each_reader_block_runs_against_the_real_artifacts(idx):
    line, code = _blocks()[idx]
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"reproduce.sh block at line {line} fails against the current "
        f"artifacts:\n{proc.stderr.strip()[-800:]}")


def test_fusion_block_reports_the_canonical_runs():
    """A reader that silently accepted a non-canonical artifact would hide the
    exact bug this file exists to catch."""
    code = [c for _, c in _blocks() if "fusion.json" in c]
    assert code, "no fusion reader in reproduce.sh"
    proc = subprocess.run([sys.executable, "-c", code[0]],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "NOT the canonical set" not in proc.stdout, (
        "fusion.json was not built from CANONICAL.json")
    assert "0.618" in proc.stdout, "fusion arm should equal the headline"
