#!/usr/bin/env bash
set -euo pipefail
export THEIA_ROOT="${THEIA_ROOT:-$(pwd)}"
CKPT="${1:-checkpoints/fold0/best.pt}"
python -m theia.reader_study.build_cases --ckpt "$CKPT"
echo "cases built; launch with: python -m theia.reader_study.serve --reader R1"
