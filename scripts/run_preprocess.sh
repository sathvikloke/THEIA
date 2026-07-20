#!/usr/bin/env bash
set -euo pipefail
export THEIA_ROOT="${THEIA_ROOT:-$(pwd)}"
python -m theia.data.download --raw_dir data/raw/nsclc_radiogenomics
python -m theia.data.preprocess
