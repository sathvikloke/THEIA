#!/usr/bin/env bash
set -euo pipefail
export THEIA_ROOT="${THEIA_ROOT:-$(pwd)}"
python -m theia.engine.train --config configs/default.yaml
