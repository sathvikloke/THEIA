#!/usr/bin/env bash
# Multi-seed headline estimate.
#
# One run's third decimal is not meaningful here. Measured across runs on
# essentially the same configuration, pooled EGFR AUC moved 0.621 / 0.656 / 0.660
# — a spread of ~0.04 from initialisation and MPS non-determinism alone. Quoting
# a single run implies a precision the estimator does not have.
#
# Runs are SEQUENTIAL on purpose. Two concurrent runs pushed this machine to
# 15.6 GB of swap and a diagnostic slowed from 1.7 s/it to 90 s/it at 8.5% CPU;
# the wall-clock saving from overlapping them is negative.
#
# loss_weights.gen=0 is the measured-stable configuration. With the generation
# term active, folds hit a marginal overflow in the MPS backward, skip every
# optimizer step, and report an untrained model. See docs/RESULTS.md.
#
# Usage:  bash scripts/multiseed.sh [seed ...]      (default: 1337 7 42)
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-1337 7 42}")
read -r -a SEEDS <<< "${SEEDS[@]}"

for seed in "${SEEDS[@]}"; do
    run_id="ms-s${seed}"
    if [ -f "results/${run_id}.json" ]; then
        echo "[multiseed] ${run_id} already archived, skipping"
        continue
    fi
    echo "[multiseed] === seed ${seed} -> ${run_id} ==="
    python -m theia.engine.train \
        --run_id "${run_id}" \
        --set "seed=${seed}" \
        --set train.loss_weights.gen=0.0
done

echo "[multiseed] aggregating"
python -m theia.analysis.aggregate --pattern 'results/ms-s*.json'
