#!/usr/bin/env bash
# How often does externally-aligned attention actually emerge?
#
# THE QUESTION. The paper's primary endpoint holds across three training runs,
# but one of the three largely fails to align, and once the training run is
# treated as a random effect the margin over a centre prior is no longer
# distinguishable from zero. Three seeds cannot characterise that. This trains 20
# independent runs from the locked configuration and evaluates each on the same
# 420 external patients, turning "one run failed" into a distribution.
#
# NOT A FIX FOR H1. The external cohort has already been inspected, so nothing
# run here is blind or confirmatory. This is a labelled post-hoc characterisation
# of training variance, and it is reported as one.
#
# RUN IDS are `stab-s<seed>`, deliberately NOT `ms-s<seed>`: the canonical set is
# base-s1337-v2 / ms-s7 / ms-s42, and a glob over results/ms-s*.json is exactly
# how a superseded run got into three documents once already. Nothing here is
# swept into the headline.
#
# DISK. A full checkpoint is 1.8 GB and a run is 8.9 GB; 20 runs is 358 GB
# against 47 GB free. Each run is therefore evaluated and then slimmed to the
# vision+grounding weights the external endpoint actually loads, which keeps the
# peak near 15 GB and leaves every run re-evaluable without retraining.
#
# SEQUENTIAL on purpose: two concurrent runs pushed this machine to 15.6 GB of
# swap and slowed a diagnostic from 1.7 s/it to 90 s/it.
#
# RESUMABLE: a seed whose results/stab-s<seed>.json exists is skipped, so the
# script can be killed and restarted.
#
# Usage:  bash scripts/stability.sh [seed ...]     (default: 101..120)
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=($(seq 101 120))
fi

EXT_ROWS="data/processed_pretrain/rows.jsonl"
mkdir -p results/stability

echo "[stab] $(date '+%F %T') starting ${#SEEDS[@]} seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
    run_id="stab-s${seed}"
    ext_out="results/stability/${run_id}_external.json"

    if [ -f "$ext_out" ]; then
        echo "[stab] ${run_id} already evaluated, skipping"
        continue
    fi

    free_gb=$(df -g . | awk 'NR==2{print $4}')
    if [ "${free_gb:-0}" -lt 12 ]; then
        echo "[stab] ABORT: only ${free_gb} GB free, need >=12 for one run"
        exit 1
    fi

    if [ ! -f "results/${run_id}.json" ]; then
        echo "[stab] === $(date '+%T') training seed ${seed} -> ${run_id} ==="
        python -m theia.engine.train \
            --run_id "${run_id}" \
            --set "seed=${seed}" \
            --set train.loss_weights.gen=0.0 || {
                echo "[stab] training failed for ${run_id}; continuing"; continue; }
    fi

    echo "[stab] === $(date '+%T') external evaluation ${run_id} ==="
    python -m theia.analysis.external_grounding \
        --runs "${run_id}" \
        --rows "${EXT_ROWS}" \
        --out "${ext_out}" || {
            echo "[stab] external eval failed for ${run_id}; continuing"; continue; }

    echo "[stab] === $(date '+%T') slimming ${run_id} ==="
    python -m theia.analysis.slim_checkpoint --run "${run_id}" || true

    # Slimming gives 9.6 GB -> 1.75 GB, so 20 retained runs is ~35 GB against 47
    # free -- too tight to finish. Metrics and the per-patient npz are archived
    # before this point and are never pruned; only the re-evaluable weights go,
    # oldest first, and only when disk actually demands it.
    while [ "$(df -g . | awk 'NR==2{print $4}')" -lt 15 ]; do
        victim=$(ls -dt checkpoints/stab-s* 2>/dev/null | tail -1)
        [ -z "$victim" ] && break
        [ "$victim" = "checkpoints/${run_id}" ] && break
        echo "[stab] disk low; pruning slim weights for $(basename "$victim")"
        rm -rf "$victim"
    done

    echo "[stab] ${run_id} done; $(df -h . | awk 'NR==2{print $4}') free"
done

echo "[stab] $(date '+%F %T') all seeds finished; summarising"
python -m theia.analysis.stability || true
