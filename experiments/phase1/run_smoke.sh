#!/bin/bash
# Phase 1 smoke test: quick validation that all patches work end-to-end.
# ~2-3 minutes on 4xL40S or 4xH100.
set -euo pipefail
cd "$(dirname "$0")/../.."

# --- Model config ---
export NUM_LAYERS=${NUM_LAYERS:-10}
export MLP_MULT=${MLP_MULT:-3}
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.99}

# --- Phase 1 features ---
export QUANT_BITS=${QUANT_BITS:-6}
export COMPRESS_METHOD=${COMPRESS_METHOD:-zstd}
export SLIDING_STRIDE=${SLIDING_STRIDE:-64}
export MUON_WD=${MUON_WD:-0.04}

# --- Smoke test limits ---
export ITERATIONS=${ITERATIONS:-60}
export WARMUP_STEPS=${WARMUP_STEPS:-10}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-120}
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-65536}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-65536}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-60}
export TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-10}
export RUN_ID=${RUN_ID:-phase1_smoke}

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}

echo "=== Phase 1 Smoke Test ==="
echo "GPUs: ${NGPU}"
echo "Model: ${NUM_LAYERS}L, ${MLP_MULT}x MLP, dim=512"
echo "Quant: int${QUANT_BITS} + ${COMPRESS_METHOD}"
echo "Eval: sliding window stride=${SLIDING_STRIDE}"
echo "Muon WD: ${MUON_WD}"
echo ""

torchrun --standalone --nproc_per_node="${NGPU}" experiments/phase1/train_phase1.py
