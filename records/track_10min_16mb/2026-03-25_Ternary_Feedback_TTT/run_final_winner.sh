#!/usr/bin/env bash
# =============================================================================
# FINAL WINNER REPRO: 16K Batch + Pure DDP (or LocalSGD if needed)
# Matches the sweep winner (8L/320D/8experts -> BPB 1.6825)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FINAL_LOG="${SCRIPT_DIR}/final_winner_$(date +%Y%m%d_%H%M%S).log"

export RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: Please export RUNPOD_API_KEY before running.}"

# ── Architecture: SKC + MoE — sweep winner ──
export NUM_LAYERS=8
export MODEL_DIM=320
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export KOOPMAN_MIXER_RANK=4
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=4
export BIGRAM_HASH_BUCKETS=16384

# ── Training settings ──
# KEY: 16K batch maximizes gradient updates in 599s.
export TRAIN_BATCH_TOKENS=16384
export TRAIN_SEQ_LEN=2048

# ── Optimization ──
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.02
export SCALAR_LR=0.015
export MUON_MOMENTUM_WARMUP_STEPS=0
export WARMDOWN_FRACTION=0.4

# ── Extra Stable Flags ──
export LOCAL_SGD_SYNC_EVERY=8   # LocalSGD: skip allreduce, average weights every 8 steps
export LOCAL_SGD_WARMUP_STEPS=30  # Standard DDP for first 30 steps, then LocalSGD
export FEEDBACK_ENABLED=0
export LAWA_ENABLED=1
export SWA_ENABLED=1
export BIGRAM_HASH_ENABLED=1

# ── Export & Hardware ──
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=40
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_STEPS=3
export EXPORT_ONLY=0

printf "[%s] Launching FINAL WINNER REPRO (16K Batch)...\n" "$(date +%H:%M:%S)" | tee -a "$FINAL_LOG"
bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee -a "$FINAL_LOG"

# Results
printf "\n============================================\n"
printf "RETRY COMPLETE\n"
grep "final_ternary_roundtrip" "$FINAL_LOG" || echo "Check log for results."
printf "============================================\n"
