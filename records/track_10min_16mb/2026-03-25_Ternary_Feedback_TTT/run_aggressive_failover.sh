#!/usr/bin/env bash
# =============================================================================
# AGGRESSIVE FAILOVER: NVIDIA A100 Tier / 16K Batch
# Blacklisting A40 nodes by forcing A100 memory requirements.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FAILOVER_LOG="${SCRIPT_DIR}/aggressive_failover_$(date +%Y%m%d_%H%M%S).log"

export RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: Please export RUNPOD_API_KEY before running.}"

# ── Architecture: SKC + MoE (Winning 1.68 BPB Config) ──
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

# ── Training Settings ──
export TRAIN_BATCH_TOKENS=16384
export TRAIN_SEQ_LEN=2048

# ── Optimization ──
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.02
export SCALAR_LR=0.015
export MUON_MOMENTUM_WARMUP_STEPS=0
export WARMDOWN_FRACTION=0.4

# ── Safe Flags ──
export LOCAL_SGD_SYNC_EVERY=1
export FEEDBACK_ENABLED=0
export LAWA_ENABLED=1
export SWA_ENABLED=1
export BIGRAM_HASH_ENABLED=1

# ── Hardware Aggressive Failover ──
# Setting MIN_GPU_MEMORY_GB=80 explicitly blacklists A40 (48GB) and L40S (48GB).
# This forces the selection of NVIDIA A100 or H100 nodes, which have 
# the highest provisioning success rates and fastest NVLink interconnects.
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=80
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_STEPS=3
export EXPORT_ONLY=0
export DATA_SHARDS=12

printf "[%s] Launching AGGRESSIVE FAILOVER (A100 Tier)...\n" "$(date +%H:%M:%S)" | tee -a "$FAILOVER_LOG"
bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee -a "$FAILOVER_LOG"

# Results
printf "\n============================================\n"
printf "AGGRESSIVE FAILOVER COMPLETE\n"
grep "final_ternary_roundtrip" "$FAILOVER_LOG" || echo "Check log for results."
printf "============================================\n"
