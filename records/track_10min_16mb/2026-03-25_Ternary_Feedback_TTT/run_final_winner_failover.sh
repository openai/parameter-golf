#!/usr/bin/env bash
# =============================================================================
# FINAL WINNER FAILOVER: 16K Batch / Expanded GPU Candidates
# To recover from the A40 SSH timeout issue.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FAILOVER_LOG="${SCRIPT_DIR}/failover_winner_$(date +%Y%m%d_%H%M%S).log"

export RUNPOD_API_KEY="rpa_IXDRQPZKIX32BK35KRMWWJSV0I41ZA80L504CTIMjealdx"

# ── Architecture: SKC + MoE (Proven 1.68 BPB Winner) ──
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

# ── Extra Stable Flags ──
export LOCAL_SGD_SYNC_EVERY=1
export FEEDBACK_ENABLED=0
export LAWA_ENABLED=1
export SWA_ENABLED=1
export BIGRAM_HASH_ENABLED=1

# ── Hardware Failover strategy ──
# Broadening the search to include A100-SXM, L40S, or 6000 Ada
# to bypass the problematic A40 node in the previous region.
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=44
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_STEPS=3
export EXPORT_ONLY=0

# Orchestrator parameters
# We limit data shards to 6 for faster sync if needed, though high-speed pods handle 12 fine.
export DATA_SHARDS=12

printf "[%s] Launching FAILOVER WINNER REPRO (Expanded GPU pool)...\n" "$(date +%H:%M:%S)" | tee -a "$FAILOVER_LOG"
bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee -a "$FAILOVER_LOG"

# Results
printf "\n============================================\n"
printf "FAILOVER RUN COMPLETE\n"
grep "final_ternary_roundtrip" "$FAILOVER_LOG" || echo "Check log for results."
printf "============================================\n"
