#!/usr/bin/env bash
# =============================================================================
# PRODUCTION STABLE RUN: Pure DDP / 320D / 8 Experts
# Stabilized version without experimental Sync/Feedback logic.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROD_LOG="${SCRIPT_DIR}/production_stable_$(date +%Y%m%d_%H%M%S).log"

export RUNPOD_API_KEY="rpa_IXDRQPZKIX32BK35KRMWWJSV0I41ZA80L504CTIMjealdx"

# Winning Architecture
export NUM_LAYERS=8
export MODEL_DIM=320
export KOOPMAN_MIXER_RANK=4
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=4
export BIGRAM_HASH_BUCKETS=16384

# Stable DDP Flags
export LOCAL_SGD_SYNC_EVERY=8
export LOCAL_SGD_WARMUP_STEPS=30
export FEEDBACK_ENABLED=0
export HESSIAN_TERNARY_GPTQ=0

# Hardware & Scaling
# Unsetting TRAIN_BATCH_TOKENS to allow autotuning for max throughput (Target: ~800ms/step)
unset TRAIN_BATCH_TOKENS
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=40
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_STEPS=3
export EXPORT_ONLY=0

printf "[%s] Launching PRODUCTION STABLE (Pure DDP)...\n" "$(date +%H:%M:%S)" | tee -a "$PROD_LOG"
bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee -a "$PROD_LOG"

# Extract Result Summary
BEST_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$PROD_LOG" | sort -n | head -1 || echo "N/A")
FINAL_BPB=$(grep 'final_ternary_roundtrip' "$PROD_LOG" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1 || echo "N/A")
SIZE_MB=$(grep 'artifact:' "$PROD_LOG" | grep -oP 'artifact:\K[0-9.]+MB' | tail -1 || echo "N/A")

printf "\n============================================\n"
printf "PRODUCTION RUN COMPLETE\n"
printf "Best Proxy BPB:  %s\n" "${BEST_BPB}"
printf "Final Roundtrip: %s\n" "${FINAL_BPB}"
printf "Artifact Size:   %s\n" "${SIZE_MB}"
printf "============================================\n"
