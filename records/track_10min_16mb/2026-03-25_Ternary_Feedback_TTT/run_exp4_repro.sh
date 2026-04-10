#!/usr/bin/env bash
# =============================================================================
# REPRODUCE BEST RESULT: 320D / 8 Experts
# Reproduces the 1.6825 BPB win from the latest sweep.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO_LOG="${SCRIPT_DIR}/repro_exp4_$(date +%Y%m%d_%H%M%S).log"

export RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: Please export RUNPOD_API_KEY before running.}"

# Winning hyperparameters from Experiment 4
export NUM_LAYERS=8
export MODEL_DIM=320
export KOOPMAN_MIXER_RANK=4
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=4
export BIGRAM_HASH_BUCKETS=16384

# Run configuration
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=40   # A40/H100 class
export TRAIN_BATCH_TOKENS=65536
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_STEPS=3
export SLIDING_EVAL=0
export EXPORT_ONLY=0

# Hardware tweaks
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

printf "[%s] Launching REPRO Exp 4 (320D / 8 experts)...\n" "$(date +%H:%M:%S)" | tee -a "$REPRO_LOG"
bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee -a "$REPRO_LOG"

# Extract Result Summary
BEST_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$REPRO_LOG" | sort -n | head -1 || echo "N/A")
FINAL_BPB=$(grep 'final_eval_proxy' "$REPRO_LOG" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1 || echo "N/A")
SIZE_MB=$(grep 'artifact:' "$REPRO_LOG" | grep -oP 'artifact:\K[0-9.]+MB' | tail -1 || echo "N/A")

printf "\n============================================\n"
printf "REPRODUCTION COMPLETE\n"
printf "Best Proxy BPB:  %s\n" "${BEST_BPB}"
printf "Final Roundtrip: %s\n" "${FINAL_BPB}"
printf "Artifact Size:   %s\n" "${SIZE_MB}"
printf "============================================\n"
