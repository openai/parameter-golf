#!/usr/bin/env bash
# =============================================================================
# 8K FLASH TRAINING (A100 Tier Upgrade)
# Target: 200ms Step / 2000+ Total Steps
# Hardware: Forcing A100 (80GB) for stability and speed.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLASH_LOG="${SCRIPT_DIR}/flash_8k_a100_$(date +%Y%m%d_%H%M%S).log"

export RUNPOD_API_KEY="rpa_IXDRQPZKIX32BK35KRMWWJSV0I41ZA80L504CTIMjealdx"

# ── Architecture ──
export NUM_LAYERS=8
export MODEL_DIM=320
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=4

# ── 8K Flash Training ──
# Small batch (8192) + High-end GPUs = Sub-200ms steps.
export TRAIN_BATCH_TOKENS=8192
export TRAIN_SEQ_LEN=2048

# ── Optimization ──
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.02
export SCALAR_LR=0.015
export WARMDOWN_FRACTION=0.4

# ── Hardware Guarantee ──
# MIN_GPU_MEMORY_GB=80 blacklists all 48GB nodes (A40, L40, 6000 Ada).
# This forces the use of H100 or A100 SXM, ensuring 200ms latency.
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=80
export DATA_SHARDS=12

printf "[%s] Launching 8K FLASH (A100 Tier Guarantee)...\n" "$(date +%H:%M:%S)" | tee -a "$FLASH_LOG"
bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee -a "$FLASH_LOG"

# Results
printf "\n============================================\n"
printf "8K FLASH (A100) COMPLETE\n"
grep "final_ternary_roundtrip" "$FLASH_LOG" || echo "Check log for results."
printf "============================================\n"
