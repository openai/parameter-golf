#!/bin/bash
# LOCAL TEST: Quick validation on 1xGPU (4080/H100)
# Runs ~60s to verify warmdown, SWA, QAT timing and config correctness.
# NOT for competition scores — just for debugging.

set -e
cd "$(dirname "$0")"

export TRAIN_SEQ_LEN=512 EVAL_SEQ_LEN=512 UNET_SKIPS=1
export TRAIN_BATCH_TOKENS=32768
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=500

# Match PR #414: EMA + Tight SWA, no QAT
export EMA_ENABLED=1
export SWA=1
export QAT=0

# No TTT
export TTT_ENABLED=0
export TTT_CAUSAL=0

# Short run for validation
export MAX_WALLCLOCK_SECONDS=60
export TIER2_MODE=0
export SEED=${1:-1337}

echo "=== LOCAL TEST (1xGPU, 60s) ==="
echo "SEED=$SEED EMA=1 SWA=1 QAT=0 warmdown=500"
echo "Checking: SWA activates, EMA applies, sliding window eval runs"
echo "================================"

python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
