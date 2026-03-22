#!/usr/bin/env bash
# =============================================================================
# H100 SUBMISSION LAUNCH — OpenAI Parameter Golf
# 8x H100 SXM5, 10 minutes, 16MB artifact limit
# Target: 10L×960D SwiGLU-3× BigramHash SmearGate CGGR SWA QAT
# Est. params: ~96M, est. artifact: ~14.9MB (safely under 16MB)
# =============================================================================
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs artifacts

RUN_ID="${RUN_ID:-h100_submission_10x960_mlp3}"
LOG="$REPO/logs/${RUN_ID}.txt"

echo "=== H100 Submission Launch: 10L×960D SwiGLU-3× BigramHash SmearGate ==="
echo "Log: $LOG"

# TRAIN_BATCH_TOKENS=786432 divisible by 8 GPUs * 1 grad_accum * 2048 seq_len = 16384 ✓ (48 seqs/GPU/step)
# Competition infrastructure sets RANK, WORLD_SIZE, LOCAL_RANK via torchrun.

env \
  RUN_ID="$RUN_ID" \
  NUM_LAYERS=10 \
  MODEL_DIM=960 \
  ATTN_NHEADS=8 \
  ATTN_KV_HEADS=4 \
  ATTN_LINEAR_IMPL=casted \
  ATTN_FFN_EXPAND=3.0 \
  MLP_ACTIVATION=swiglu \
  BIGRAM_HASH_SIZE=10240 \
  BIGRAM_HASH_DIM=128 \
  SMEAR_GATE=1 \
  RESET_ON_BOS=1 \
  VOCAB_SIZE=1024 \
  TRAIN_SEQ_LEN=2048 \
  TRAIN_BATCH_TOKENS=786432 \
  ITERATIONS=20000 \
  WARMUP_STEPS=40 \
  WARMDOWN_ITERS=2000 \
  MAX_WALLCLOCK_SECONDS=590 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.04 \
  TIED_EMBED_LR=0.05 \
  EMBED_LR=0.6 \
  MUON_WD=0.04 \
  MUON_MOMENTUM=0.99 \
  MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=1500 \
  CGGR_RATIO=0.5 \
  CGGR_WARMUP=500 \
  SWA_START_FRACTION=0.7 \
  QAT_START_FRACTION=0.3 \
  EVAL_STRIDE=64 \
  COMPILE_MODEL=1 \
  VAL_LOSS_EVERY=500 \
  TRAIN_LOG_EVERY=100 \
  SEED=1337 \
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    -m train_gpt \
    2>&1 | tee "$LOG"
