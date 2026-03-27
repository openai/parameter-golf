#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Competitive-style run: 8× GPU, 10-minute training wallclock, 16MB cap check.
#
# Based on public PR #374 (unnir) ideas: https://github.com/openai/parameter-golf/pull/374
#
# What PR #374 actually adds (you do NOT get all of this from stock train_gpt.py):
#   - Tight SWA: average only when LR scale < 0.2 (last ~600 steps), every 50 steps
#   - Architecture: Partial RoPE 16/64, LN scale 1/sqrt(layer), XSA on last 4 layers,
#     Shared Value Embedding (layers 9–10), SmearGate, FA3 on Hopper, etc.
#   - Late QAT: fake int6 when LR scale < 0.1 (not "last 15% of steps")
#   - Quant: int6 MLP+attn, int8 embed, zstd-22 artifact, GPTQ-lite clip search
#
# This script uses YOUR train_gpt.py: EMA + last-fraction QAT + mixed quant + zlib.
# Expect different val_bpb vs the PR; to reproduce SOTA, use the PR branch / records.
# -----------------------------------------------------------------------------

cd "$(dirname "$0")"

export RUN_ID="${RUN_ID:-comp_8xh100_10min}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export MAX_SUBMISSION_BYTES="${MAX_SUBMISSION_BYTES:-16000000}"
# Stop on time, not on step count
export ITERATIONS="${ITERATIONS:-999999}"

# PR #374-style global batch / seq (8-way data parallel)
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"

export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-3}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MODEL_DIM="${MODEL_DIM:-512}"

export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"

# PR #374 optimizer snapshot (AdamW/Muon WD both 0.04 in issue text)
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"

export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"

# BigramHash: PR uses 2048 buckets × 128 dim
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export MIXED_QUANT_ENABLED="${MIXED_QUANT_ENABLED:-1}"

export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_DECAY="${EMA_DECAY:-0.999}"
export QAT_ENABLED="${QAT_ENABLED:-1}"
export QAT_FINAL_FRAC="${QAT_FINAL_FRAC:-0.15}"

# Fewer val passes in a 10-minute race (raise if you want more checkpoints)
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-3000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
  echo "[1/2] Dataset (sp1024, train_shards=${TRAIN_SHARDS})..."
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
else
  echo "[1/2] SKIP_DATA_PREP=1 — using existing data under DATA_PATH"
fi

echo "[2/2] Training: 8 proc, MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} ..."
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "Done. Logs: logs/${RUN_ID}.txt — check final_int8_zlib_roundtrip_exact and Total submission size int8+zlib"
