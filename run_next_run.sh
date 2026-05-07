#!/usr/bin/env bash
set -euo pipefail

# Next-run preset focused on pushing from ~1.3x toward ~1.20x.
# - Uses more train data than smoke (default 40 shards)
# - Disables wallclock cap so ITERATIONS can complete
# - Keeps a stable no-compile-friendly batch size by default
# - Enables Bigram + mixed quant + EMA + QAT (last 15%) + decoupled WD stack

TRAIN_SHARDS="${TRAIN_SHARDS:-40}"

echo "[1/2] Ensuring dataset/tokenizer are present (sp1024, train_shards=${TRAIN_SHARDS})..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"

echo "[2/2] Starting tuned training run..."
RUN_ID="${RUN_ID:-runpod_next_try}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
ITERATIONS="${ITERATIONS:-30000}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-2000}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-10240}" \
BIGRAM_DIM="${BIGRAM_DIM:-128}" \
MIXED_QUANT_ENABLED="${MIXED_QUANT_ENABLED:-1}" \
EMA_ENABLED="${EMA_ENABLED:-1}" \
EMA_DECAY="${EMA_DECAY:-0.999}" \
QAT_ENABLED="${QAT_ENABLED:-1}" \
QAT_FINAL_FRAC="${QAT_FINAL_FRAC:-0.15}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}" \
NUM_LAYERS="${NUM_LAYERS:-10}" \
MLP_MULT="${MLP_MULT:-3}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}" \
MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}" \
MATRIX_LR="${MATRIX_LR:-0.03}" \
SCALAR_LR="${SCALAR_LR:-0.03}" \
TIED_EMBED_LR="${TIED_EMBED_LR:-0.04}" \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "Run complete. Check logs/\${RUN_ID}.txt for final_int8_zlib_roundtrip_exact."
