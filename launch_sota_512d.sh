#!/usr/bin/env bash
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs

RUN_ID="sota_10x512_mlp3_bigram"
LOG="$REPO/logs/${RUN_ID}.txt"

echo "Launching SOTA-aligned: 10x512D SwiGLU-3x + BigramHash + SmearGate + Muon WD=0.04"
echo "Log: $LOG"

nohup env \
  PYTHONUNBUFFERED=1 \
  NUM_LAYERS=10 \
  MODEL_DIM=512 \
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
  TRAIN_SEQ_LEN=256 \
  TRAIN_BATCH_TOKENS=16384 \
  ITERATIONS=300 \
  WARMUP_STEPS=5 \
  WARMDOWN_ITERS=60 \
  MATRIX_LR=0.005 \
  SCALAR_LR=0.010 \
  TIED_EMBED_LR=0.015 \
  EMBED_LR=0.015 \
  MUON_WD=0.04 \
  MUON_MOMENTUM=0.99 \
  MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=150 \
  SEED=1337 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=50 \
  COMPILE_MODEL=1 \
  EVAL_STRIDE=64 \
  SWA_START_FRACTION=0.7 \
  QAT_START_FRACTION=0.3 \
  RUN_ID="$RUN_ID" \
  /usr/bin/python3 -u train_gpt.py > "$LOG" 2>&1 &

echo "PID: $!"
echo "Waiting for startup..."
sleep 20
tail -10 "$LOG"
