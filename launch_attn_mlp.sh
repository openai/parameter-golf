#!/usr/bin/env bash
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs

RUN_ID="attn_10x640_mlp2_bigram"
LOG="$REPO/logs/${RUN_ID}.txt"

echo "Launching Run B: 10x640 MLP expand=2 + bigram=10240"
echo "Log: $LOG"

tmux new-session -d -s run_b 2>/dev/null || true
tmux send-keys -t run_b "nohup env \
  PYTHONUNBUFFERED=1 \
  NUM_LAYERS=10 \
  MODEL_DIM=640 \
  ATTN_NHEADS=8 \
  ATTN_KV_HEADS=4 \
  ATTN_LINEAR_IMPL=casted \
  ATTN_FFN_EXPAND=2.0 \
  BIGRAM_HASH_SIZE=10240 \
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
  SEED=1337 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=50 \
  COMPILE_MODEL=1 \
  SWA_START_FRACTION=0.7 \
  QAT_START_FRACTION=0.1 \
  RUN_ID=\"$RUN_ID\" \
  /usr/bin/python3 -u train_gpt.py > \"$LOG\" 2>&1" Enter

echo "Waiting for startup..."
sleep 20
tail -5 "$LOG"
