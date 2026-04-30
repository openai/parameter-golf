#!/usr/bin/env bash
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs

RUN_ID="swa_qat_14x640_cont"
LOG="$REPO/logs/${RUN_ID}.txt"

echo "Launching SWA+QAT continuation from play_cont512_lr010 (2.0731 bpb)"
echo "Log: $LOG"

nohup env \
  PYTHONUNBUFFERED=1 \
  ARCH=attention \
  ATTN_EVERY=1 \
  NUM_LAYERS=14 \
  MODEL_DIM=640 \
  ATTN_NHEADS=8 \
  ATTN_KV_HEADS=4 \
  ATTN_LINEAR_IMPL=casted \
  SSM_LINEAR_IMPL=casted \
  BITLINEAR_EVAL_MODE=float \
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
  VAL_LOSS_EVERY=150 \
  TRAIN_LOG_EVERY=50 \
  COMPILE_MODEL=1 \
  INIT_CKPT=artifacts/play_cont512_lr010_120051.final_model.pt \
  SWA_START_FRACTION=0.7 \
  QAT_START_FRACTION=0.1 \
  RUN_ID="$RUN_ID" \
  /usr/bin/python3 -u train_gpt.py > "$LOG" 2>&1 &

echo "PID: $!"
echo "Waiting for startup..."
sleep 15
tail -5 "$LOG"
