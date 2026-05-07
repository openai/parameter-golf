#!/usr/bin/env bash
set -euo pipefail

configs=(
  "R128_L12 BTT_RANK=128 NUM_LAYERS=12"
  "R256_L16 BTT_RANK=256 NUM_LAYERS=16"
  "R512_L20 BTT_RANK=512 NUM_LAYERS=20"
  "R1024_L20 BTT_RANK=1024 NUM_LAYERS=20"
  "R1024_L24 BTT_RANK=1024 NUM_LAYERS=24"
)

for cfg in "${configs[@]}"; do
  set -- $cfg
  name=$1
  shift
  echo "=== $name ==="
  env \
    PYTHONUNBUFFERED=1 \
    RUN_ID="scout_${name}" \
    ITERATIONS=0 \
    COMPILE_STRUCTURED_MLP=0 \
    ENABLE_COMPILE=0 \
    VAL_TOKEN_LIMIT=2048 \
    VAL_BATCH_SIZE=2048 \
    TRAIN_BATCH_TOKENS=4096 \
    TRAIN_SEQ_LEN=256 \
    RATE_LAMBDA=0.0002 \
    SCALE_LAMBDA=0.0002 \
    "$@" \
    python3 train_gpt.py
done
