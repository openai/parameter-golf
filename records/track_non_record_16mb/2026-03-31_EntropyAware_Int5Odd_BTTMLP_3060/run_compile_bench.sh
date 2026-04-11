#!/usr/bin/env bash
set -euo pipefail

for compile_flag in 0 1; do
  if [[ "$compile_flag" == "1" ]]; then
    name="compile_on_bench"
  else
    name="compile_off_bench"
  fi
  echo "=== $name ==="
  env \
    RUN_ID="$name" \
    ITERATIONS=8 \
    WARMUP_STEPS=1 \
    VAL_LOSS_EVERY=0 \
    COMPILE_STRUCTURED_MLP="$compile_flag" \
    ENABLE_COMPILE=0 \
    VAL_TOKEN_LIMIT=65536 \
    VAL_BATCH_SIZE=65536 \
    TRAIN_BATCH_TOKENS=4096 \
    TRAIN_SEQ_LEN=256 \
    RATE_LAMBDA=0.00002 \
    SCALE_LAMBDA=0.0002 \
    python3 train_gpt.py
done
