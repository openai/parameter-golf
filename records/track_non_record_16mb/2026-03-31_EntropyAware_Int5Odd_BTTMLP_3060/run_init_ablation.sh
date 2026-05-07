#!/usr/bin/env bash
set -euo pipefail

for init in mup xavier; do
  echo "=== BTT_INIT=$init ==="
  env \
    RUN_ID="init_${init}_bench" \
    ITERATIONS=12 \
    WARMUP_STEPS=1 \
    VAL_LOSS_EVERY=0 \
    COMPILE_STRUCTURED_MLP=1 \
    ENABLE_COMPILE=0 \
    BTT_INIT="$init" \
    VAL_TOKEN_LIMIT=65536 \
    VAL_BATCH_SIZE=65536 \
    TRAIN_BATCH_TOKENS=4096 \
    TRAIN_SEQ_LEN=256 \
    RATE_LAMBDA=0.00002 \
    SCALE_LAMBDA=0.0002 \
    python3 train_gpt.py
done
