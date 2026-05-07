#!/usr/bin/env bash
set -euo pipefail

lambdas=(0.002 0.0002 0.00002)

for lambda in "${lambdas[@]}"; do
  name=${lambda//./p}
  echo "=== RATE_LAMBDA=$lambda ==="
  env \
    RUN_ID="lambda_${name}" \
    ITERATIONS=12 \
    WARMUP_STEPS=1 \
    COMPILE_STRUCTURED_MLP=1 \
    ENABLE_COMPILE=0 \
    VAL_TOKEN_LIMIT=65536 \
    VAL_BATCH_SIZE=65536 \
    TRAIN_BATCH_TOKENS=4096 \
    TRAIN_SEQ_LEN=256 \
    RATE_LAMBDA="$lambda" \
    SCALE_LAMBDA=0.0002 \
    python3 train_gpt.py
done
