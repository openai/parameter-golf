#!/bin/bash
# Run training + evaluation on 8xH100 (competition setting)
# Usage: SEED=1337 bash eval/eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-1337}"

echo "=== Parameter Golf: 8xH100 Training + Eval ==="
echo "Seed: $SEED"
echo "Config: 11L, 3xMLP, XSA4, PartialRoPE16, LNScale, EMA, GPTQ-lite, SWA, Late QAT"

torchrun --standalone --nproc_per_node=8 train.py \
  --run-id "submission_seed${SEED}" \
  2>&1 | tee "logs/submission_seed${SEED}.log"

echo "=== Done ==="
