#!/usr/bin/env bash
# Quick smoke test on 1 GPU (60 second wall clock, no GPTQ, just trains + validates)
# Run from repo root:   bash records/track_10min_16mb/2026-04-16_SP8192_CleanStack_SGD_TTT/run_smoke_1gpu.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # repo root

SCRIPT="records/track_10min_16mb/2026-04-16_SP8192_CleanStack_SGD_TTT/train_gpt.py"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_8192_bpe.model}"
export VOCAB_SIZE=8192
export MAX_WALLCLOCK_SECONDS=60
export TTT_ENABLED=0
export SEED=42
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=25
export TRAIN_LOG_EVERY=10
export GPTQ_CALIBRATION_BATCHES=4

echo "=== Smoke test (1 GPU, 60s) ==="
torchrun \
  --standalone \
  --nproc_per_node=1 \
  "$SCRIPT" \
  2>&1 | tee smoke_test.log
echo "=== Smoke test done ==="
