#!/usr/bin/env bash
set -euo pipefail

# Smoke test: 1 GPU, 2 minutes, reduced settings
# Verifies the training + GPTQ + eval pipeline works end-to-end

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"

export NUM_LAYERS=11
export MAX_WALLCLOCK_SECONDS=120
export WARMDOWN_ITERS=800
export WARMUP_STEPS=5
export TRAIN_BATCH_TOKENS=262144
export TRAIN_SEQ_LEN=1024
export EVAL_SEQ_LEN=1024
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=50
export ITERATIONS=20000
export EVAL_STRIDE=64
export SEED=1337
export RUN_ID="smoke_1gpu"
export TARGET_MB="15.9"

echo "=== SMOKE TEST: 11L FullGPTQ + XSA + BigramHash (1 GPU, ~2min) ==="
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "DATA_PATH=$DATA_PATH"

cd "$SCRIPT_DIR"
python train_gpt.py
