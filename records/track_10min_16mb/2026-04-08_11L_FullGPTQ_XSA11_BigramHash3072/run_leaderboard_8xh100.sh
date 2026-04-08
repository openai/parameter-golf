#!/usr/bin/env bash
set -euo pipefail

# Leaderboard run: 8×H100 SXM, 10 minutes
# This is the full submission configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"

export NUM_LAYERS=11
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS=4000
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export VAL_LOSS_EVERY=4000
export TRAIN_LOG_EVERY=500
export ITERATIONS=20000
export EVAL_STRIDE=64
export SEED=${SEED:-1337}
export RUN_ID="leaderboard_${SEED}"
export TARGET_MB="15.9"

echo "=== LEADERBOARD RUN: 11L FullGPTQ + XSA + BigramHash (8x H100 SXM, 10min) ==="
echo "SEED=$SEED"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "DATA_PATH=$DATA_PATH"

cd "$SCRIPT_DIR"
torchrun --standalone --nproc_per_node=8 train_gpt.py
