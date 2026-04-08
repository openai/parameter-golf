#!/bin/bash
# Smoke test on 1 GPU – quick sanity check (~2 min)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="smoke_1gpu_$(date +%s)"
export SEED=1337
export ITERATIONS=500
export WARMDOWN_ITERS=100
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=100
export TRAIN_LOG_EVERY=50
export MAX_WALLCLOCK_SECONDS=120
export GRAD_ACCUM_STEPS=8
export SLIDING_EVAL=1
export EVAL_STRIDE=256
export COMPRESS_METHOD=lzma

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"

echo "=== Smoke Test 1xGPU ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
python "$SCRIPT_DIR/train_gpt.py"
echo "=== Done ==="
