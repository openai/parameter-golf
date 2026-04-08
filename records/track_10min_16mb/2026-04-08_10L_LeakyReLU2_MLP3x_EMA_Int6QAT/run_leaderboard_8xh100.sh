#!/bin/bash
# Full leaderboard run on 8xH100 SXM – 10 minute budget
set -euo pipefail

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="leaderboard_8xh100_$(date +%s)"
export SEED="${SEED:-1337}"
export ITERATIONS=20000
export WARMDOWN_ITERS=3500
export WARMUP_STEPS=20
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=200
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=524288
export VAL_BATCH_SIZE=524288
export SLIDING_EVAL=1
export EVAL_STRIDE=256
export COMPRESS_METHOD=lzma

NUM_GPUS=$(nvidia-smi -L | wc -l)
export EMA_DECAY=0.997
export EMA_START_STEP=100
export QAT_START_FRAC=0.80
export QAT_BITS=6

# Optimizer
export MUON_WD=0.04
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export TIED_EMBED_LR=0.05

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
echo "=== Leaderboard Run: ${NUM_GPUS}xGPU ==="
torchrun --standalone --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/train_gpt.py"
echo "=== Done ==="
