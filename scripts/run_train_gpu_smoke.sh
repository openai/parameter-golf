#!/usr/bin/env bash
# Single-GPU smoke run for integration/train_gpt_bese.py (requires CUDA + real data shards).
# Usage:
#   export DATA_PATH=/path/to/dataset_with_fineweb_train_*.bin
#   export TOKENIZER_PATH=/path/to/bese_bpe_250.json
#   export VOCAB_SIZE=288   # must match tokenizer JSON
#   ./scripts/run_train_gpu_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${DATA_PATH:?Set DATA_PATH to shard directory}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to bese_bpe *.json}"
: "${VOCAB_SIZE:?Set VOCAB_SIZE to match tokenizer}"

export ITERATIONS="${ITERATIONS:-5}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"

torchrun --standalone --nproc_per_node=1 integration/train_gpt_bese.py
