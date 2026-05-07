#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export ITERATIONS="${ITERATIONS:-600}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8192}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

run_one() {
  local run_id="$1"
  local num_layers="$2"
  local model_dim="$3"
  local num_heads="$4"
  local num_kv_heads="$5"
  shift
  shift
  shift
  shift
  shift
  echo "==== Running $run_id (L=$num_layers D=$model_dim H=$num_heads KV=$num_kv_heads) ===="
  RUN_ID="$run_id" \
  NUM_LAYERS="$num_layers" \
  MODEL_DIM="$model_dim" \
  NUM_HEADS="$num_heads" \
  NUM_KV_HEADS="$num_kv_heads" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py "$@"
}

# Control: baseline-ish
run_one "aria_sweep_9x512_kv4" 9 512 8 4

# Slightly deeper, narrower
run_one "aria_sweep_10x480_kv4" 10 480 8 4

# Slightly shallower, wider
run_one "aria_sweep_8x576_kv4" 8 576 8 4

echo "Sweep complete. Check logs/*.txt for final_int8_zlib_roundtrip_exact and size."
