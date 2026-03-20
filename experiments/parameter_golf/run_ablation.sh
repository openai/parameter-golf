#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${NPROC:=8}"
: "${SEED:=1337}"
: "${RUN_ID_PREFIX:=sota_push}"

COMMON_ENV=(
  DATA_PATH="$DATA_PATH"
  TOKENIZER_PATH="$TOKENIZER_PATH"
  VOCAB_SIZE=1024
  NUM_LAYERS=10
  MODEL_DIM=512
  NUM_HEADS=8
  NUM_KV_HEADS=4
  TRAIN_BATCH_TOKENS=524288
  TRAIN_SEQ_LEN=1024
  MAX_WALLCLOCK_SECONDS=600
  MATRIX_LR=0.04
  MUON_WEIGHT_DECAY=0.02
  TIED_EMBED_LR=0.10
  WARMDOWN_ITERS=2500
  INT8_ALWAYS_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight
  SEED="$SEED"
)

run_case () {
  local case_name="$1"
  shift
  echo "=== Running case: ${case_name} ==="
  env RUN_ID="${RUN_ID_PREFIX}_${case_name}_s${SEED}" "${COMMON_ENV[@]}" "$@" \
    torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py
}

# 1) Baseline final eval (non-overlap)
run_case standard FINAL_EVAL_MODE=standard

# 2) Sliding-window final eval
run_case sliding FINAL_EVAL_MODE=sliding EVAL_SEQ_LEN=1024 EVAL_STRIDE=64 EVAL_BATCH_SEQS=256

# 3) LoRA TTT final eval (default chunk=256)
run_case ttt_256 FINAL_EVAL_MODE=ttt TTT_CHUNK_SIZE=256 TTT_EVAL_SEQ_LEN=1024 TTT_LORA_RANK=8 TTT_LORA_LR=0.01 TTT_BATCH_SIZE=64

# 4) LoRA TTT finer chunks (more adaptation, higher eval cost)
run_case ttt_128 FINAL_EVAL_MODE=ttt TTT_CHUNK_SIZE=128 TTT_EVAL_SEQ_LEN=1024 TTT_LORA_RANK=8 TTT_LORA_LR=0.01 TTT_BATCH_SIZE=64
run_case ttt_64 FINAL_EVAL_MODE=ttt TTT_CHUNK_SIZE=64 TTT_EVAL_SEQ_LEN=1024 TTT_LORA_RANK=8 TTT_LORA_LR=0.01 TTT_BATCH_SIZE=64
