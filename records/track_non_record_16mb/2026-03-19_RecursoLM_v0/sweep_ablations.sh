#!/usr/bin/env bash
set -euo pipefail

# Phase-3 ablation sweep launcher.
# Edit arrays below for larger sweeps.
DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024}
TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}
RUN_DIR="records/track_non_record_16mb/2026-03-19_RecursoLM_v0"

MODEL_DIMS=(384 448 512)
REC_STEPS=(12 16 20)
MLP_DIMS=(1024 1280)

for d in "${MODEL_DIMS[@]}"; do
  for r in "${REC_STEPS[@]}"; do
    for f in "${MLP_DIMS[@]}"; do
      run_id="recurso_sweep_d${d}_r${r}_f${f}"
      echo "Starting $run_id"
      RUN_ID="$run_id" \
      DATA_PATH="$DATA_PATH" \
      TOKENIZER_PATH="$TOKENIZER_PATH" \
      VOCAB_SIZE=1024 \
      TRAIN_SEQ_LEN=512 \
      MODEL_DIM="$d" \
      NUM_LAYERS=2 \
      RECURRENCE_STEPS="$r" \
      NUM_HEADS=4 \
      NUM_KV_HEADS=1 \
      MLP_DIM="$f" \
      ITERATIONS=${ITERATIONS:-2000} \
      VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-500} \
      TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288} \
      MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600} \
      torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-8} "$RUN_DIR/train_gpt.py"
    done
  done
done
