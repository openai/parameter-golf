#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-random_up_12l_5layers_rank16}"

case "$CONFIG" in
  baseline_12l)
    RUN_ID="baseline_12l" \
    NUM_LAYERS=12 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2 \
    TRAIN_SEQ_LEN=1024 \
    TRAIN_BATCH_TOKENS=524288 \
    MATRIX_LR=0.02 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    RANDOM_MLP_UP_LAYERS="" \
    RANDOM_MLP_UP_RANK=16 \
    RANDOM_MLP_UP_GAIN=1 \
    RANDOM_MLP_UP_BASE_SEED=20260403 \
    RANDOM_MLP_UP_INIT=qr \
    FINAL_SLIDING_EVAL=1 \
    EVAL_STRIDE=64 \
    EVAL_SEQ_LEN=1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    ;;
  random_up_12l_5layers_rank16)
    RUN_ID="random_up_12l_5layers_rank16" \
    NUM_LAYERS=12 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2 \
    TRAIN_SEQ_LEN=1024 \
    TRAIN_BATCH_TOKENS=524288 \
    MATRIX_LR=0.02 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    RANDOM_MLP_UP_LAYERS="0,1,2,3,4" \
    RANDOM_MLP_UP_RANK=16 \
    RANDOM_MLP_UP_GAIN=1 \
    RANDOM_MLP_UP_BASE_SEED=20260403 \
    RANDOM_MLP_UP_INIT=qr \
    RANDOM_MLP_UP_NUM_EXPERTS=1 \
    FINAL_SLIDING_EVAL=1 \
    EVAL_STRIDE=64 \
    EVAL_SEQ_LEN=1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    ;;
  random_up_moe_12l_5layers_e2)
    RUN_ID="random_up_moe_12l_5layers_e2" \
    NUM_LAYERS=12 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2 \
    TRAIN_SEQ_LEN=1024 \
    TRAIN_BATCH_TOKENS=524288 \
    MATRIX_LR=0.02 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    RANDOM_MLP_UP_LAYERS="0,1,2,3,4" \
    RANDOM_MLP_UP_RANK=0 \
    RANDOM_MLP_UP_GAIN=1 \
    RANDOM_MLP_UP_BASE_SEED=20260403 \
    RANDOM_MLP_UP_INIT=qr \
    RANDOM_MLP_UP_NUM_EXPERTS=2 \
    FINAL_SLIDING_EVAL=1 \
    EVAL_STRIDE=64 \
    EVAL_SEQ_LEN=1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    ;;
  random_up_moe_12l_5layers_e2_rank8)
    RUN_ID="random_up_moe_12l_5layers_e2_rank8" \
    NUM_LAYERS=12 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2 \
    TRAIN_SEQ_LEN=1024 \
    TRAIN_BATCH_TOKENS=524288 \
    MATRIX_LR=0.02 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    RANDOM_MLP_UP_LAYERS="0,1,2,3,4" \
    RANDOM_MLP_UP_RANK=8 \
    RANDOM_MLP_UP_GAIN=1 \
    RANDOM_MLP_UP_BASE_SEED=20260403 \
    RANDOM_MLP_UP_INIT=qr \
    RANDOM_MLP_UP_NUM_EXPERTS=2 \
    FINAL_SLIDING_EVAL=1 \
    EVAL_STRIDE=64 \
    EVAL_SEQ_LEN=1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    ;;
  *)
    echo "Unknown config: $CONFIG" >&2
    echo "Available configs: baseline_12l, random_up_12l_5layers_rank16, random_up_moe_12l_5layers_e2, random_up_moe_12l_5layers_e2_rank8" >&2
    exit 1
    ;;
esac
