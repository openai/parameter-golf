#!/bin/bash

RUN_ID=qwen3_tiny \
QWEN_MODEL_ID=Qwen/Qwen3-1.7B-Base \
MATRIX_LR=0.04 \
TINY_VOCAB_DIM=512 \
VAL_BATCH_SIZE=65536 \
TRAIN_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS=1200 \
uv run torchrun --standalone --nproc_per_node=2 qwen3_tiny_vocab.py

RUN_ID=qwen3_tiny_mlr0.001 \
QWEN_MODEL_ID=Qwen/Qwen3-1.7B-Base \
MATRIX_LR=0.001 \
TINY_VOCAB_DIM=512 \
VAL_BATCH_SIZE=65536 \
TRAIN_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS=1200 \
uv run torchrun --standalone --nproc_per_node=2 qwen3_tiny_vocab.py

RUN_ID=qwen3_tiny_mlr0.0002 \
QWEN_MODEL_ID=Qwen/Qwen3-1.7B-Base \
MATRIX_LR=0.0002 \
TINY_VOCAB_DIM=512 \
VAL_BATCH_SIZE=65536 \
TRAIN_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS=1200 \
uv run torchrun --standalone --nproc_per_node=2 qwen3_tiny_vocab.py

# ITERATIONS=5 \
# WARMUP_STEPS=0 \
# VAL_LOSS_EVERY=0 \
# TRAIN_LOG_EVERY=1 \