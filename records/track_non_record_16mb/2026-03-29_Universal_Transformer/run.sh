#!/bin/bash
# Universal Transformer — Depth Recurrence with Weight Sharing
# Usage: SEED=42 bash run.sh

USE_UNIVERSAL=1 \
UNIVERSAL_UNIQUE_BLOCKS=3 \
UNIVERSAL_ITERATIONS=4 \
NGRAM_EVAL=0 \
KNN_LAMBDA=0 \
SEED=${SEED:-42} \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
