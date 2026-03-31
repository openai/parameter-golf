#!/bin/bash
# H-Net Tokenization — Hierarchical Token Processing
# Note: TORCH_COMPILE_DISABLE=1 because dynamic seq_len (coarse/fine) breaks fullgraph
# Usage: SEED=42 bash run.sh

HNET_ENABLED=1 \
HNET_MERGE_FACTOR=2 \
HNET_COARSE_LAYERS=5 \
TORCH_COMPILE_DISABLE=1 \
NGRAM_EVAL=0 \
KNN_LAMBDA=0 \
SEED=${SEED:-42} \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
