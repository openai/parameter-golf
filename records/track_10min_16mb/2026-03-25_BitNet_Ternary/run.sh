#!/bin/bash
# BitNet Ternary Training Script for 8×H100

# Usage:
#   ./run.sh          # Run with seed 1337
#   ./run.sh 42       # Run with seed 42
#   ./run.sh 2025     # Run with seed 2025

SEED=${1:-1337}

echo "Running BitNet Ternary with SEED=$SEED"

TERNARY_ENABLED=1 \
EMA_ENABLED=0 \
SWA_ENABLED=0 \
NUM_LAYERS=18 \
MODEL_DIM=640 \
SEED=$SEED \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run_seed${SEED}.log

echo "Done! Check logs/run_seed${SEED}.log for results"
