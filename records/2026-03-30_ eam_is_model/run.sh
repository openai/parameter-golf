#!/bin/bash
set -e
cd "$(dirname "$0")"

# Download data if needed
if [ ! -f ../../data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    python3 ../../data/cached_challenge_fineweb.py --variant sp1024
fi

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=${NGPU:-8} train_gpt.py
