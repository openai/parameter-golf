#!/usr/bin/env bash
set -euo pipefail

# Quick low-cost run (~4 minutes max)
cd /workspace/parameter-golf

RUN_ID="${RUN_ID:-smoke_sp1024_$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=240
export VAL_LOSS_EVERY=0

mkdir -p logs

torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "logs/${RUN_ID}.log"

echo "Smoke run done: logs/${RUN_ID}.log"
