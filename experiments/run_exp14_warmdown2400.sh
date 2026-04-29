#!/bin/bash
# Exp 14 — SP1024 + Depth Recurrence + warmdown_iters=2400
# Branch: exp/depth-recurrence
# Question: longer warmdown helps? (default 1200, top submissions use 3500)

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
WARMDOWN_ITERS=2400 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp14_warmdown2400.log

echo "Exp 14 done."
