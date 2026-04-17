#!/bin/bash
# Exp 13 — SP4096 + Depth Recurrence + train_seq_len=2048
# Branch: exp/depth-recurrence
# Question: longer context helps? Top submissions use 2048.

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
TRAIN_SEQ_LEN=2048 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp13_seq2048.log

echo "Exp 13 done."
