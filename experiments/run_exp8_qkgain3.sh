#!/bin/bash
# Exp 8 — SP1024 + Depth Recurrence + q_gain=3.0
# Branch: exp/depth-recurrence
# Question: is q_gain=3.0 better than default 1.5?

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
QK_GAIN_INIT=3.0 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp8_qkgain3.log

echo "Exp 8 done."
