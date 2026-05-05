#!/bin/bash
# Exp 10 — SP1024 + 3-Layer Recurrence [2,3,4]
# Branch: exp/depth-recurrence
# Block schedule: [0,1,2,3,4,2,3,4,5,6,7,8] — 12 virtual passes
# Question: does recurring 3 layers beat 2?

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
RECUR_LAYERS=2,3,4 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp10_3layer_recur.log

echo "Exp 10 done."
