#!/bin/bash
# Exp 11 — 11 Layers + narrow (model_dim=448) + SP4096 + Recurrence
# Branch: exp/depth-recurrence
# Params: ~15.5M (fits in 16 MB with INT8)
# head_dim = 448/8 = 56 (even, OK for RoPE)
# Question: is more layers + narrower better than fewer + wider?

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

NUM_LAYERS=11 \
MODEL_DIM=448 \
VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
RECUR_LAYERS=3,4 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp11_11L_narrow.log

echo "Exp 11 done."
