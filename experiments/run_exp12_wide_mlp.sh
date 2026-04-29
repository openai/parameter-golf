#!/bin/bash
# Exp 12 — Wide MLP (mlp_mult=3) + narrow (model_dim=384) + SP1024 + Recurrence
# Branch: exp/depth-recurrence
# Params: ~13.5M (fits in 16 MB with INT8)
# head_dim = 384/8 = 48 (even, OK for RoPE)
# Question: is wider MLP worth the reduced model_dim?

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

MLP_MULT=3 \
MODEL_DIM=384 \
VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
RECUR_LAYERS=3,4 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp12_wide_mlp.log

echo "Exp 12 done."
