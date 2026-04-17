#!/bin/bash
# Exp 6 — SP4096 + Depth Recurrence
# Branch: exp/depth-recurrence (commit ea1898a)
# Hypothesis: combining SP4096 + recurrence stacks both improvements
# Compare to: Exp 5 (SP4096 baseline) and Exp 4 (recurrence SP1024)

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp6_sp4096_recurrence.log

echo "Exp 6 done."
