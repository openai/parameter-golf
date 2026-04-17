#!/bin/bash
# Exp 7 — SP1024 + Depth Recurrence + QK-Gain 5.25
# Branch: exp/depth-recurrence (commit ea1898a)
# Hypothesis: higher q_gain sharpens attention → better quality (top submissions used 5.25)
# Compare to: Exp 6 (same but q_gain=1.5 default)

set -euo pipefail
cd /workspace/parameter-golf
git checkout exp/depth-recurrence

VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
QK_GAIN_INIT=5.25 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp7_sp4096_recur_qkgain.log

echo "Exp 7 done."
