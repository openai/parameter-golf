#!/bin/bash
# Exp 5 — SP4096 Baseline
# Branch: main (no code changes, env vars only)
# Hypothesis: larger vocab → better bpb, same model architecture as baseline
# Compare to: Exp 3 (baseline SP1024, 2×H100, val_bpb=1.2732)

set -euo pipefail
cd /workspace/parameter-golf
git checkout main

VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  2>&1 | tee /workspace/logs/exp5_sp4096_baseline.log

echo "Exp 5 done."
