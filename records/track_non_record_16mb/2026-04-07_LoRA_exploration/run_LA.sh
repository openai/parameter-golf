#!/bin/bash
# Learned adapters experiment on top of the baseline trainer.
# Usage: SEED=42 bash run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ITERATIONS=12000 \
WARMDOWN_ITERS=720 \
SEED=${SEED:-42} \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=4 "$SCRIPT_DIR/train_gpt.py"
