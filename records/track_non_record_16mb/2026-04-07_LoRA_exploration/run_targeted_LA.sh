#!/bin/bash
# Targeted learned adapters experiment on top of the baseline trainer.
# Usage: SEED=42 bash run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ITERATIONS=12000 \
WARMDOWN_ITERS=720 \
USE_LEARNED_ADAPTERS=${USE_LEARNED_ADAPTERS:-1} \
LEARNED_ADAPTER_RANK=${LEARNED_ADAPTER_RANK:-64} \
LEARNED_ADAPTER_LAYERS=${LEARNED_ADAPTER_LAYERS:-all} \
LEARNED_ADAPTER_TARGETS=${LEARNED_ADAPTER_TARGETS:-all} \
SEED=${SEED:-42} \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=1 "$SCRIPT_DIR/train_gpt.py"
