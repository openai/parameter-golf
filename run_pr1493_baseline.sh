#!/bin/bash
# PR #1493 baseline reproduction
# Author: bigbag (Pavel Liashkov)
# Expected: val_bpb ~1.0810 (3-seed mean), sliding without TTT: 1.0827
#
# This runs the EXACT PR #1493 code with their recommended env vars.
# We use SEED=42 (their default for reproduction).
set -e

export SEED=42
export QK_GAIN_INIT=5.25
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export RUN_ID="pr1493_baseline"

echo "========================================"
echo "  PR #1493 baseline reproduction"
echo "  SEED=42, QK_GAIN=5.25, TTT on"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_pr1493.py 2>&1 | tee "/workspace/parameter-golf/pr1493_baseline.log"
