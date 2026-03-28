#!/bin/bash
# Approach B: PR #576 fork — "train larger, quantize harder" (33.6M params, int5 GPTQ)
# Requires: pip install zstandard (for zstd compression)
pip install --break-system-packages zstandard 2>/dev/null

NCCL_IB_DISABLE=1 SEED=${SEED:-1337} \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/run_b.log
