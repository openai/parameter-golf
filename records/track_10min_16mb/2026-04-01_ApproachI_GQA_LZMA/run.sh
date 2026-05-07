#!/bin/bash
# Approach I: GQA + LZMA + Selective Pruning (adopted from merged SOTA #1019)
pip install --break-system-packages zstandard 2>/dev/null
NCCL_IB_DISABLE=1 SEED=${SEED:-1337} \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/run_i.log
