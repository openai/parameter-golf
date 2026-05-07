#!/bin/bash
# Approach D: TurboQuant-guided mixed precision quantization
# V/O and MLP at int3 in middle layers, Q/K at int5 everywhere, boundary layers at int5
pip install --break-system-packages zstandard 2>/dev/null
NCCL_IB_DISABLE=1 SEED=${SEED:-1337} \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/run_d.log
