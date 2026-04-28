#!/bin/bash
# QAT training: fake quantize STE during warmdown (frac >= 0.28)
# Compare against PR #1493 baseline:
#   pre-quant non-sliding: 1.08757
#   post-quant non-sliding: 1.10014 (quant gap = 0.01257)
#   post-quant sliding: 1.08329
#   post-quant TTT: 1.08103

export SEED=42
export QK_GAIN_INIT=5.25
export QAT_ENABLED=1
export QAT_START_FRAC=0.28
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export RUN_ID=qat_run1
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=2000

cd /workspace/parameter-golf
torchrun --standalone --nproc_per_node=8 train_qat.py 2>&1 | tee logs/qat_run1.log
