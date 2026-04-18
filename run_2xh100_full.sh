#!/bin/bash
# Full run on 2×H100 — equivalent to 8×H100 10min (~6000 steps)
# 40 min wall time (8 GPUs × 10min = 80 GPU-min → 80/2 = 40min)
# warmdown=1200 (20% of ~6000 steps)
TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4 \
RECUR_START_STEP=1000 PARALLEL_START_LAYER=7 \
ROPE_DIMS=16 VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=2400 WARMDOWN_ITERS=1200 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
