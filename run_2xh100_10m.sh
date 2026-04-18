#!/bin/bash
# Quick 10-min test on 2×H100
# ~1500 steps (8×H100 gets ~6000 in 10min, 2× gets 1/4 of that)
# warmdown=300 (20% of ~1500)
TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4 \
RECUR_START_STEP=1000 PARALLEL_START_LAYER=7 \
ROPE_DIMS=16 VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=300 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
