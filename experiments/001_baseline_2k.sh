#!/bin/bash
# Experiment 001: Unmodified baseline, 2K steps
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export WANDB_MODE=disabled  # first run before wandb was added
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export RUN_ID=baseline_2k
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
python3 train_gpt.py
