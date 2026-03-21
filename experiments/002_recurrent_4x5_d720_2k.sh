#!/bin/bash
# Experiment 002: Depth recurrence 4x5@720 (FAILED - eval bug)
cd /home/ubuntu/parameter-golf
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export RUN_ID=recurrent_4x5_d720_2k
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export NUM_UNIQUE_BLOCKS=4
export NUM_LOOPS=5
export EVAL_NUM_LOOPS=7  # BUG: caused graph mismatch
export MODEL_DIM=720
export NUM_HEADS=10
export NUM_KV_HEADS=5
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export QAT_ENABLED=1
export WARMDOWN_ITERS=1200
python3 train_gpt_recurrent.py
