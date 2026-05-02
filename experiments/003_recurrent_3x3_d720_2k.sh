#!/bin/bash
# Experiment 003: Depth recurrence 3x3@720 with QAT (worse than baseline)
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export RUN_ID=exp003_3x3_d720_2k
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export NUM_UNIQUE_BLOCKS=3
export NUM_LOOPS=3
export EVAL_NUM_LOOPS=3  # fixed: same as train
export MODEL_DIM=720
export NUM_HEADS=10
export NUM_KV_HEADS=5
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export QAT_ENABLED=1
export WARMDOWN_ITERS=1200
python3 train_gpt_recurrent.py
