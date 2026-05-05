#!/bin/bash
# Exp 013: 8 unique blocks x 2 loops = 16 effective, dim=640 (more unique blocks)
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export RUN_ID=exp013_8x2_d640
export NUM_UNIQUE_BLOCKS=8
export NUM_LOOPS=2
export EVAL_NUM_LOOPS=2
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export QAT_ENABLED=0
export WARMDOWN_ITERS=1200
python3 train_gpt_recurrent.py
