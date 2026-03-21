#!/bin/bash
# Exp 011: 5 unique blocks x 2 loops = 10 effective, dim=704 (PR #21 style)
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export RUN_ID=exp011_5x2_d704
export NUM_UNIQUE_BLOCKS=5
export NUM_LOOPS=2
export EVAL_NUM_LOOPS=2
export MODEL_DIM=704
export NUM_HEADS=8
export NUM_KV_HEADS=4
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export QAT_ENABLED=0
export WARMDOWN_ITERS=1200
python3 train_gpt_recurrent.py
