#!/bin/bash
cd ~/parameter-golf/parameter-golf || exit 1
mkdir -p logs
mkdir -p models

# Helper to run an experiment with optimized script
run_exp() {
    echo "Starting $RUN_ID..."
    PYTHONUNBUFFERED=1 python3 train_gpt_optimized.py | tee logs/$RUN_ID.txt
    cp final_model.int8.ptz models/$RUN_ID.ptz
}

# EXP 1 (Deep Stable)
export RUN_ID=exp_deep_stable
export NUM_LAYERS=12
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=10
export TRAIN_BATCH_TOKENS=393216
export EMBED_LR=0.4
export TIED_EMBED_LR=0.03
export WARMUP_STEPS=40
export WARMDOWN_ITERS=2000
export MAX_WALLCLOCK_SECONDS=43200
export DATA_PATH=./data/datasets/fineweb10B_sp1024
run_exp

# EXP 2 (Wide Aggressive)
export RUN_ID=exp_wide_aggressive
export NUM_LAYERS=10
export MODEL_DIM=768
export NUM_HEADS=12
export NUM_KV_HEADS=12
export TRAIN_BATCH_TOKENS=524288
export EMBED_LR=0.5
export TIED_EMBED_LR=0.035
export WARMUP_STEPS=30
export MAX_WALLCLOCK_SECONDS=43200
export DATA_PATH=./data/datasets/fineweb10B_sp1024
run_exp

# EXP 3 (Balanced Peak)
export RUN_ID=exp_balanced_peak
export NUM_LAYERS=12
export MODEL_DIM=768
export NUM_HEADS=12
export NUM_KV_HEADS=12
export TRAIN_BATCH_TOKENS=524288
export EMBED_LR=0.42
export TIED_EMBED_LR=0.032
export WARMUP_STEPS=50
export WARMDOWN_ITERS=2500
export MAX_WALLCLOCK_SECONDS=43200
export DATA_PATH=./data/datasets/fineweb10B_sp1024
run_exp

echo "All experiments completed. Logs available in logs folder."
ls -lh logs
