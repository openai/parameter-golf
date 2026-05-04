#!/bin/bash
cd ~/parameter-golf/parameter-golf || exit 1
mkdir -p logs
pkill -f train_gpt_optimized.py || true
rm -f logs/exp_gpu.txt
RUN_ID=exp_gpu NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=8 TRAIN_BATCH_TOKENS=2000000 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=14400 DATA_PATH=./data/datasets/fineweb10B_sp1024 PYTHONUNBUFFERED=1 python3 train_gpt_optimized.py | tee logs/exp_gpu.txt
