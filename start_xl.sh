#!/bin/bash
cd ~/parameter-golf/parameter-golf || exit 1
mkdir -p logs
pkill -f train_gpt_optimized.py || true
rm -f logs/exp_xl.txt

RUN_ID=exp_xl \
NUM_LAYERS=10 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=10 \
TRAIN_BATCH_TOKENS=1000000 \
ITERATIONS=150000 \
MAX_WALLCLOCK_SECONDS=21600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
PYTHONUNBUFFERED=1 \
python3 train_gpt_optimized.py | tee logs/exp_xl.txt
