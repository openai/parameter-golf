#!/usr/bin/env bash
set -euo pipefail
cd '/home/jovyan/vasiliev/notebooks/parameter-golf'

export RUN_ID='lr_schedule_rsqrt'
export DATA_PATH='./data/datasets/fineweb10B_sp1024/'
export TOKENIZER_PATH='./data/tokenizers/fineweb_1024_bpe.model'
export VOCAB_SIZE=1024
export COMET_ENABLE=1
export EXPERIMENT_NAME='lr_schedule=rsqrt'
export NUM_LAYERS=16
export MODEL_DIM=256
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export USE_COMPILE=1
export LR_SCHEDULE=rsqrt

torchrun --standalone --nproc_per_node=4 train_gpt.py
