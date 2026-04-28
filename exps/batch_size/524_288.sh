#!/usr/bin/env bash
cd '/home/jovyan/vasiliev/notebooks/parameter-golf'

export RUN_ID='bs_524_288'
export DATA_PATH='./data/datasets/fineweb10B_sp1024/'
export TOKENIZER_PATH='./data/tokenizers/fineweb_1024_bpe.model'
export BATCH_SIZE=524288
export COMET_ENABLE=1
export EXPERIMENT_NAME='bs=524_288'
export NUM_LAYERS=16
export MODEL_DIM=256
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export USE_COMPILE=1

torchrun --standalone --nproc_per_node=4 train_gpt.py
