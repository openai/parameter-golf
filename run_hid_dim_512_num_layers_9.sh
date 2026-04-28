#!/usr/bin/env bash

export RUN_ID=baseline_test
export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export COMET_ENABLE=1
export COMET_API_KEY="wKvWIXBmWdm5O9w8buIWrqKEV"
export EXPERIMENT_NAME="hid_dim=352, num_layers=12"
export NUM_LAYERS=12
export MODEL_DIM=352
# export NUM_HEADS=8
# export NUM_KV_HEADS=4
# export USE_MHC=1
# export MHC_TYPE=mhc
# export MHC_NUM_STREAMS=4
export USE_COMPILE=1

torchrun --standalone --nproc_per_node=4 train_gpt.py
