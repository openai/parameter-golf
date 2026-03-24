#!/usr/bin/env bash
set -euo pipefail
# Авто-сгенерировано из check_model_size.ipynb
# int8_payload_bytes=12954112

cd '/home/jovyan/vasiliev/notebooks/parameter-golf'



export RUN_ID='golf_L16_d256_mlp3_h8_kv4'
export DATA_PATH='./data/datasets/fineweb10B_sp1024/'
export TOKENIZER_PATH='./data/tokenizers/fineweb_1024_bpe.model'
export VOCAB_SIZE=1024
export COMET_ENABLE=1
# ключ не вшит: см. .comet_api_key в launch_grid_generated/
export EXPERIMENT_NAME='L=16 d=256 mlp=3 nh=8 nkv=4 | params=9,717,888 int8≈12.354MiB'
export NUM_LAYERS=16
export MODEL_DIM=256
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export USE_COMPILE=1

torchrun --standalone --nproc_per_node=4 train_gpt.py
