#!/usr/bin/env bash
set -euo pipefail
# Авто-сгенерировано из check_model_size.ipynb
# int8_payload_bytes=10242304

cd '/home/jovyan/vasiliev/notebooks/parameter-golf'



export RUN_ID='golf_L8_d320_mlp4_h8_kv4'
export DATA_PATH='./data/datasets/fineweb10B_sp1024/'
export TOKENIZER_PATH='./data/tokenizers/fineweb_1024_bpe.model'
export VOCAB_SIZE=1024
export COMET_ENABLE=1
# ключ не вшит: см. .comet_api_key в launch_grid_generated/
export EXPERIMENT_NAME='L=8 d=320 mlp=4 nh=8 nkv=4 | params=9,350,464 int8≈9.768MiB'
export NUM_LAYERS=8
export MODEL_DIM=320
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export USE_COMPILE=1

torchrun --standalone --nproc_per_node=4 train_gpt.py
