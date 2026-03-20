#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${RUN_ID:=antdx316_depth12_dim416_sw4096_s64_8gpu}"
: "${VOCAB_SIZE:=1024}"
: "${NUM_LAYERS:=12}"
: "${MODEL_DIM:=416}"
: "${NUM_HEADS:=8}"
: "${NUM_KV_HEADS:=4}"
: "${MLP_MULT:=2}"
: "${TIE_EMBEDDINGS:=1}"
: "${TRAIN_SEQ_LEN:=1024}"
: "${TRAIN_BATCH_TOKENS:=524288}"
: "${ITERATIONS:=20000}"
: "${WARMUP_STEPS:=20}"
: "${WARMDOWN_ITERS:=1200}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${VAL_LOSS_EVERY:=0}"
: "${VAL_BATCH_SIZE:=524288}"
: "${TRAIN_LOG_EVERY:=200}"
: "${EVAL_SEQ_LEN:=4096}"
: "${SLIDING_WINDOW_STRIDE:=64}"

export DATA_PATH TOKENIZER_PATH RUN_ID VOCAB_SIZE NUM_LAYERS MODEL_DIM NUM_HEADS NUM_KV_HEADS MLP_MULT
export TIE_EMBEDDINGS TRAIN_SEQ_LEN TRAIN_BATCH_TOKENS ITERATIONS WARMUP_STEPS WARMDOWN_ITERS
export MAX_WALLCLOCK_SECONDS VAL_LOSS_EVERY VAL_BATCH_SIZE TRAIN_LOG_EVERY EVAL_SEQ_LEN SLIDING_WINDOW_STRIDE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p logs

echo "[run] RUN_ID=$RUN_ID"
echo "[run] sliding eval: EVAL_SEQ_LEN=$EVAL_SEQ_LEN SLIDING_WINDOW_STRIDE=$SLIDING_WINDOW_STRIDE"

torchrun --standalone --nnodes=1 --nproc_per_node=8 train_gpt.py | tee "logs/${RUN_ID}.txt"

echo
echo "[summary]"
grep -E 'eval_seq_len:|sliding_window_stride:|Total submission size int8\+zlib:|final_int8_zlib_roundtrip |final_int8_zlib_roundtrip_exact ' "logs/${RUN_ID}.txt" || true
