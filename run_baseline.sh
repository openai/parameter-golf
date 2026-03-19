#!/bin/bash
# New baseline: upstream train_gpt.py with MTP(1), EMA, QAT, FP16 embed
cd /workspace/parameter-golf

RUN_ID=baseline_v2_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee baseline_v2.log
