#!/bin/bash
# Memory tokens experiment: 32 learnable tokens as global context
cd /workspace/parameter-golf

NUM_MEMORY_TOKENS=32 \
RUN_ID=memory32_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee memory32.log
