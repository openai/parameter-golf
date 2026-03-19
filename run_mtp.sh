#!/bin/bash
# Experiment: more MTP heads + higher weight on top of new baseline
cd /workspace/parameter-golf

MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.2 \
RUN_ID=mtp2_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee mtp2.log
