#!/bin/bash
# Setup (run once per pod)
cd /workspace
rm -rf parameter-golf
git clone https://github.com/sp00mm/parameter-golf.git
cd parameter-golf
git checkout mtp-auxiliary-heads
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Combined: memory tokens + MTP + longer seq length
NUM_MEMORY_TOKENS=32 \
MTP_NUM_HEADS=2 \
MTP_ALPHA=0.2 \
MTP_ALPHA_DECAY=1 \
MTP_HEAD_LR=0.008 \
TRAIN_SEQ_LEN=2048 \
RUN_ID=combo_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee combo.log
