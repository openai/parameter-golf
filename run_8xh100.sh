#!/bin/bash
# Setup (run once per pod)
cd /workspace
rm -rf parameter-golf
git clone https://github.com/sp00mm/parameter-golf.git
cd parameter-golf
git checkout mtp-auxiliary-heads
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Full submission run: 8xH100 SXM
NUM_MEMORY_TOKENS=32 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
FP16_EMBED_EXPORT=1 \
MUON_WEIGHT_DECAY=0.02 \
RUN_ID=submission_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee submission.log
