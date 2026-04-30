#!/bin/bash
set -e
set -o pipefail

COMMON="MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 VAL_LOSS_EVERY=200 EMA_DECAY=0.997 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

mkdir -p logs

echo "=== exp6_full: XSA4+EMA (80 shards) ==="
eval $COMMON RUN_ID=exp6_full torchrun --standalone --nproc_per_node=4 train_exp6.py 2>&1 | tee logs/exp6_full.log

echo "=== exp9_full: Weight-Tied 4x3 (80 shards) ==="
eval $COMMON RUN_ID=exp9_full torchrun --standalone --nproc_per_node=4 train_exp9.py 2>&1 | tee logs/exp9_full.log

git add -A && git commit -m "exp6_full + exp9_full: H100x4 600s full-data results" && git push origin baseline-try
echo "=== ALL DONE ==="
