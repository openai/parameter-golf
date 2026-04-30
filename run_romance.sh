#!/bin/bash
set -e
set -o pipefail

COMMON="MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 VAL_LOSS_EVERY=200 EMA_DECAY=0.997 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

mkdir -p logs

echo "=== exp11_full: d=704 4x3 + XSA+EMA+BH (80 shards) ==="
eval $COMMON RUN_ID=exp11_full torchrun --standalone --nproc_per_node=4 train_exp11.py 2>&1 | tee logs/exp11_full.log

echo "=== exp12_full: d=736 4x3 + XSA+EMA+BH+LoRA(r=8) (80 shards) ==="
eval $COMMON RUN_ID=exp12_full torchrun --standalone --nproc_per_node=4 train_exp12.py 2>&1 | tee logs/exp12_full.log

git add -A && git commit -m "exp11_full + exp12_full: romance variants on 80-shard data" && git push origin baseline-try
echo "=== ALL DONE ==="
