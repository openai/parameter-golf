#!/bin/bash
set -e
set -o pipefail  # so torchrun failures propagate through `tee`

# NOTE: EMA_DECAY=0.997 is added to COMMON because the user spec says 0.997
# but the file defaults are 0.99. This affects exp6 and exp8 (only experiments
# that read EMA_DECAY). exp7/exp9/exp10 ignore it.
COMMON="MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 VAL_LOSS_EVERY=200 EMA_DECAY=0.997 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024"

mkdir -p logs

echo "=== exp6: Full Stack (XSA4+EMA) ==="
eval $COMMON RUN_ID=exp6 torchrun --standalone --nproc_per_node=4 train_exp6.py 2>&1 | tee logs/exp6.log

echo "=== exp7: XSA4 only ==="
eval $COMMON RUN_ID=exp7 torchrun --standalone --nproc_per_node=4 train_exp7.py 2>&1 | tee logs/exp7.log

echo "=== exp8: EMA only ==="
eval $COMMON RUN_ID=exp8 torchrun --standalone --nproc_per_node=4 train_exp8.py 2>&1 | tee logs/exp8.log

echo "=== exp9: Weight-Tied 4x3 ==="
eval $COMMON RUN_ID=exp9 torchrun --standalone --nproc_per_node=4 train_exp9.py 2>&1 | tee logs/exp9.log

echo "=== exp10: BigramHash ==="
eval $COMMON RUN_ID=exp10 torchrun --standalone --nproc_per_node=4 train_exp10.py 2>&1 | tee logs/exp10.log

git add -A && git commit -m "exp6-10: H100x4 600s results" && git push origin baseline-try
echo "=== ALL DONE ==="
