#!/bin/bash
# Quick experiment runner - just train 1000 steps, compare train_loss
# Skip the expensive post-training eval
cd /c/Users/deepc/parameter-golf

BASE="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=1800 ITERATIONS=1000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5"
SCRIPT="records/track_10min_16mb/our_submission/train_gpt.py"

echo "============================================"
echo "CONTROL: train_loss=2.6553 @ step 1000"
echo "         val_bpb=1.5298 (post-quant)"
echo "============================================"

run() {
    local name=$1; shift
    echo ""; echo ">>> EXP: $name | $(date)"
    eval "$BASE RUN_ID=exp_${name} $@ python $SCRIPT" 2>&1 | grep -E "^(model_params|step:1000)" || true
    local loss=$(grep "^step:1000/" logs/exp_${name}.txt 2>/dev/null | grep -oP "train_loss:\K[0-9.]+")
    echo ">>> $name: loss=$loss (control=2.6553, lower=better)"
}

# Exp 1: Higher LR — maybe we're under-learning at 1000 steps
run "high_lr" "MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.06"

# Exp 2: 12 layers — more depth, same budget
run "12_layers" "NUM_LAYERS=12"

# Exp 3: Wider (640 dim) + 8 layers — width vs depth tradeoff
run "wide_8L" "NUM_LAYERS=8 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5"

# Exp 4: No bigram — isolate its contribution
run "no_bigram" "BIGRAM_VOCAB_SIZE=0"

# Exp 5: Bigger batch (more tokens per step, fewer steps)
run "big_batch" "TRAIN_BATCH_TOKENS=131072"

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
