#!/bin/bash
# Parameter Golf Experiment Suite
# Each experiment runs 1000 steps (~28 min) on RTX 4090
# Compare train_loss at step 1000 against control (2.71)

COMMON="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=1800 ITERATIONS=1000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5"

cd /c/Users/deepc/parameter-golf

echo "============================================"
echo "EXPERIMENT SUITE - Parameter Golf"
echo "Each run: 1000 steps, ~28 min"
echo "Control baseline: train_loss ~2.71 @ step 1000"
echo "============================================"

run_experiment() {
    local name=$1
    local extra_env=$2
    local script=$3
    echo ""
    echo ">>> STARTING: $name"
    echo ">>> Extra env: $extra_env"
    echo ">>> Script: $script"
    echo ">>> $(date)"
    eval "$COMMON RUN_ID=exp_${name} $extra_env python $script" 2>&1 | grep -E "^(step:(1|2|3|4|5|6|7|8|9|10)/|step:100/|step:200/|step:500/|step:1000/|model_params|stopping|peak|final_int8)"
    # Extract final loss
    local final_loss=$(grep "^step:1000/" logs/exp_${name}.txt 2>/dev/null | grep -o "train_loss:[0-9.]*" | cut -d: -f2)
    if [ -z "$final_loss" ]; then
        final_loss=$(grep "^step:" logs/exp_${name}.txt 2>/dev/null | tail -1 | grep -o "train_loss:[0-9.]*" | cut -d: -f2)
    fi
    echo ">>> RESULT $name: train_loss=$final_loss"
    echo ">>> $(date)"
    echo "============================================"
}

# ── Experiment 0: CONTROL (SOTA as-is) ──
run_experiment "control" "" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── Experiment 1: Larger batch (128K instead of 65K) ──
run_experiment "bigger_batch" "TRAIN_BATCH_TOKENS=131072" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── Experiment 2: Higher LR (matrix 0.04 instead of 0.02) ──
run_experiment "higher_lr" "MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── Experiment 3: No BigramHash (test if it actually helps at low steps) ──
run_experiment "no_bigram" "BIGRAM_VOCAB_SIZE=0" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── Experiment 4: 12 layers instead of 10 (more depth, same width) ──
run_experiment "12_layers" "NUM_LAYERS=12" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── Experiment 5: Wider model (640 dim instead of 512) with fewer layers ──
run_experiment "wide_8L" "NUM_LAYERS=8 MODEL_DIM=640" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================"
