#!/bin/bash
# Round 3: Long validation run + architectural experiments
# Base config: 12L, bigram 20K, MLP 3.5x (our new best)

BEST="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5 NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5"

cd /c/Users/deepc/parameter-golf

run_experiment() {
    local name=$1
    local extra_env=$2
    local script=$3
    echo ""
    echo ">>> STARTING: $name at $(date)"
    eval "$BEST RUN_ID=exp_${name} $extra_env python $script" 2>&1 | grep -E "^(step:(1|2|3|4|5|6|7|8|9|10)/|step:100/|step:200/|step:500/|step:1000/|step:1500/|step:2000/|step:2500/|step:3000/|model_params|stopping|peak|final_int8|val_loss|val_bpb)"
    echo ">>> DONE: $name at $(date)"
    echo "============================================"
}

echo "============================================"
echo "ROUND 3 — LONG RUN + ARCH EXPERIMENTS — started $(date)"
echo "============================================"

# ── LONG RUN: 3000 steps to validate scaling (90 min + eval) ──
run_experiment "long_3000" "ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=5400" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── ARCH A: SwiGLU MLP (used by Llama, may compress better) ──
# Need to add USE_SWIGLU support to the submission script
# For now test via train_exp.py which already has it
run_experiment "swiglu" "ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=2400 USE_SWIGLU=1" "train_exp.py"

# ── ARCH B: Depth recurrence — share 6 blocks, repeat 2x = 12 effective layers ──
# Halves unique params, lets us pack more into 16MB
run_experiment "depth_recur_6x2" "ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=2400 DEPTH_RECURRENCE=6" "train_exp.py"

# ── ARCH C: Depth recurrence 4x3 — share 4 blocks, repeat 3x = 12 effective layers ──
run_experiment "depth_recur_4x3" "ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=2400 DEPTH_RECURRENCE=4" "train_exp.py"

# ── ARCH D: Trigram hash — add 3-token context on top of bigram ──
run_experiment "trigram" "ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=2400 TRIGRAM_VOCAB_SIZE=10240" "train_exp.py"

# ── ARCH E: 14 layers — push depth further, more params but better with quantization ──
run_experiment "14_layers" "ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=2400 NUM_LAYERS=14" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── ARCH F: Narrower + deeper — 16L x 448dim (similar param count, more depth) ──
run_experiment "16L_448d" "ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=2400 NUM_LAYERS=16 MODEL_DIM=448 NUM_HEADS=7 NUM_KV_HEADS=7" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL ROUND 3 EXPERIMENTS COMPLETE at $(date)"
