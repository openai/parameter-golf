#!/bin/bash
# Round 4: Validate 16L x 448d config + KV head variants + long scaling run
# Base: 16L, 448dim, 7 heads, 7 KV heads, bigram 20K, MLP 3.5x

BEST="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5 NUM_LAYERS=16 MODEL_DIM=448 NUM_HEADS=7 NUM_KV_HEADS=7 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5"

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
echo "ROUND 4 — 16L CONFIG VARIANTS + LONG RUN — started $(date)"
echo "============================================"

# ── KV head variants (fewer KV heads = fewer params, more room for other things) ──

# A: MHA baseline (7 KV heads = 7 heads, already set)
run_experiment "16L_kv7" "MAX_WALLCLOCK_SECONDS=2400 ITERATIONS=1000" "records/track_10min_16mb/our_submission/train_gpt.py"

# B: GQA with 1 KV head (aggressive sharing)
run_experiment "16L_kv1" "MAX_WALLCLOCK_SECONDS=2400 ITERATIONS=1000 NUM_KV_HEADS=1" "records/track_10min_16mb/our_submission/train_gpt.py"

# C: GQA with 7 heads but only 1 KV head, wider model (480dim, 8 heads)
run_experiment "16L_480d_kv1" "MAX_WALLCLOCK_SECONDS=2400 ITERATIONS=1000 MODEL_DIM=480 NUM_HEADS=8 NUM_KV_HEADS=1" "records/track_10min_16mb/our_submission/train_gpt.py"

# D: 18 layers x 416dim (push depth even further)
run_experiment "18L_416d" "MAX_WALLCLOCK_SECONDS=2400 ITERATIONS=1000 NUM_LAYERS=18 MODEL_DIM=416 NUM_HEADS=8 NUM_KV_HEADS=1" "records/track_10min_16mb/our_submission/train_gpt.py"

# E: 20 layers x 384dim (extreme depth)
run_experiment "20L_384d" "MAX_WALLCLOCK_SECONDS=2400 ITERATIONS=1000 NUM_LAYERS=20 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=1" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── LONG RUN: best 16L config for 3000 steps ──
run_experiment "16L_long_3000" "MAX_WALLCLOCK_SECONDS=5400 ITERATIONS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL ROUND 4 EXPERIMENTS COMPLETE at $(date)"
