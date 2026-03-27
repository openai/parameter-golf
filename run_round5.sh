#!/bin/bash
# Round 5: SWA, sliding eval, warmdown optimization
# Base: 12L 512d, bigram 20K, MLP 3.5x (our best scaling config)
# All experiments use 3000 steps to test at scale

BASE="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=5400 ITERATIONS=3000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 WARMUP_STEPS=5 NUM_LAYERS=12 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5"

cd /c/Users/deepc/parameter-golf

run_experiment() {
    local name=$1
    local extra_env=$2
    local script=$3
    echo ""
    echo ">>> STARTING: $name at $(date)"
    eval "$BASE RUN_ID=exp_${name} $extra_env python $script" 2>&1 | grep -E "^(step:(1|2|3|4|5|6|7|8|9|10)/|step:100/|step:500/|step:1000/|step:1500/|step:2000/|step:2500/|step:3000/|model_params|stopping|peak|final_int8|val_loss|val_bpb|swa:)"
    echo ">>> DONE: $name at $(date)"
    echo "============================================"
}

echo "============================================"
echo "ROUND 5 — SWA / EVAL / WARMDOWN — started $(date)"
echo "============================================"

# ── CONTROL: no SWA, no sliding eval (matches long_3000 from round 3) ──
run_experiment "r5_control" "SWA_ENABLED=0 EVAL_STRIDE=0 WARMDOWN_ITERS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── SWA EXPERIMENTS (all with standard eval, no sliding) ──

# SWA with default settings (start at 0.4 LR frac, every 50 steps)
run_experiment "r5_swa_default" "SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 EVAL_STRIDE=0 WARMDOWN_ITERS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

# SWA starting earlier (0.6 frac = collects more checkpoints)
run_experiment "r5_swa_early" "SWA_ENABLED=1 SWA_START_FRAC=0.6 SWA_EVERY=50 EVAL_STRIDE=0 WARMDOWN_ITERS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

# SWA collecting more frequently (every 25 steps)
run_experiment "r5_swa_freq25" "SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=25 EVAL_STRIDE=0 WARMDOWN_ITERS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── WARMDOWN EXPERIMENTS (with best SWA from above, no sliding eval) ──

# Longer warmdown (5000 iters — more gradual LR decay)
run_experiment "r5_wd5000" "SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 EVAL_STRIDE=0 WARMDOWN_ITERS=5000" "records/track_10min_16mb/our_submission/train_gpt.py"

# Shorter warmdown (1500 iters — more time at full LR)
run_experiment "r5_wd1500" "SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 EVAL_STRIDE=0 WARMDOWN_ITERS=1500" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── SLIDING EVAL EXPERIMENTS (with SWA, test eval-only improvements) ──

# Sliding eval stride=64 (default)
run_experiment "r5_slide64" "SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 WARMDOWN_ITERS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

# Sliding eval stride=32 (denser, slower, potentially better)
run_experiment "r5_slide32" "SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 EVAL_STRIDE=32 EVAL_BATCH_SEQS=16 WARMDOWN_ITERS=3000" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL ROUND 5 EXPERIMENTS COMPLETE at $(date)"
