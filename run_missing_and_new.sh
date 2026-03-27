#!/bin/bash
# Missing experiments + new experiments based on learnings
# RTX 4090, single GPU, NO_COMPILE=1

COMMON="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=1800 ITERATIONS=1000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5"

cd /c/Users/deepc/parameter-golf

run_experiment() {
    local name=$1
    local extra_env=$2
    local script=$3
    echo ""
    echo ">>> STARTING: $name"
    echo ">>> Extra env: $extra_env"
    echo ">>> $(date)"
    eval "$COMMON RUN_ID=exp_${name} $extra_env python $script" 2>&1 | tee -a "logs/exp_${name}_full.log" | grep -E "^(step:(1|2|3|4|5|6|7|8|9|10)/|step:100/|step:200/|step:500/|step:1000/|model_params|stopping|peak|final_int8|val_loss|val_bpb)"
    echo ">>> DONE: $name at $(date)"
    echo "============================================"
}

echo "============================================"
echo "ROUND 2 EXPERIMENTS"
echo "============================================"

# ── MISSING 1: Bigger batch (128K tokens instead of 65K) ──
run_experiment "bigger_batch" "TRAIN_BATCH_TOKENS=131072" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── MISSING 2: 12 layers with enough wallclock (2400s) to finish eval ──
run_experiment "12_layers_v2" "NUM_LAYERS=12 MAX_WALLCLOCK_SECONDS=2400" "records/track_10min_16mb/our_submission/train_gpt.py"

# ── NEW EXPERIMENTS based on learnings ──

# Exp A: Larger BigramHash vocab (20480 vs 10240) — bigram is critical, can we get more from it?
run_experiment "bigram_20k" "BIGRAM_VOCAB_SIZE=20480" "records/track_10min_16mb/our_submission/train_gpt.py"

# Exp B: Higher bigram dim (256 vs 128)
run_experiment "bigram_dim256" "BIGRAM_DIM=256" "records/track_10min_16mb/our_submission/train_gpt.py"

# Exp C: More MLP capacity (3.5x instead of 3x) — MLP is where learning happens
run_experiment "mlp_3.5x" "MLP_MULT=3.5" "records/track_10min_16mb/our_submission/train_gpt.py"

# Exp D: Muon momentum 0.95 instead of 0.99 — faster adaptation with short training
run_experiment "muon_095" "MUON_MOMENTUM=0.95" "records/track_10min_16mb/our_submission/train_gpt.py"

# Exp E: Lower weight decay (0.02 vs 0.04) — might be regularizing too much for 1000 steps
run_experiment "wd_002" "WEIGHT_DECAY=0.02" "records/track_10min_16mb/our_submission/train_gpt.py"

# Exp F: 2048 seq len (matches leaderboard winners)
run_experiment "seq2048" "TRAIN_SEQ_LEN=2048" "records/track_10min_16mb/our_submission/train_gpt.py"

# Exp G: Combined — bigram 20K + MLP 3.5x + seq 2048 (best guesses combined)
run_experiment "combined_v1" "BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 TRAIN_SEQ_LEN=2048" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL ROUND 2 EXPERIMENTS COMPLETE"
echo "$(date)"
