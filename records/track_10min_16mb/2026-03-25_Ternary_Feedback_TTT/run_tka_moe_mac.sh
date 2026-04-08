#!/bin/bash
# ============================================================================
# TERNARY KOOPMAN-ATTENTION MOE HYBRID (TKA-MoE) - MLX BENCHMARK
# Runs the new MoE hybrid architecture for exactly 10 minutes.
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

LOG_FILE="moe_mac_benchmark.log"
echo "--- MOE HYBRID 10-MIN RUN START: $(date) ---" > "$LOG_FILE"

# Common settings for 10-minute runs
ITERATIONS=1500
VAL_LOSS_EVERY=100
MAX_WALLCLOCK_SECONDS=600
TRAIN_BATCH_TOKENS=16384
GRAD_ACCUM_STEPS=2
SLIDING_EVAL=1
SLIDING_EVAL_STRIDE=64
TEMP_SCALING=1

run_config() {
    local name=$1
    shift 1
    echo "==========================================================================" | tee -a "$LOG_FILE"
    echo "RUNNING: $name FOR EXACTLY 10 MINS (Seed: 42)" | tee -a "$LOG_FILE"
    echo "==========================================================================" | tee -a "$LOG_FILE"
    
    # Run and capture essential metrics
    SEED=42 ITERATIONS=$ITERATIONS VAL_LOSS_EVERY=$VAL_LOSS_EVERY \
    MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS \
    GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS SLIDING_EVAL=$SLIDING_EVAL \
    SLIDING_EVAL_STRIDE=$SLIDING_EVAL_STRIDE TEMP_SCALING=$TEMP_SCALING \
    env "$@" bash run_mlx_reasoner.sh >> "$LOG_FILE" 2>&1
}

# TKA-MoE: Alternating Attention + Koopman SSM layers with Sparse Mixture of Experts
run_config "TKA-MoE" \
    ARCHITECTURE=hybrid NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=256 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 CAPSULE_NUM=16 \
    CAPSULE_DIM=128 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_STATE_DIM=128 KOOPMAN_MIXER_RANK=4 \
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback \
    MOE_ENABLED=1 MOE_NUM_EXPERTS=16 MOE_TOP_K=2 MOE_ROUTER_AUX_LOSS_COEF=0.01

echo "--- RUN COMPLETE ---" | tee -a "$LOG_FILE"
