#!/bin/bash
# ============================================================================
# ULTIMATE 10-HOUR ABLATION STUDY
# 12 Rigorous Runs | Sliding Eval | Multi-Seed for Top Contenders
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

LOG_FILE="ultimate_ablation.log"
echo "--- UTIMATE ABLATION START: $(date) ---" > "$LOG_FILE"

# Common settings for all runs
ITERATIONS=500
VAL_LOSS_EVERY=100
MAX_WALLCLOCK_SECONDS=600
TRAIN_BATCH_TOKENS=16384
GRAD_ACCUM_STEPS=2
SLIDING_EVAL=1
SLIDING_EVAL_STRIDE=64
TEMP_SCALING=1

run_config() {
    local name=$1
    local seed=$2
    shift 2
    echo "==========================================================================" | tee -a "$LOG_FILE"
    echo "RUNNING: $name (Seed: $seed)" | tee -a "$LOG_FILE"
    echo "==========================================================================" | tee -a "$LOG_FILE"
    
    # Run and capture essential metrics
    SEED=$seed ITERATIONS=$ITERATIONS VAL_LOSS_EVERY=$VAL_LOSS_EVERY \
    MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS \
    GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS SLIDING_EVAL=$SLIDING_EVAL \
    SLIDING_EVAL_STRIDE=$SLIDING_EVAL_STRIDE TEMP_SCALING=$TEMP_SCALING \
    env "$@" bash run_mlx_reasoner.sh 2>&1 | tee -a "$LOG_FILE" | grep -E "^(step|val_|final_|model_|koopman|ttt|engram|arch)"
}

# --- Phase 2: Core Ablations (1 Seed) ---

# 1. Floor: Plain Ternary Transformer
run_config "1.Floor" 42 \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=0

# 2. Memory: Floor + 3-Order Engram
run_config "2.Memory" 42 \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=1 \
    BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3

# 3. Feedback: Memory + Shared Blocks
run_config "3.Feedback" 42 \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=1 \
    BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3

# --- Phase 3: High-Fidelity Validation (3 Seeds) ---

SEEDS=(42 1337 2026)

# 4. Reasoning: Feedback + KoopCaps
for s in "${SEEDS[@]}"; do
    run_config "4.Reasoning" "$s" \
        ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
        EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 CAPSULE_NUM=16 \
        CAPSULE_DIM=64 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
        BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3
done

# 5. Linear: Vanilla Koopman SSM (ISO-params)
for s in "${SEEDS[@]}"; do
    run_config "5.Linear" "$s" \
        ARCHITECTURE=koopman_ssm NUM_LAYERS=8 MODEL_DIM=256 KOOPMAN_STATE_DIM=192 MLP_MULT=3 \
        EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=0
done

# 6. Champion: Extreme Universal SSM
for s in "${SEEDS[@]}"; do
    run_config "6.Champion" "$s" \
        ARCHITECTURE=koopman_ssm NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
        EMBED_DIM=128 KOOPMAN_STATE_DIM=384 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 \
        CAPSULE_NUM=32 CAPSULE_DIM=256 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 \
        BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=2 \
        TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback
done

echo "--- ULTIMATE ABLATION COMPLETE: $(date) ---" | tee -a "$LOG_FILE"
