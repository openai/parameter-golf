#!/bin/bash
# ============================================================================
# 1-HOUR MARATHON LIMIT TESTS
# Validating true architecture efficiency without compiler-limit starvation
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

LOG_FILE="1hr_marathon.log"
echo "--- 1-HOUR MARATHON START: $(date) ---" > "$LOG_FILE"

# Common settings for 1-hour runs
ITERATIONS=3000
VAL_LOSS_EVERY=200
MAX_WALLCLOCK_SECONDS=3600
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
    echo "RUNNING MARATHON: $name (Seed: $seed) over 3600 seconds" | tee -a "$LOG_FILE"
    echo "==========================================================================" | tee -a "$LOG_FILE"
    
    # Run and capture essential metrics
    SEED=$seed ITERATIONS=$ITERATIONS VAL_LOSS_EVERY=$VAL_LOSS_EVERY \
    MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS \
    GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS SLIDING_EVAL=$SLIDING_EVAL \
    SLIDING_EVAL_STRIDE=$SLIDING_EVAL_STRIDE TEMP_SCALING=$TEMP_SCALING \
    env "$@" bash run_mlx_reasoner.sh 2>&1 | tee -a "$LOG_FILE" | grep -E "^(step|val_|final_|model_|koopman|ttt|engram|arch)"
}

# 1. Linear (Vanilla Koopman SSM)
run_config "Linear_Vanilla_Koopman" 42 \
    ARCHITECTURE=koopman_ssm NUM_LAYERS=8 MODEL_DIM=256 KOOPMAN_STATE_DIM=192 MLP_MULT=3 \
    EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=0 \
    TTT_ENABLED=0

# 2. Champion (Extreme Universal SSM)
run_config "Champion_Universal_Koopman" 42 \
    ARCHITECTURE=koopman_ssm NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 KOOPMAN_STATE_DIM=384 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 \
    CAPSULE_NUM=32 CAPSULE_DIM=256 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 \
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=2 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback

# 3. Reasoning (Transformer + Shared)
run_config "Reasoning_Transformer" 42 \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 CAPSULE_NUM=16 \
    CAPSULE_DIM=64 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
    BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback

echo "--- 1-HOUR MARATHON COMPLETE: $(date) ---" | tee -a "$LOG_FILE"
