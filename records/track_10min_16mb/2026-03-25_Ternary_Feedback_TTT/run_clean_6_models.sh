#!/bin/bash
# ============================================================================
# FINAL CLEAN 10-MINUTE RUNNER
# Perfect 6-architecture shootout directly responding to user request.
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

LOG_FILE="clean_6_models.log"
echo "--- CLEAN 6-MODEL RUN START: $(date) ---" > "$LOG_FILE"

# Common settings for 10-minute runs
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
    shift 1
    echo "==========================================================================" | tee -a "$LOG_FILE"
    echo "RUNNING: $name FOR EXACTLY 10 MINS (Seed: 42)" | tee -a "$LOG_FILE"
    echo "==========================================================================" | tee -a "$LOG_FILE"
    
    # Run and capture essential metrics
    SEED=42 ITERATIONS=$ITERATIONS VAL_LOSS_EVERY=$VAL_LOSS_EVERY \
    MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS \
    GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS SLIDING_EVAL=$SLIDING_EVAL \
    SLIDING_EVAL_STRIDE=$SLIDING_EVAL_STRIDE TEMP_SCALING=$TEMP_SCALING \
    env "$@" bash run_mlx_reasoner.sh 2>&1 | tee -a "$LOG_FILE" | grep -E "^(step|val_|final_|model_|koopman|ttt|engram|arch)"
}

# 1. Floor: Plain Ternary Transformer
run_config "1.Floor" \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=0 \
    TTT_ENABLED=0

# 2. Memory: Floor + 3-Order Engram
run_config "2.Memory" \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=1 \
    BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3 \
    TTT_ENABLED=0

# 3. Feedback: Memory + Shared Blocks
run_config "3.Feedback" \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=1 \
    BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3 \
    TTT_ENABLED=0

# 4. Reasoning: Feedback + KoopCaps
run_config "4.Reasoning" \
    ARCHITECTURE=transformer NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 CAPSULE_NUM=16 \
    CAPSULE_DIM=64 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
    BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=3 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback

# 5. Linear: Vanilla Koopman SSM (ISO-params)
run_config "5.Linear" \
    ARCHITECTURE=koopman_ssm NUM_LAYERS=8 MODEL_DIM=256 KOOPMAN_STATE_DIM=192 MLP_MULT=3 \
    EMBED_DIM=128 SHARED_BLOCKS=0 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 BIGRAM_HASH_ENABLED=0 \
    TTT_ENABLED=0

# 6. Champion: Extreme Universal SSM
run_config "6.Champion" \
    ARCHITECTURE=koopman_ssm NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 KOOPMAN_STATE_DIM=384 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=1 \
    CAPSULE_NUM=32 CAPSULE_DIM=256 KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 \
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 ENGRAM_NUM_ORDERS=2 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback

echo "--- CLEAN RUN COMPLETE ---" | tee -a "$LOG_FILE"
