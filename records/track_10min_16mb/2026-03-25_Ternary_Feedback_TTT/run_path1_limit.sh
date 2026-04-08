#!/bin/bash
# ============================================================================
# PATH 1 LIMIT TEST (SATURATING 3.08M BUDGET)
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "=========================================================================="
echo "PATH 1 LIMIT: Shared-Block M-Capsules + 3-Grams (Seed 42)"
echo "=========================================================================="

ARCHITECTURE=transformer \
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
SHARED_BLOCKS=2 \
FEEDBACK_ENABLED=0 \
CAPSULE_ENABLED=1 CAPSULE_NUM=32 CAPSULE_DIM=128 \
KOOPMAN_ENABLED=1 KOOPMAN_RANK=4 KOOPMAN_DIAG_INIT=0.9 \
KOOPMAN_CONSISTENCY_WEIGHT=0.005 \
KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=3 \
ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 \
ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=3 ENGRAM_INJECT_LAYER=1 \
VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=1 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|koopman|optimizer|eval|turbo|arch)"
echo ""

echo "=========================================================================="
echo "BENCHMARK COMPLETE at: $(date)"
echo "=========================================================================="
