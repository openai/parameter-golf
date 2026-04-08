#!/bin/bash
# ============================================================================
# KOOPMAN SSM V2 BENCHMARK
# ============================================================================
# Upgraded Koopman SSM targeting ~3.08M params (ISO-FLOP/Parameter)
#
# Changes VS V1:
#   - Layers 8 -> 6 (shorter recurrence path)
#   - State Dim 128 -> 192 (wider recurrence bandwidth per layer)
#   - MLP Mult 2 -> 3 (stronger per-layer feed-forward)
#   - Strict initialization for causal scan stability
#   - Engram Hash (8192 buckets) enabled!
# ============================================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "=========================================================================="
echo "PATH 2 V2: Koopman SSM + Engram Memory (Seed 42)"
echo "=========================================================================="

ARCHITECTURE=koopman_ssm \
NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=3 EMBED_DIM=128 \
KOOPMAN_STATE_DIM=192 KOOPMAN_MIXER_RANK=4 KOOPMAN_CONV_KERNEL=4 KOOPMAN_DECAY_WINDOW=32 \
SHARED_BLOCKS=0 \
FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 \
BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 \
ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=2 ENGRAM_INJECT_LAYER=1 \
VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=1 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|koopman_ssm|optimizer|eval|turbo|arch)"
echo ""

echo "=========================================================================="
echo "BENCHMARK COMPLETE at: $(date)"
echo "=========================================================================="
