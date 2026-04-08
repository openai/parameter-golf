#!/bin/bash

# This script runs the key head-to-head ablations that prove the architectural innovations
# beat both the plain ternary baseline and the stronger no-capsule baseline.

DIR="/Users/akhileshgogikar/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT"
cd "$DIR" || exit 1

echo "=================================================================================="
echo "1. Baseline A — plain ternary only (Seed 42)"
echo "=================================================================================="
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
CAPSULE_ENABLED=0 FEEDBACK_ENABLED=0 BIGRAM_HASH_ENABLED=0 \
VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=0 PARTIAL_ROPE_DIMS=0 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SEED=42 ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo)"
echo ""

echo "=================================================================================="
echo "2. No-capsule Baseline B — feedback + Engram + cheap tricks (Seed 42)"
echo "=================================================================================="
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
CAPSULE_ENABLED=0 FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
VRL_ENABLED=1 XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SEED=42 ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo)"
echo ""

echo "=================================================================================="
echo "3. Hadamard-KoopCaps improved model E (Seed 42)"
echo "=================================================================================="
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
CAPSULE_ENABLED=1 FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
KOOPMAN_ENABLED=1 ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
VRL_ENABLED=1 XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SEED=42 ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|koopman)"
echo ""

echo "=================================================================================="
echo "4. No-capsule Baseline B (Seed 1337)"
echo "=================================================================================="
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
CAPSULE_ENABLED=0 FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
VRL_ENABLED=1 XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SEED=1337 ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(final_|step:200)"
echo ""

echo "=================================================================================="
echo "5. Hadamard-KoopCaps improved model E (Seed 1337)"
echo "=================================================================================="
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
CAPSULE_ENABLED=1 FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
KOOPMAN_ENABLED=1 ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
VRL_ENABLED=1 XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SEED=1337 ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(final_|step:200)"
echo ""

echo "=================================================================================="
echo "6. Latest Full-Stack of Innovation (All optimizations enabled) (Seed 42)"
echo "=================================================================================="
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
CAPSULE_ENABLED=1 FEEDBACK_ENABLED=1 BIGRAM_HASH_ENABLED=1 \
KOOPMAN_ENABLED=1 ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
VRL_ENABLED=1 XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
TTT_ENABLED=1 NGRAM_CACHE_ENABLED=1 EMA_ENABLED=1 \
SEED=42 ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|koopman|ttt|ngram|ema)"
echo ""

echo "=================================================================================="
echo "7. Path 1: Shared-Block Recurrent Reasoner (ISO-FLOP Winner) (Seed 42)"
echo "=================================================================================="
ARCHITECTURE=transformer \
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
SHARED_BLOCKS=2 \
FEEDBACK_ENABLED=0 \
CAPSULE_ENABLED=1 CAPSULE_NUM=16 CAPSULE_DIM=64 \
KOOPMAN_ENABLED=1 KOOPMAN_RANK=4 KOOPMAN_DIAG_INIT=0.9 \
KOOPMAN_CONSISTENCY_WEIGHT=0.005 \
KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=3 \
ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=8192 BIGRAM_HASH_DIM=128 \
ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=2 ENGRAM_INJECT_LAYER=1 \
VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo|arch)"
echo ""

# echo "=================================================================================="
# echo "8. Path 2: Koopman SSM (Attention-free architecture) (Seed 42)"
# echo "=================================================================================="
# ARCHITECTURE=koopman_ssm \
# NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 EMBED_DIM=128 \
# KOOPMAN_STATE_DIM=128 KOOPMAN_MIXER_RANK=4 KOOPMAN_CONV_KERNEL=4 KOOPMAN_DECAY_WINDOW=32 \
# SHARED_BLOCKS=0 \
# FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 BIGRAM_HASH_ENABLED=0 \
# VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=0 \
# ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=300 \
# TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
# TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
# SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
# bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|koopman_ssm|optimizer|eval|turbo|arch)"
# echo ""
