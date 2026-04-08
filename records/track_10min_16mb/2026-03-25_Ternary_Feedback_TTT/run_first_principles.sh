#!/bin/bash
# ============================================================================
# FIRST-PRINCIPLES ARCHITECTURE BENCHMARK
# ============================================================================
# Two architectures designed from first principles to prove that our
# innovations provide genuine value at ISO-FLOP / ISO-parameter budget.
#
# Path 1: Shared-Block Recurrent Reasoner
#   - 2 unique transformer blocks, applied twice across 4 positions
#   - Same FLOP count as baseline (4 block traversals)
#   - Freed params → Engram (8192 buckets!) + KoopCaps + feedback
#   - NO explicit feedback loop (shared weights ARE the feedback)
#
# Path 2: Koopman State Space Model
#   - Replaces Self-Attention with O(T) Koopman linear recurrence
#   - 8 layers in same FLOP budget as 4 attention layers
#   - MLP_MULT=2 (half width, double depth = same total MLP budget)
#   - Causal decay window=32 with Hadamard rotation
#
# Baseline: 4-layer plain ternary transformer (~4.01M params, ~425ms/step)
# ============================================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "============================================================================"
echo "FIRST-PRINCIPLES ARCHITECTURE BENCHMARK"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

# --------------------------------------------------------------------------
# Path 1: Shared-Block Recurrent Reasoner
# --------------------------------------------------------------------------
# First-principles design:
#   - 2 unique blocks × 2 applications = 4 effective layers = ISO-FLOP
#   - Weight sharing means the 2nd pass through the same weights
#     acts as implicit "correction" of the 1st pass
#   - Per-layer scalars (attn_scale, mlp_scale, resid_mix) let each
#     position use the shared weights differently (ALBERT-style adapters)
#   - Freed ~1.44M params reinvested in:
#     • Big Engram (8192 buckets, 128-dim, 4h2o) — proven valuable
#     • Full KoopCaps (16 caps, 64-dim, rank-4 Koopman) — reasoning state
#     • NO explicit feedback loop (saving FLOP overhead entirely)
#   - LN Scale Damping ON (position-aware, not hardcoded in shared blocks)
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "PATH 1: Shared-Block Recurrent Reasoner (SHARED_BLOCKS=2)"
echo "=========================================================================="

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
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo|arch)"
echo ""

# --------------------------------------------------------------------------
# Path 2: Koopman State Space Model
# --------------------------------------------------------------------------
# Radical first-principles redesign:
#   - Self-attention replaced entirely by Koopman token mixer
#   - Koopman mixer: proj_in → short_conv → causal_decay_scan → proj_out
#     All O(T) per token, no O(T²) attention
#   - 8 layers with MLP_MULT=2 ≈ same total params as 4 layers MLP_MULT=4
#     (Same total MLP capacity, spread across more depth)
#   - Hadamard-rotated diagonal + low-rank evolution in state space
#   - No U-Net structure, no skip connections, no feedback loop
#   - Just a clean, deep stack of Koopman blocks
#   - Engram disabled to maintain truly ISO-parameter
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "PATH 2: Koopman State Space Model (8 layers, no attention)"
echo "=========================================================================="

ARCHITECTURE=koopman_ssm \
NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 EMBED_DIM=128 \
KOOPMAN_STATE_DIM=128 KOOPMAN_MIXER_RANK=4 KOOPMAN_CONV_KERNEL=4 KOOPMAN_DECAY_WINDOW=32 \
SHARED_BLOCKS=0 \
FEEDBACK_ENABLED=0 \
CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 \
BIGRAM_HASH_ENABLED=0 \
VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|koopman_ssm|optimizer|eval|turbo|arch)"
echo ""

# --------------------------------------------------------------------------
# Baseline (control): Plain 4-layer transformer
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "BASELINE: Plain 4-layer ternary transformer (control)"
echo "=========================================================================="

ARCHITECTURE=transformer \
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
SHARED_BLOCKS=0 \
FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 \
BIGRAM_HASH_ENABLED=0 \
VRL_ENABLED=0 XSA_START_LAYER=-1 LN_SCALE_DAMPING=0 PARTIAL_ROPE_DIMS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|optimizer|eval|turbo|arch)"
echo ""

echo "=========================================================================="
echo "BENCHMARK COMPLETE at: $(date)"
echo "=========================================================================="
