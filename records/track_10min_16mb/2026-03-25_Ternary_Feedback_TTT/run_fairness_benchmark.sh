#!/bin/bash
# ============================================================================
# APPLES-TO-APPLES FAIRNESS BENCHMARK
# ============================================================================
# Three experiments proving architectural innovations are genuinely superior:
#
#   Run 1: Full current model (all innovations ON)
#   Run 2: ISO-PARAMETER (all innovations, matched to baseline param count)
#   Run 3: ISO-FLOP (zero-cost tricks only, matched params + step time)
#
#   Baseline ref from ablation_results.txt:
#     model_params:3085840 | step ~667ms | final BPB 2.5567
# ============================================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "============================================================================"
echo "APPLES-TO-APPLES FAIRNESS BENCHMARK"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

# --------------------------------------------------------------------------
# Run 1: Full Current Model (Control)
# All innovations at their normal dimensions.
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "RUN 1: Full Current Model — All Innovations (Control)"
echo "=========================================================================="

NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
FEEDBACK_ENABLED=1 FEEDBACK_DIM=64 FEEDBACK_SKETCH_TOKENS=4 FEEDBACK_PASSES=1 \
EVAL_FEEDBACK_PASSES=2 FEEDBACK_EVERY=2 \
CAPSULE_ENABLED=1 CAPSULE_NUM=16 CAPSULE_DIM=64 \
KOOPMAN_ENABLED=1 KOOPMAN_RANK=4 KOOPMAN_DIAG_INIT=0.9 \
KOOPMAN_CONSISTENCY_WEIGHT=0.005 \
ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 \
ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=2 ENGRAM_INJECT_LAYER=1 \
VRL_ENABLED=1 VRL_START_LAYER=2 \
XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
SHARED_BLOCKS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo)"
echo ""

# --------------------------------------------------------------------------
# Run 2: ISO-PARAMETER — All Innovations, Matched Param Count
# Shrink engram + embed_dim to match baseline ~4.01M params.
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "RUN 2: ISO-PARAMETER — All Innovations, Matched Params (~4.01M)"
echo "=========================================================================="

NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=96 \
FEEDBACK_ENABLED=1 FEEDBACK_DIM=64 FEEDBACK_SKETCH_TOKENS=4 FEEDBACK_PASSES=1 \
EVAL_FEEDBACK_PASSES=2 FEEDBACK_EVERY=2 \
CAPSULE_ENABLED=1 CAPSULE_NUM=8 CAPSULE_DIM=64 \
KOOPMAN_ENABLED=1 KOOPMAN_RANK=2 KOOPMAN_DIAG_INIT=0.9 \
KOOPMAN_CONSISTENCY_WEIGHT=0.005 \
ADAPTIVE_HALT_ENABLED=1 CAPSULE_CARRY_ENABLED=1 \
BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=2048 BIGRAM_HASH_DIM=64 \
ENGRAM_NUM_HEADS=2 ENGRAM_NUM_ORDERS=2 ENGRAM_INJECT_LAYER=1 \
VRL_ENABLED=1 VRL_START_LAYER=2 \
XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
SHARED_BLOCKS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo)"
echo ""

# --------------------------------------------------------------------------
# Run 3: ISO-FLOP — Zero-Cost Innovations Only
# --------------------------------------------------------------------------
# FIRST-PRINCIPLES REDESIGN:
#
# The key insight: VRL, Partial RoPE, XSA, and LN Scale Damping add
# ZERO parameters and ZERO compute overhead. They modify existing
# operations in-place. If these alone beat the plain baseline, we have
# proven that our architectural understanding improves learning quality
# without any cost — the purest form of "innovation."
#
# This config is IDENTICAL to Baseline A except:
#   + VRL_ENABLED=1 (pass V₀ from layer 0 to deep layers)
#   + PARTIAL_ROPE_DIMS=8 (only rotate 8 dims; rest attend by content)
#   + XSA_START_LAYER=2 (subtract self-value in upper layers)
#   + LN_SCALE_DAMPING=1 (1/sqrt(layer+1) scaling on pre-norm)
#
# If this wins: zero-cost tricks are proven valuable.
# If this ties/loses: the tricks need the bigger innovations to shine.
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "RUN 3: ISO-FLOP — Zero-Cost Innovations Only (Matched Params + Step Time)"
echo "=========================================================================="

NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 EMBED_DIM=128 \
FEEDBACK_ENABLED=0 \
CAPSULE_ENABLED=0 KOOPMAN_ENABLED=0 \
BIGRAM_HASH_ENABLED=0 \
VRL_ENABLED=1 VRL_START_LAYER=2 \
XSA_START_LAYER=2 LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=8 \
SHARED_BLOCKS=0 \
ITERATIONS=200 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=2 SEED=42 \
TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 EMA_ENABLED=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 TURBO_QUANT_KV=0 \
bash run_mlx_reasoner.sh 2>&1 | grep -E "^(step|val_|final_|model_|feedback|koopman|optimizer|eval|turbo)"
echo ""

echo "=========================================================================="
echo "BENCHMARK COMPLETE at: $(date)"
echo "=========================================================================="
echo ""
echo "Reference: Baseline A (from ablation_results.txt, Seed 42):"
echo "  model_params:3085840  step ~667ms  final_sliding BPB=2.5590"
echo ""
echo "Reference: Baseline B — feedback+engram+tricks (Seed 42):"
echo "  model_params:3422736  step ~844ms  final_sliding BPB=2.4084"
echo ""
echo "Compare: params | step time (ms) | final_sliding val_bpb"
