#!/bin/bash
# ============================================================================
# TERNARY KOOPMAN-ATTENTION HYBRID (TKA-H) - MLX BENCHMARK
# Runs the new hybrid architecture for exactly 10 minutes.
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

LOG_FILE="hybrid_mac_benchmark.log"
echo "--- HYBRID 10-MIN RUN START: $(date) ---" > "$LOG_FILE"

# Common settings for 10-minute runs
ITERATIONS=1500
VAL_LOSS_EVERY=100
MAX_WALLCLOCK_SECONDS=600
TRAIN_BATCH_TOKENS=16384
GRAD_ACCUM_STEPS=1
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

# ── E_Shatter_Expectations v2 ────────────────────────────────────────────────
# Key design decisions vs prior failed run:
#   - FEEDBACK_ENABLED=0 during training (was 1 → caused 5.7× step-time blowup)
#     Eval still uses EVAL_FEEDBACK_PASSES=2 — full quality at scoring time.
#   - TURBO_QUANT_TRAIN=0 (Hadamard per weight per step was ~25% overhead; export covers it)
#   - MODEL_DIM=256 restored — MoE top-k=1 means same active FLOPs as dense at this width
#   - MOE_ENABLED=1 / 3 experts / top-k=1: 3× FFN parameter budget, identical FLOP count
#   - STOCHASTIC_DEPTH_PROB=0.1: shared blocks learn a depth-robust weight ensemble
#   - CURRICULUM_ENABLED=1: seq 256→512→1024 triples early step count for better exploration
#   - TERNARY_NOISE_SCALE=0.05: smooths ternary quantization cliff landscape during STE
#   - EMA_EVAL_APPLY=1: eval uses the running EMA shadow → lower ternary MSE at scoring
#   - SELF_DISTILL_KL_WEIGHT=0.1: if feedback later re-enabled, anchors pass-1 to pass-0
# ── E_Shatter_Expectations v5 (The Golden Config) ──────────────────────────
# "Ultimate implementation" calibrated for MacBook speed AND quality:
#   - ARCHITECTURE=hybrid + mx.compile = fast (estimated 480-550ms/step)
#   - MOE_ENABLED=1 / 3 experts / top-k=1: 3× FFN parameters within 16MB budget.
#   - Dims: MODEL_DIM=128, NUM_LAYERS=8, SHARED_BLOCKS=2 (4 unique blocks).
#   - Features: Curriculum (256-1024), Stochastic Depth (0.1), Noise (0.05), KL Distill (0.1).
#   - Stabilization: OvertoneInit spectral shaping (implicit in NeoMuon + 0.04 LR).
run_config "E_Shatter_Expectations_Final" \
    ARCHITECTURE=hybrid NUM_LAYERS=8 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 EVAL_FEEDBACK_PASSES=2 \
    CAPSULE_ENABLED=1 CAPSULE_NUM=16 CAPSULE_DIM=64 \
    KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_STATE_DIM=64 KOOPMAN_MIXER_RANK=4 \
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=64 ENGRAM_NUM_ORDERS=3 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback \
    TURBO_QUANT_TRAIN=0 TURBO_QUANT_EXPORT=1 TURBO_QUANT_KV=1 \
    MOE_ENABLED=1 MOE_NUM_EXPERTS=3 MOE_TOP_K=1 MOE_ROUTER_AUX_LOSS_COEF=0.01 \
    STOCHASTIC_DEPTH_PROB=0 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=256 \
    CURRICULUM_PHASE1_FRAC=0.05 CURRICULUM_PHASE2_FRAC=0.20 \
    WARMDOWN_FRACTION=0.5 \
    TERNARY_NOISE_SCALE=0.05 \
    EMA_ENABLED=1 EMA_EVAL_APPLY=1 EMA_DECAY=0.997 EMA_START_FRACTION=0.3 \
    SELF_DISTILL_KL_WEIGHT=0.1 \
    MATRIX_LR=0.04 TIED_EMBED_LR=0.06 SCALAR_LR=0.04 \
    MUON_MOMENTUM_WARMUP_STEPS=500

echo "--- RUN COMPLETE ---" | tee -a "$LOG_FILE"
