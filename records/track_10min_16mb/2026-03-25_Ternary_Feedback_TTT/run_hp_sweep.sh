#!/bin/bash
# ============================================================================
# HYPERPARAMETER SWEEP: First-Principles Tuning for 1hr TKA-H
# Each experiment runs for 15 minutes (enough to see curriculum effects).
# ============================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

SWEEP_LOG="hp_sweep_results.log"
echo "--- HP SWEEP START: $(date) ---" > "$SWEEP_LOG"

# Base config (same as E_Shatter_Expectations_Final)
BASE_ARGS=(
    ARCHITECTURE=hybrid NUM_LAYERS=8 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4
    EMBED_DIM=128 SHARED_BLOCKS=2 FEEDBACK_ENABLED=0 EVAL_FEEDBACK_PASSES=2
    CAPSULE_ENABLED=1 CAPSULE_NUM=16 CAPSULE_DIM=64
    KOOPMAN_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_STATE_DIM=64 KOOPMAN_MIXER_RANK=4
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=64 ENGRAM_NUM_ORDERS=3
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=feedback
    TURBO_QUANT_TRAIN=0 TURBO_QUANT_EXPORT=1 TURBO_QUANT_KV=1
    MOE_ENABLED=1 MOE_NUM_EXPERTS=3 MOE_TOP_K=1 MOE_ROUTER_AUX_LOSS_COEF=0.01
    STOCHASTIC_DEPTH_PROB=0.1
    EMA_ENABLED=1 EMA_EVAL_APPLY=1 EMA_DECAY=0.997
    SELF_DISTILL_KL_WEIGHT=0.1
    MUON_MOMENTUM_WARMUP_STEPS=500
)

# Common sweep settings: 15 min, eval every 50 steps for fine-grained curves
SWEEP_ARGS=(
    SEED=42 ITERATIONS=6000 VAL_LOSS_EVERY=50
    MAX_WALLCLOCK_SECONDS=900 TRAIN_BATCH_TOKENS=16384
    GRAD_ACCUM_STEPS=1 SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1
)

run_experiment() {
    local name=$1
    shift 1
    echo "==========================================================================" | tee -a "$SWEEP_LOG"
    echo "EXPERIMENT: $name | START: $(date)" | tee -a "$SWEEP_LOG"
    echo "==========================================================================" | tee -a "$SWEEP_LOG"

    RUN_ID="sweep_${name}" \
    env "${BASE_ARGS[@]}" "${SWEEP_ARGS[@]}" "$@" \
    bash run_mlx_reasoner.sh >> "$SWEEP_LOG" 2>&1

    echo "--- $name DONE: $(date) ---" | tee -a "$SWEEP_LOG"
    echo "" >> "$SWEEP_LOG"
}

# ── A: BASELINE (current config) ──────────────────────────────────────────────
# Current 10-min config extrapolated to 15 min. Control group.
run_experiment "A_baseline" \
    CURRICULUM_PHASE1_FRAC=0.35 CURRICULUM_PHASE2_FRAC=0.65 \
    WARMDOWN_FRACTION=0.5 \
    MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.06 \
    TERNARY_NOISE_SCALE=0.05 \
    EMA_START_FRACTION=0.3 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=256 CURRICULUM_PHASE2_SEQ=512

# ── B: FAST CURRICULUM ────────────────────────────────────────────────────────
# Hypothesis: seq=256 converges by ~8 min. Push more time to seq=1024.
# Phase1=15% (2.25min seq=256), Phase2=40% (3.75min seq=512), Phase3=45% (6.75min seq=1024)
run_experiment "B_fast_curriculum" \
    CURRICULUM_PHASE1_FRAC=0.15 CURRICULUM_PHASE2_FRAC=0.40 \
    WARMDOWN_FRACTION=0.5 \
    MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.06 \
    TERNARY_NOISE_SCALE=0.05 \
    EMA_START_FRACTION=0.3 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=256 CURRICULUM_PHASE2_SEQ=512

# ── C: FAST CURRICULUM + SHORTER WARMDOWN ──────────────────────────────────────
# Hypothesis: With fast curriculum, we want more full-LR steps at seq=1024.
# Warmdown=0.35 (starts at 65% wallclock = 9.75min). More time at peak LR.
run_experiment "C_fast_curr_short_wd" \
    CURRICULUM_PHASE1_FRAC=0.15 CURRICULUM_PHASE2_FRAC=0.40 \
    WARMDOWN_FRACTION=0.35 \
    MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.06 \
    TERNARY_NOISE_SCALE=0.05 \
    EMA_START_FRACTION=0.3 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=256 CURRICULUM_PHASE2_SEQ=512

# ── D: FAST CURRICULUM + SHORT WD + LOWER LR + LESS NOISE ─────────────────────
# Hypothesis: For longer effective training (more full-LR steps), lower peak LR
# prevents overshooting. Lower noise prevents interference during warmdown.
run_experiment "D_full_tune" \
    CURRICULUM_PHASE1_FRAC=0.15 CURRICULUM_PHASE2_FRAC=0.40 \
    WARMDOWN_FRACTION=0.35 \
    MATRIX_LR=0.03 SCALAR_LR=0.03 TIED_EMBED_LR=0.045 \
    TERNARY_NOISE_SCALE=0.02 \
    EMA_START_FRACTION=0.40 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=256 CURRICULUM_PHASE2_SEQ=512

echo "==========================================================================" | tee -a "$SWEEP_LOG"
echo "SWEEP COMPLETE: $(date)" | tee -a "$SWEEP_LOG"
echo "==========================================================================" | tee -a "$SWEEP_LOG"

# Extract summary
echo "" >> "$SWEEP_LOG"
echo "=== SUMMARY ===" >> "$SWEEP_LOG"
grep -E "EXPERIMENT:|final_eval|final_sliding|ngram_cache|artifact" "$SWEEP_LOG" >> "$SWEEP_LOG"
