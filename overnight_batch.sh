#!/usr/bin/env bash
set -euo pipefail

# Overnight batch: 6 experiments, ~80 min each, ~8 hours total
# Each waits for the previous to finish before starting

run_experiment() {
    local RUN_ID=$1
    shift
    echo ""
    echo "============================================"
    echo "Starting: $RUN_ID at $(date)"
    echo "============================================"
    env RUN_ID="$RUN_ID" \
        ITERATIONS=20000 \
        MAX_WALLCLOCK_SECONDS=4800 \
        VAL_LOSS_EVERY=200 \
        TRAIN_LOG_EVERY=50 \
        "$@" \
        bash start_baseline.sh
    echo "Finished: $RUN_ID at $(date)"
    echo ""
}

# ── Run 1: QAT + int6 + MLP 4× (THE KEY TEST)
# Tests if QAT recovers the int6 degradation with our best architecture
run_experiment "overnight_qat_int6_mlp4x" \
    env USE_INT6=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    NUM_LAYERS=7 MLP_MULT=4 \
    MATRIX_LR=0.08261619767374824 \
    SCALAR_LR=0.014691154447587356 \
    TIED_EMBED_LR=0.021552090970329115 \
    MUON_MOMENTUM=0.9382982028913158 \
    WARMDOWN_ITERS=1558

# ── Run 2: QAT + int6 + MLP 4× + 8 LAYERS
# If int6 compresses well, 8 layers might fit under 16MB and score better
run_experiment "overnight_qat_int6_8L_mlp4x" \
    env USE_INT6=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    NUM_LAYERS=8 MLP_MULT=4 \
    MATRIX_LR=0.08261619767374824 \
    SCALAR_LR=0.014691154447587356 \
    TIED_EMBED_LR=0.021552090970329115 \
    MUON_MOMENTUM=0.9382982028913158 \
    WARMDOWN_ITERS=1558

# ── Run 3: QAT + int6 + MLP 5× + 7 LAYERS
# Push MLP even wider — does 5× beat 4×?
run_experiment "overnight_qat_int6_mlp5x" \
    env USE_INT6=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    NUM_LAYERS=7 MLP_MULT=5 \
    MATRIX_LR=0.08261619767374824 \
    SCALAR_LR=0.014691154447587356 \
    TIED_EMBED_LR=0.021552090970329115 \
    MUON_MOMENTUM=0.9382982028913158 \
    WARMDOWN_ITERS=1558

# ── Run 4: QAT + int6 + MLP 4× + HIGHER LR
# Optuna found high MATRIX_LR works well. Push it further.
run_experiment "overnight_qat_int6_mlp4x_highLR" \
    env USE_INT6=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    NUM_LAYERS=7 MLP_MULT=4 \
    MATRIX_LR=0.12 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    MUON_MOMENTUM=0.9382982028913158 \
    WARMDOWN_ITERS=1558

# ── Run 5: QAT + int6 + MLP 4× + HIGH WEIGHT DECAY
# PR #1218 used ADAM_WD=0.085. Test higher WD to prevent overfitting.
run_experiment "overnight_qat_int6_mlp4x_highWD" \
    env USE_INT6=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    NUM_LAYERS=7 MLP_MULT=4 \
    MATRIX_LR=0.08261619767374824 \
    SCALAR_LR=0.014691154447587356 \
    TIED_EMBED_LR=0.021552090970329115 \
    MUON_MOMENTUM=0.9382982028913158 \
    MUON_WD=0.085 ADAM_WD=0.085 \
    WARMDOWN_ITERS=1558

# ── Run 6: QAT + int6 + MLP 4× + 8L + HIGH WD (YOLO combo)
# Combine the most promising settings from above
run_experiment "overnight_qat_int6_8L_mlp4x_highWD" \
    env USE_INT6=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    NUM_LAYERS=8 MLP_MULT=4 \
    MATRIX_LR=0.10 \
    SCALAR_LR=0.018 \
    TIED_EMBED_LR=0.025 \
    MUON_MOMENTUM=0.94 \
    MUON_WD=0.085 ADAM_WD=0.085 \
    WARMDOWN_ITERS=1558

echo ""
echo "============================================"
echo "ALL 6 OVERNIGHT RUNS COMPLETE at $(date)"
echo "============================================"
