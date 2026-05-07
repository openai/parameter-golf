#!/bin/bash
# Hyperparameter sweep for improved Parameter Golf submission.
# Run on 8xH100: bash sweep_hp.sh
# Or on 1xH100 for faster iteration: NPROC=1 bash sweep_hp.sh
set -euo pipefail

NPROC="${NPROC:-8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"

# Base config (SOTA defaults with our improvements)
export DATA_DIR="./data/"
export VOCAB_SIZE=8192
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export WARMDOWN_FRAC=0.72
export WARMUP_STEPS=20
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export PARALLEL_RESIDUAL_START=7
export SKIP_GATES_ENABLED=1
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1
export COMPRESSOR=brotli
export SLIDING_WINDOW_ENABLED=1
export MIXED_PRECISION_QUANT=1
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9

RESULTS_DIR="${SCRIPT_DIR}/sweep_results"
mkdir -p "$RESULTS_DIR"

run_experiment() {
    local name="$1"
    shift
    echo "=== Running experiment: $name ==="
    local logfile="${RESULTS_DIR}/${name}.log"
    env "$@" \
        RUN_ID="$name" \
        SEED=42 \
        VAL_LOSS_EVERY=1000 \
        torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
        2>&1 | tee "$logfile"
    echo "=== Finished: $name ==="
}

# ---- Sweep 1: QK-Gain ----
for qk in 5.0 5.25 5.5 5.75 6.0; do
    run_experiment "qk_gain_${qk}" QK_GAIN_INIT="$qk"
done

# ---- Sweep 2: Weight Decay ----
for wd in 0.085 0.095 0.100 0.110 0.120; do
    run_experiment "muon_wd_${wd}" MUON_WD="$wd"
done

# ---- Sweep 3: EMA Decay ----
for ema in 0.996 0.9965 0.997 0.9975 0.998; do
    run_experiment "ema_${ema}" EMA_DECAY="$ema"
done

# ---- Sweep 4: Matrix LR ----
for mlr in 0.018 0.020 0.022 0.024 0.026; do
    run_experiment "matrix_lr_${mlr}" MATRIX_LR="$mlr"
done

# ---- Sweep 5: Mixed Precision Quant fractions ----
for hfrac in 0.10 0.15 0.20 0.25; do
    for lfrac in 0.15 0.25 0.35; do
        run_experiment "mixq_h${hfrac}_l${lfrac}" \
            QUANT_HIGH_FRAC="$hfrac" QUANT_LOW_FRAC="$lfrac"
    done
done

# ---- Sweep 6: Compressor comparison ----
for comp in brotli lzma zstd; do
    run_experiment "compressor_${comp}" COMPRESSOR="$comp"
done

echo "=== All sweeps complete. Results in ${RESULTS_DIR}/ ==="
