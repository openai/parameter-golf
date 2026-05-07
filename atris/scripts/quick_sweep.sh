#!/bin/bash
# Quick hyperparameter sweep — run on 1xH100 to save money
# Each experiment uses 2 min instead of 10 to iterate faster
#
# Usage: bash quick_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

NPROC=1
WALLCLOCK=120  # 2 minutes for quick tests
COMMON="NCCL_IB_DISABLE=1 MAX_WALLCLOCK_SECONDS=$WALLCLOCK VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50"

run_exp() {
    local name=$1
    shift
    echo ""
    echo "================================================================"
    echo "EXPERIMENT: $name"
    echo "================================================================"

    env $COMMON "$@" \
        RUN_ID="sweep_${name}_$(date +%s)" \
        torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | \
        tee "atris/logs/${name}.log" | \
        grep -E "(val_bpb|val_loss|final_int8|model_params|submission size|stopping)"
}

echo "=== Quick Sweep (1xH100, 2min each) ==="
echo "=== Testing hyperparameter sensitivity ==="
echo ""

# Baseline
run_exp "baseline" \
    VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4

# Learning rate sweep
run_exp "lr_high" MATRIX_LR=0.06 SCALAR_LR=0.06
run_exp "lr_low" MATRIX_LR=0.02 SCALAR_LR=0.02
run_exp "lr_very_high" MATRIX_LR=0.10 SCALAR_LR=0.10

# Batch size
run_exp "batch_2x" TRAIN_BATCH_TOKENS=1048576
run_exp "batch_half" TRAIN_BATCH_TOKENS=262144

# Sequence length
run_exp "seq_512" TRAIN_SEQ_LEN=512
run_exp "seq_2048" TRAIN_SEQ_LEN=2048

# Model shape — CRITICAL: test wider vs deeper
run_exp "wider_768" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4
run_exp "wider_640" MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4
run_exp "deeper_12" NUM_LAYERS=12 MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4
run_exp "deeper_15" NUM_LAYERS=15 MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4

# Vocab size
run_exp "vocab_2048" VOCAB_SIZE=2048
run_exp "vocab_4096" VOCAB_SIZE=4096

# Muon optimizer
run_exp "muon_mom_98" MUON_MOMENTUM=0.98
run_exp "muon_mom_90" MUON_MOMENTUM=0.90
run_exp "muon_steps_3" MUON_BACKEND_STEPS=3
run_exp "muon_steps_7" MUON_BACKEND_STEPS=7

# Warmdown
run_exp "warmdown_2400" WARMDOWN_ITERS=2400
run_exp "warmdown_600" WARMDOWN_ITERS=600

# MLP multiplier
run_exp "mlp_3x" MLP_MULT=3
run_exp "mlp_4x" MLP_MULT=4

echo ""
echo "================================================================"
echo "SWEEP COMPLETE"
echo "Check atris/logs/ for results"
echo "================================================================"
