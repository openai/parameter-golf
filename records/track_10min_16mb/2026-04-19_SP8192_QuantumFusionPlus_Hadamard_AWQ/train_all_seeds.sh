#!/bin/bash

# 3-Seed Training Script for SP8192 + Quantum Fusion Plus
# Runs on 8xH100 with torchrun

set -e

echo "=========================================="
echo "SP8192 + Quantum Fusion Plus Training"
echo "=========================================="

# Setup paths
SUBMISSION_DIR="/root/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ"
DATA_PATH="/root/data/datasets/fineweb10B_sp8192"
TOKENIZER_PATH="/root/data/tokenizers/fineweb_8192_bpe.model"
RESULTS_DIR="/root/results"
LOGS_DIR="/root/logs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Seeds for 3-seed evaluation
SEEDS=(42 314 999)

# Training configuration
export DATA_PATH=$DATA_PATH
export TOKENIZER_PATH=$TOKENIZER_PATH
export MAX_WALLCLOCK_SECONDS=600

# Enable all quantization modules
export HADAMARD_ROTATION_ENABLED=1
export AWQ_ENABLED=1
export HESSIAN_AWARE_CALIBRATION_ENABLED=1
export LAYER_WISE_PRECISION_ENABLED=1
export TTT_ENABLED=1

# Model configuration
export NUM_LAYERS=11
export MODEL_DIM=512
export VOCAB_SIZE=8192
export NUM_HEADS=8
export NUM_KV_HEADS=4
export ITERATIONS=4550
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export QK_GAIN_INIT=5.25

echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Model: ${NUM_LAYERS}L x ${MODEL_DIM}d"
echo "  Vocab: $VOCAB_SIZE"
echo "  Seeds: ${SEEDS[@]}"
echo ""

# Run training for each seed
for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with SEED=$SEED"
    echo "=========================================="
    
    export SEED=$SEED
    
    # Create log file
    LOG_FILE="$LOGS_DIR/train_seed${SEED}.log"
    
    # Run training with torchrun
    echo "Starting training... (log: $LOG_FILE)"
    
    cd "$SUBMISSION_DIR"
    
    # Run with torchrun for distributed training
    torchrun --nproc_per_node=8 train_gpt_sp8192_fusion.py 2>&1 | tee "$LOG_FILE"
    
    # Copy log to results
    cp "$LOG_FILE" "$RESULTS_DIR/train_seed${SEED}.log"
    
    # Extract metrics
    echo ""
    echo "Extracting metrics for SEED=$SEED..."
    if grep -q "val_bpb" "$LOG_FILE"; then
        METRICS=$(grep "val_bpb" "$LOG_FILE" | tail -1)
        echo "✅ Metrics: $METRICS"
    else
        echo "⚠️ No metrics found in log"
    fi
    
    echo "Completed training for SEED=$SEED"
    echo ""
done

echo "=========================================="
echo "All training runs completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="

# Summary
echo ""
echo "Training Summary:"
echo "  Seed 42: $RESULTS_DIR/train_seed42.log"
echo "  Seed 314: $RESULTS_DIR/train_seed314.log"
echo "  Seed 999: $RESULTS_DIR/train_seed999.log"
