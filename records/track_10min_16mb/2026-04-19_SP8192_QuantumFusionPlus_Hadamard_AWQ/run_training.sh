#!/bin/bash

# SP8192 + Quantum Fusion Plus - 3-Seed Training Script
# Run on 8xH100 GPU cluster with torchrun

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
DATA_PATH="${DATA_PATH:-./ data/datasets/fineweb10B_sp8192}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./ data/tokenizers/fineweb_8192_bpe.model}"
MAX_WALLCLOCK_SECONDS=600

# Seeds for 3-seed evaluation
SEEDS=(42 314 999)

echo "=========================================="
echo "SP8192 + Quantum Fusion Plus Training"
echo "=========================================="
echo "Data path: $DATA_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Max wallclock: $MAX_WALLCLOCK_SECONDS seconds"
echo ""

# Create output directory
mkdir -p results

# Run training for each seed
for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with SEED=$SEED"
    echo "=========================================="
    
    export SEED=$SEED
    export DATA_PATH=$DATA_PATH
    export TOKENIZER_PATH=$TOKENIZER_PATH
    export MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS
    
    # Enable quantization modules
    export HADAMARD_ROTATION_ENABLED=1
    export AWQ_ENABLED=1
    export HESSIAN_AWARE_CALIBRATION_ENABLED=1
    export LAYER_WISE_PRECISION_ENABLED=1
    export TTT_ENABLED=1
    
    # Run training with torchrun for distributed training
    torchrun --nproc_per_node=8 train_gpt_sp8192_fusion.py \
        --seed $SEED \
        --data_path $DATA_PATH \
        --tokenizer_path $TOKENIZER_PATH \
        --max_wallclock_seconds $MAX_WALLCLOCK_SECONDS
    
    # Save results
    cp train_seed${SEED}.log results/ 2>/dev/null || true
    
    echo "Completed training for SEED=$SEED"
    echo ""
done

echo "=========================================="
echo "All training runs completed!"
echo "Results saved in: results/"
echo "=========================================="
