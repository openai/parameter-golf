#!/bin/bash
# Competitive Baseline Training Pipeline
# Run this on the CUDA machine with: bash run_training_pipeline.sh

set -e  # Exit on error

PROJECT_DIR="/workspace/parameter-golf"
BASELINE_DIR="records/track_10min_16mb/2026-03-22_competitive_baseline"

echo "======================================"
echo "Parameter Golf - Competitive Baseline"
echo "======================================"
echo ""

# Check prerequisites
echo "Step 1: Checking prerequisites..."
if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found"
    exit 1
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not installed"
    exit 1
fi

echo "✓ Prerequisites OK"
echo ""

# Install zstandard if needed
echo "Step 2: Ensuring zstandard is installed..."
pip install -q zstandard
echo "✓ zstandard installed"
echo ""

# Check if dataset exists
echo "Step 3: Checking dataset..."
if [ ! -d "./data/datasets/fineweb10B_sp1024/fineweb_train_0.bin" ] && [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_0.bin" ]; then
    echo "Dataset not found. Preparing..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
    echo "✓ Dataset prepared"
else
    echo "✓ Dataset already present"
fi
echo ""

# Run smoke test
echo "Step 4: Running SMOKE TEST (200 iterations, 1 GPU)..."
echo "=========================================="
RUN_ID=smoke_test \
ITERATIONS=200 \
VAL_LOSS_EVERY=100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 \
  "$BASELINE_DIR/train_gpt.py"

echo "=========================================="
echo "✓ Smoke test complete"
echo ""

# Read user confirmation
read -p "Smoke test successful? Press Enter to continue with full training runs, or Ctrl+C to stop..."
echo ""

# Full training runs
for SEED in 1337 42 7; do
    RUN_NUM=$((SEED == 1337 ? 1 : SEED == 42 ? 2 : 3))
    echo "Step $((4 + RUN_NUM)): Running FULL TRAINING #$RUN_NUM (8 GPUs, SEED=$SEED)..."
    echo "=========================================="
    
    RUN_ID="competitive_run_$RUN_NUM" SEED=$SEED \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 SEQ_LEN=4096 TRAIN_BATCH_TOKENS=131072 \
    NUM_LAYERS=10 MODEL_DIM=512 MLP_MULT=3 WEIGHT_DECAY=0.04 SWA_RATIO=0.4 \
    torchrun --standalone --nproc_per_node=8 \
      "$BASELINE_DIR/train_gpt.py" \
      2>&1 | tee "$BASELINE_DIR/train_$RUN_NUM.log"
    
    echo "=========================================="
    echo "✓ Training run #$RUN_NUM complete"
    echo ""
done

echo "======================================"
echo "All training runs complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Extract metrics: python3 extract_metrics.py ..."
echo "2. Fill in submission.json"
echo "3. Fill in README.md"
echo "4. Submit!"
