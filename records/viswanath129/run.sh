#!/bin/bash
# Parameter Golf Challenge - Run Script
# Requires: 8xH100 SXM GPUs, CUDA, PyTorch
set -euo pipefail

echo "============================================"
echo "   Parameter Golf Challenge - Training"
echo "============================================"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA required."
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
pip install -q sentencepiece numpy torch

# Download data if not present
DATA_DIR="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Downloading cached FineWeb data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER"
    echo "Run: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi

# Train with torchrun (8 GPUs)
echo ""
echo "Starting training on 8 GPUs..."
echo "Max wallclock: 600 seconds"
echo ""

torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "============================================"
echo "   Training Complete!"
echo "============================================"

# Check output artifact size
if [ -f "final_model.int8.ptz" ]; then
    MODEL_SIZE=$(stat -c%s "final_model.int8.ptz" 2>/dev/null || stat -f%z "final_model.int8.ptz")
    CODE_SIZE=$(wc -c < train_gpt.py)
    TOTAL=$((MODEL_SIZE + CODE_SIZE))
    echo "Model artifact: $MODEL_SIZE bytes"
    echo "Code size: $CODE_SIZE bytes"
    echo "Total: $TOTAL bytes (limit: 16,000,000)"
    if [ $TOTAL -le 16000000 ]; then
        echo "✅ Size check PASSED"
    else
        echo "❌ Size check FAILED - too large!"
    fi
fi
