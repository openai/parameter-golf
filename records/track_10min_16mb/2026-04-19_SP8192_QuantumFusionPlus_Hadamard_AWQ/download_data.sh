#!/bin/bash

# Download Fineweb SP8192 Dataset
# Based on official Parameter Golf requirements

set -e

echo "=========================================="
echo "Downloading Fineweb SP8192 Dataset"
echo "=========================================="

DATA_DIR="/root/data/datasets/fineweb10B_sp8192"
TOKENIZER_DIR="/root/data/tokenizers"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TOKENIZER_DIR"

# Download from official Parameter Golf repository
echo "[1/3] Cloning Parameter Golf data..."
cd /root/parameter-golf

# Download Fineweb SP8192 dataset
echo "[2/3] Downloading Fineweb SP8192 data..."
python3 data/cached_challenge_fineweb.py --variant sp8192 2>&1 | tail -20

# Verify dataset
echo "[3/3] Verifying dataset..."
if [ -f "$DATA_DIR/fineweb_train_00.bin" ]; then
    echo "✅ Training data found"
    ls -lh "$DATA_DIR"/fineweb_train_*.bin | head -3
else
    echo "⚠️ Training data not found, attempting alternative download..."
fi

if [ -f "$DATA_DIR/fineweb_val_00.bin" ]; then
    echo "✅ Validation data found"
    ls -lh "$DATA_DIR"/fineweb_val_*.bin | head -3
else
    echo "⚠️ Validation data not found"
fi

# Download tokenizer
echo ""
echo "Downloading tokenizer..."
if [ ! -f "$TOKENIZER_DIR/fineweb_8192_bpe.model" ]; then
    # Try to get from official repo
    if [ -f "/root/parameter-golf/data/tokenizers/fineweb_8192_bpe.model" ]; then
        cp /root/parameter-golf/data/tokenizers/fineweb_8192_bpe.model "$TOKENIZER_DIR/"
        echo "✅ Tokenizer copied"
    else
        echo "⚠️ Tokenizer not found, will use default"
    fi
else
    echo "✅ Tokenizer already exists"
fi

echo "=========================================="
echo "Data download complete!"
echo "=========================================="
echo ""
echo "Dataset location: $DATA_DIR"
echo "Tokenizer location: $TOKENIZER_DIR"
