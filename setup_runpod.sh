#!/bin/bash
# Setup script for Runpod 8xH100 environment
# Run this ONCE when you first connect to the pod
set -e

echo "=== Parameter Golf Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"

# Install dependencies
pip install -q sentencepiece numpy torch --upgrade 2>/dev/null || true

# Install FlashAttention 3 (Hopper only)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "FA3 install failed, will use SDPA fallback"

# Download full training data (80 shards)
echo "=== Downloading training data (80 shards) ==="
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

echo "=== Setup complete ==="
echo "Data shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
echo "Val shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l)"
echo ""
echo "To run baseline:   bash run_baseline.sh"
echo "To run enhanced:   bash run_enhanced.sh"
echo "To run ablation:   bash run_ablation.sh <config>"
echo ""
echo "Available ablation configs: baseline, depth_only, recovery_only, combined, recovery_30ep"
