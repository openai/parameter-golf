#!/bin/bash
# RunPod setup script for SubSixteen v2
# Run this after SSH'ing into your RunPod pod
# Usage: bash runpod_setup.sh

set -e

echo "=== SubSixteen RunPod Setup ==="

# Clone your fork (has the v2 train_gpt.py)
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/TevBenji/parameter-golf.git
    cd parameter-golf
else
    cd parameter-golf
    git pull origin main
fi

# Install zstandard (not in the base template)
pip install zstandard

# Download full dataset (80 shards, ~8B tokens)
echo "=== Downloading dataset ==="
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "=== Setup complete ==="
echo ""
echo "To run training:"
echo "  bash run_train.sh"
