#!/bin/bash

# RunPod Setup Script for SP8192 + Quantum Fusion Plus
# This script initializes the RunPod environment

set -e

echo "=========================================="
echo "RunPod Environment Setup"
echo "=========================================="

# Update system
echo "[1/5] Updating system packages..."
apt-get update -qq
apt-get install -y -qq git curl wget 2>&1 | grep -v "^Reading\|^Building\|^Selecting" || true

# Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q torch numpy sentencepiece 2>&1 | tail -5

# Verify CUDA
echo "[3/5] Verifying CUDA setup..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Create directories
echo "[4/5] Creating directories..."
mkdir -p /root/data/datasets
mkdir -p /root/data/tokenizers
mkdir -p /root/results
mkdir -p /root/logs

# Clone official repository
echo "[5/5] Cloning official repository..."
cd /root
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: bash download_data.sh"
echo "2. Run: bash train_all_seeds.sh"
echo "3. Run: bash collect_results.sh"
echo "4. Run: bash upload_to_github.sh"
