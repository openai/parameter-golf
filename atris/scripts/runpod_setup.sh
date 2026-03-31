#!/bin/bash
# RunPod 8xH100 setup script
# Run this ONCE after SSH-ing into your RunPod pod
#
# Usage: bash runpod_setup.sh

set -euo pipefail

echo "=== Parameter Golf: RunPod Setup ==="

cd /workspace

# Clone our fork (update URL after forking)
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
    cd parameter-golf
else
    cd parameter-golf
    git pull
fi

# Download dataset (full — all 80 shards)
echo "Downloading full dataset..."
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify GPU setup
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""
echo "GPU count: $(nvidia-smi -L | wc -l)"

# Run baseline reproduction
echo ""
echo "=== Running Baseline Reproduction ==="
echo "This takes ~10 minutes..."

NCCL_IB_DISABLE=1 \
RUN_ID=baseline_repro \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Setup Complete ==="
echo "Check the val_bpb output above. Should be ~1.2244"
echo ""
echo "Next: Copy atris/ scripts here and start the autoresearch loop"
echo "  python atris/scripts/autoresearch.py --mode run --experiment 'your_experiment'"
echo "  python atris/scripts/autoresearch.py --mode status"
