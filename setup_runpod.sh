#!/bin/bash
# Setup script for RunPod environment (using the parameter-golf template)
# The template pre-installs PyTorch + CUDA. We just need anthropic + data.
# Run from /workspace/parameter-golf/

set -e

echo "=== Parameter Golf — RunPod Setup ==="

# The template should have most deps; install anything missing
pip install -r requirements.txt
pip install anthropic

# Download the dataset (tokenized FineWeb shards)
echo "Downloading dataset..."
python3 data/cached_challenge_fineweb.py --variant sp1024

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run autoresearch:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  # Single GPU, 3-min experiments (for fast iteration):"
echo "  python3 autoresearch.py"
echo ""
echo "  # Single GPU, custom time:"
echo "  EXPERIMENT_SECONDS=120 python3 autoresearch.py"
echo ""
echo "  # 8×GPU, full 10-min runs (for final validation):"
echo "  GPUS=8 EXPERIMENT_SECONDS=600 python3 autoresearch.py"
