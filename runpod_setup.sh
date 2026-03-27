#!/bin/bash
# RunPod 8xH100 Setup Script
# 1. Go to: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
# 2. Select 8x H100 SXM pod
# 3. SSH in and run this script

set -e

cd /workspace
git clone https://github.com/aptsalt/parameter-golf.git
cd parameter-golf

# Install dependencies
pip install zstandard sentencepiece

# Download dataset (full)
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "Setup complete. Run training with:"
echo "  bash run_submission_b.sh"
echo "  # or"
echo "  bash run_submission_a.sh"
