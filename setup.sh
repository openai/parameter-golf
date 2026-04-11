#!/bin/bash
# Setup script for parameter-golf-entry
# Works on both local 3090 and RunPod H100

set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch — use the version matching your CUDA
# The official repo pins torch==2.10 for H100/RunPod.
# For local 3090 (CUDA 12.x), this should also work.
pip install torch==2.10

# Other dependencies
pip install numpy tqdm huggingface-hub datasets sentencepiece tiktoken kernels setuptools "typing-extensions==4.15.0"

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo "Then download data:  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1"
