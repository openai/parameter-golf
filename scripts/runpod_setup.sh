#!/bin/bash
# =============================================================================
# RunPod Setup & Run Script for BESE v2 Fair Comparison
# =============================================================================
#
# This script does everything on the RunPod pod:
# 1. Clones repos
# 2. Downloads FineWeb data
# 3. Copies BESE code into position
# 4. Runs the fair comparison (baseline SP1024 vs BESE+BPE)
#
# Usage:
#   On your Mac (from ~/Projects/parameter-golf-bese/):
#     1. Push latest code to GitHub: git add -A && git commit -m "v2 updates" && git push
#     2. Start a RunPod pod (1xH100 SXM, parameter-golf template, ~$2.69/hr)
#     3. SSH into the pod
#     4. Run: curl -sSL https://raw.githubusercontent.com/mrbese/parameter-golf-bese/main/scripts/runpod_setup.sh | bash
#
#   OR copy-paste the commands below manually.
# =============================================================================

set -euo pipefail

echo "============================================="
echo "  BESE v2 RunPod Setup"
echo "============================================="

cd /workspace

# Step 1: Clone upstream parameter-golf (if not already present from template)
if [ ! -d "parameter-golf" ]; then
    echo ">> Cloning upstream parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git
fi

# Step 2: Download FineWeb data (all 10 shards + val)
echo ">> Downloading FineWeb data (this takes a few minutes)..."
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
cd /workspace

# Step 3: Clone BESE repo
if [ ! -d "bese" ]; then
    echo ">> Cloning BESE repo..."
    git clone https://github.com/mrbese/parameter-golf-bese.git bese
else
    echo ">> BESE repo already present, pulling latest..."
    cd bese && git pull && cd /workspace
fi

# Step 4: Install any missing deps
pip install sentencepiece 2>/dev/null || true

# Step 5: Run the fair comparison
echo ""
echo "============================================="
echo "  Starting BESE v2 Fair Comparison"
echo "  Baseline (SP1024 9L/512d/2x) vs BESE (11L/512d/3x)"
echo "============================================="
echo ""

cd /workspace
python3 bese/scripts/runpod_v2.py \
    --num-merges 250 \
    --num-layers 11 \
    --model-dim 512 \
    --mlp-mult 3

echo ""
echo "============================================="
echo "  Done! Check results above."
echo "============================================="
