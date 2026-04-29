#!/bin/bash
# eval_pr1493.sh — Run eval-only on PR #1493 published artifact
# Usage: bash eval_pr1493.sh [SEED]
# Expects: /workspace/data/ has SP8192 data, train_gpt.py in current dir
set -euo pipefail

SEED="${1:-42}"
echo "=== PR #1493 Eval-Only — Seed $SEED ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Download SP8192 data if not present
DATA_DIR="/workspace/data"
if [[ ! -d "$DATA_DIR/datasets/fineweb10B_sp8192" ]]; then
    echo "Downloading SP8192 data..."
    cd /workspace
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
fi

# Check data exists
ls "$DATA_DIR/datasets/fineweb10B_sp8192/" | head -5
ls "$DATA_DIR/tokenizers/fineweb_8192_bpe.model"

echo "Data ready. Running eval..."

# Run eval-only: set ITERATIONS=0 to skip training, just deserialize + eval
# The train_gpt.py should handle this if we pass the right env vars
export SEED=$SEED
export DATA_DIR="$DATA_DIR"
export ITERATIONS=0
export TTT_ENABLED=1
export SLIDING_WINDOW_ENABLED=1

# Use torchrun for 8 GPU
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "=== Eval complete ==="
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
