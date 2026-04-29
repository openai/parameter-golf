#!/bin/bash
# train_pr1493.sh — Full train+eval reproduction of PR #1493
# Usage: bash train_pr1493.sh [SEED]
# Expects: /workspace/data/ has SP8192 data, train_gpt.py in current dir
set -euo pipefail

SEED="${1:-42}"
echo "=== PR #1493 Full Reproduction — Seed $SEED ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Download SP8192 data if not present
DATA_DIR="/workspace/data"
if [[ ! -d "$DATA_DIR/datasets/fineweb10B_sp8192" ]]; then
    echo "Downloading SP8192 data..."
    cd /workspace
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
fi

echo "Data ready. Running train + eval..."

export SEED=$SEED
export DATA_DIR="$DATA_DIR"
export TTT_ENABLED=1
export SLIDING_WINDOW_ENABLED=1

# Full training run: torchrun 8 GPUs
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/train_seed${SEED}.log"

echo "=== Run complete ==="
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: /workspace/train_seed${SEED}.log"
