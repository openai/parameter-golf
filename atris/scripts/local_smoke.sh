#!/bin/bash
# Local smoke test on Mac (Apple Silicon MLX)
# Fast iteration on architectural ideas before burning GPU hours
#
# Usage:
#   ./local_smoke.sh                    # default 50 iterations
#   ITERATIONS=200 ./local_smoke.sh     # more iterations
#   RUN_ID=my_test ./local_smoke.sh     # named run

set -euo pipefail

cd "$(dirname "$0")/../.."

# Ensure venv
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
else
    source .venv/bin/activate
fi

# Ensure data
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading dataset (1 shard for local testing)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

# Run
export RUN_ID="${RUN_ID:-smoke_$(date +%s)}"
export ITERATIONS="${ITERATIONS:-50}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8192}"

echo "============================================"
echo "LOCAL SMOKE TEST: $RUN_ID"
echo "  iterations: $ITERATIONS"
echo "  batch: $TRAIN_BATCH_TOKENS tokens"
echo "============================================"

python3 train_gpt_mlx.py

echo ""
echo "Done. Check val_bpb in output above."
