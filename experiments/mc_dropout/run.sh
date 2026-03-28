#!/bin/bash
# MC Dropout experiment: train with dropout, eval with K-pass averaging
#
# Usage: bash run.sh [K]
#   K = number of MC forward passes (default: 16)
set -euo pipefail
cd "$(dirname "$0")"

K=${1:-16}
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
REPO=/root/parameter-golf

export DATA_PATH="$REPO/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$REPO/data/tokenizers/fineweb_1024_bpe.model"
export DROPOUT_RATE=${DROPOUT_RATE:-0.3}

echo "=== MC Dropout Experiment ==="
echo "DROPOUT_RATE=$DROPOUT_RATE"
echo "MC passes K=$K"
echo "GPUs: $NGPU"
echo ""

# --- Step 1: Train with dropout ---
echo "=== Training ==="
if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" train.py
else
    python3 train.py
fi

# --- Step 2: MC Dropout eval ---
echo ""
echo "=== MC Dropout Eval (K=$K) ==="
if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" eval.py --checkpoint final_model.pt --K "$K"
else
    python3 eval.py --checkpoint final_model.pt --K "$K"
fi
