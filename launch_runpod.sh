#!/usr/bin/env bash
# =================================================================
# Parameter Golf — Track A Launch Script for RunPod 8×H100 SXM
# =================================================================
# Usage (on RunPod pod):
#   bash launch_runpod.sh          # Full 10-min training run
#   bash launch_runpod.sh smoke    # Quick 200-iter smoke test
#
# Pre-requisites: Deploy the official Parameter Golf template:
#   https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
# =================================================================

set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
FORK_URL="https://github.com/Omrigotlieb/parameter-golf.git"
DATA_VARIANT="sp1024"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"

MODE="${1:-full}"

echo "===== Parameter Golf — Track A ====="
echo "Mode: $MODE"
echo "===================================="

# ---- 1. Clone / update repo ----
if [ -d "$REPO_DIR" ]; then
    echo "[1/4] Updating repo..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard origin/main
else
    echo "[1/4] Cloning repo..."
    cd /workspace
    git clone "$FORK_URL" parameter-golf
    cd "$REPO_DIR"
fi

# ---- 2. Install extra deps (zstandard for zstd-22) ----
echo "[2/4] Installing dependencies..."
pip install -q zstandard 2>/dev/null || pip install zstandard

# ---- 3. Download data ----
DATA_PATH="./data/datasets/fineweb10B_${DATA_VARIANT}"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

if [ ! -d "$DATA_PATH" ] || [ "$(ls "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "[3/4] Downloading FineWeb data (shards=$TRAIN_SHARDS)..."
    python3 data/cached_challenge_fineweb.py --variant "$DATA_VARIANT" --train-shards "$TRAIN_SHARDS"
else
    echo "[3/4] Data already present, skipping download."
fi

# ---- 4. Launch training ----
echo "[4/4] Launching training..."

if [ "$MODE" = "smoke" ]; then
    echo "  >> Smoke test: 200 iters, val every 100"
    NCCL_IB_DISABLE=1 \
    RUN_ID="track_a_smoke_$(date +%Y%m%d_%H%M%S)" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    ITERATIONS=200 \
    VAL_LOSS_EVERY=100 \
    TRAIN_LOG_EVERY=20 \
    MAX_WALLCLOCK_SECONDS=0 \
    EVAL_STRIDE=1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
else
    echo "  >> Full run: 20000 iters, 10-min wallclock cap"
    NCCL_IB_DISABLE=1 \
    RUN_ID="track_a_full_$(date +%Y%m%d_%H%M%S)" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    MAX_WALLCLOCK_SECONDS=600 \
    VAL_LOSS_EVERY=1000 \
    TRAIN_LOG_EVERY=50 \
    EVAL_STRIDE=64 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
fi

echo ""
echo "===== Training complete ====="
echo "Check logs/ for run logs and final_model.int8.ptz + final_model.int6.ptz artifacts"
echo "Submission size check:"
ls -la final_model.int8.ptz final_model.int6.ptz 2>/dev/null
CODE_BYTES=$(wc -c < train_gpt.py)
echo "Code size: $CODE_BYTES bytes"
for f in final_model.int8.ptz final_model.int6.ptz; do
    if [ -f "$f" ]; then
        MODEL_BYTES=$(wc -c < "$f")
        TOTAL=$((CODE_BYTES + MODEL_BYTES))
        echo "$f: $MODEL_BYTES bytes → total: $TOTAL bytes (limit: 16000000)"
    fi
done
