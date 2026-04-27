#!/bin/bash
# Parameter Golf — Vast.ai smoke test runner
# Tests a specific step file and reports result
# Usage: bash vast_smoke_test.sh <step_file> <step_name>

set -e

STEP_FILE="${1:-train_gpt_step6.py}"
STEP_NAME="${2:-step6}"
PG_DIR="/root/parameter-golf"
DATA_DIR="$PG_DIR/data/datasets/fineweb10B_sp1024"
TOK_PATH="$PG_DIR/data/tokenizers/fineweb_1024_bpe.model"

echo "=== Smoke Test: $STEP_NAME ==="
echo "Script: $STEP_FILE"

# Ensure repo is cloned and up to date
if [ ! -d "$PG_DIR" ]; then
    echo "Cloning repo..."
    git clone https://github.com/nickferrantelive/parameter-golf.git "$PG_DIR"
else
    echo "Pulling latest..."
    cd "$PG_DIR" && git pull --ff-only 2>/dev/null || true
fi

cd "$PG_DIR"

# Check if data exists, download if not
if [ ! -f "$DATA_DIR/fineweb_train_0.bin" ]; then
    echo "Downloading data (1 shard)..."
    pip install -q huggingface-hub sentencepiece
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

echo "Data OK. Running smoke test..."

# Run smoke test: 60 seconds, small batch
python3 "$STEP_FILE" \
    --run-id "smoke_${STEP_NAME}" \
    2>&1 | head -5 || true

# Use env vars since scripts don't support --args
MAX_WALLCLOCK_SECONDS=60 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=512 \
VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 \
python3 "$STEP_FILE" 2>&1

echo "=== Done: $STEP_NAME ==="
