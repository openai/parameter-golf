#!/usr/bin/env bash
# Phase 4: H-Net Milestone 1 pilot.
#
# Prerequisites (done once on the pod):
#   - Phase 0 already ran (parameter-golf cloned, SP8192 data downloaded).
#   - train_gpt_baseline.py already unpacked at /workspace/work/.
#   - FA3 installed (phase 2a).
#
# This script:
#   1. Converts 2 SP8192 train shards + 1 val shard into UTF-8 byte shards.
#   2. Runs train_hnet_m1.py for ITERATIONS steps (default 300).
#   3. Prints final per-byte val_bpb.
set -euo pipefail
cd /workspace

# Where H-Net code lives (uploaded alongside this script)
HNET_DIR="/workspace/hnet_m1"
DATA_DIR="/workspace/parameter-golf/data"
SP_SHARDS="${DATA_DIR}/datasets/fineweb10B_sp8192/fineweb_*_*.bin"
TOKENIZER="${DATA_DIR}/tokenizers/fineweb_8192_bpe.model"
BYTE_DIR="${DATA_DIR}/datasets/fineweb10B_bytes"

echo "=== PHASE 4 (H-Net M1) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# --- 1. byte shard preprocessing --------------------------------------------
if [ -z "$(ls -A ${BYTE_DIR} 2>/dev/null)" ]; then
    echo "--- converting SP8192 shards to UTF-8 byte shards ---"
    python "${HNET_DIR}/make_byte_shards.py" \
        --tokenizer "${TOKENIZER}" \
        --in-pattern "${SP_SHARDS}" \
        --out-dir "${BYTE_DIR}"
else
    echo "--- byte shards already present at ${BYTE_DIR}, skipping preprocess ---"
fi
ls -lh "${BYTE_DIR}/" | head

# --- 2. pilot training run --------------------------------------------------
echo
echo "--- H-Net M1 pilot ---"
export DATA_DIR
export BYTE_DATA_DIR="${BYTE_DIR}"
export TOKENIZER_PATH="${TOKENIZER}"
export BASELINE_PATH="/workspace/work/train_gpt_baseline.py"
export ITERATIONS="${ITERATIONS:-300}"
export BYTE_SEQ_LEN="${BYTE_SEQ_LEN:-4096}"
export CHUNK_STRIDE="${CHUNK_STRIDE:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LR="${LR:-3e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export RUN_ID="${RUN_ID:-hnet_m1_pilot}"
export SEED="${SEED:-42}"

python "${HNET_DIR}/train_hnet_m1.py"

echo
echo "--- tail of log ---"
tail -n 20 "/workspace/logs/${RUN_ID}.txt"

echo "=== PHASE 4 DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
