#!/usr/bin/env bash
# Stage 1: Download sp8192 dataset (idempotent — skips if already present)
# Verifies 128 train shards + at least 1 val shard before continuing
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
LOG_FILE="${REPO_DIR}/runs/01_download_data.log"
PYTHON="/opt/pg-venv/bin/python"
DATA_DIR="${REPO_DIR}/data/datasets/fineweb10B_sp8192"
REQUIRED_TRAIN_SHARDS=128

mkdir -p "${REPO_DIR}/runs"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Stage 1: Dataset download === $(date)"

# Free space guard (needs ~24 GB)
FREE_GB=$(df -BG /workspace --output=avail 2>/dev/null | tail -1 | tr -d 'G ')
if [ "${FREE_GB:-0}" -lt 30 ]; then
    echo "ERROR: only ${FREE_GB}G free in /workspace; need >= 30G for dataset" >&2
    exit 1
fi

cd "${REPO_DIR}"
mkdir -p "${DATA_DIR}"

count_shards() {
    shopt -s nullglob
    local files=("${DATA_DIR}"/fineweb_train_*.bin)
    shopt -u nullglob
    printf '%s\n' "${#files[@]}"
}
count_val() {
    shopt -s nullglob
    local files=("${DATA_DIR}"/fineweb_val_*.bin)
    shopt -u nullglob
    printf '%s\n' "${#files[@]}"
}

TRAIN_COUNT=$(count_shards)
VAL_COUNT=$(count_val)

if [ "${TRAIN_COUNT}" -ge "${REQUIRED_TRAIN_SHARDS}" ] && [ "${VAL_COUNT}" -ge 1 ]; then
    echo "Data already present: ${TRAIN_COUNT} train shards, ${VAL_COUNT} val shards. Skipping download."
    echo "01_download_data: PASS (cached)"
    exit 0
fi

echo "Downloading sp8192 from kevclark/parameter-golf..."
echo "  Train shards found so far: ${TRAIN_COUNT}/${REQUIRED_TRAIN_SHARDS}"
echo "  Val shards found so far:   ${VAL_COUNT}"

export HF_HOME=/workspace/.hf
export XDG_CACHE_HOME=/workspace/.cache
export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${HUGGINGFACE_HUB_CACHE}"

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    "${PYTHON}" data/cached_challenge_fineweb.py \
    --variant sp8192 --train-shards "${REQUIRED_TRAIN_SHARDS}"

TRAIN_COUNT=$(count_shards)
VAL_COUNT=$(count_val)
echo ""
echo "Post-download shard counts:"
echo "  Train: ${TRAIN_COUNT}"
echo "  Val:   ${VAL_COUNT}"

if [ "${VAL_COUNT}" -lt 1 ]; then
    echo "ERROR: no validation shards found after download" >&2
    exit 1
fi
if [ "${TRAIN_COUNT}" -lt "${REQUIRED_TRAIN_SHARDS}" ]; then
    echo "ERROR: only ${TRAIN_COUNT} train shards (need ${REQUIRED_TRAIN_SHARDS})" >&2
    exit 1
fi

# Verify tokenizer path expected by train_gpt.py:177 (data_dir/tokenizers/fineweb_{vocab_size}_bpe.model)
[ -f "data/tokenizers/fineweb_8192_bpe.model" ] || \
    echo "WARNING: tokenizer not found at data/tokenizers/fineweb_8192_bpe.model — train_gpt.py will fail at data load"

echo ""
echo "01_download_data: PASS"
