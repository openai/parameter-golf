#!/usr/bin/env bash
# Quick single-GPU smoke test for the anchor script on Pegasus
# Tests: no NaN, EMA runs, int6+zstd export works, sliding eval emits metrics
#
# Usage (from login node):
#   bash scripts/pegasus_smoke_test.sh [script_path]
#
# This uses A100-80GB by default (easier to get than H100).
# Override with: PARTITION=H100 bash scripts/pegasus_smoke_test.sh

set -euo pipefail

SCRIPT_PATH="${1:-records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py}"
PARTITION="${PARTITION:-A100-80GB}"
REPO_PATH="/netscratch/${USER}/parameter-golf"
LOG_PATH="/netscratch/${USER}/smoke_test_$(date +%Y%m%d_%H%M%S).log"

# Prefer fscratch if available
if [ -d "/fscratch/${USER}/parameter-golf-data/datasets/fineweb10B_sp1024" ]; then
    DATA_PATH="/fscratch/${USER}/parameter-golf-data/datasets/fineweb10B_sp1024"
    TOKENIZER_PATH="/fscratch/${USER}/parameter-golf-data/tokenizers/fineweb_1024_bpe.model"
else
    DATA_PATH="${REPO_PATH}/data/datasets/fineweb10B_sp1024"
    TOKENIZER_PATH="${REPO_PATH}/data/tokenizers/fineweb_1024_bpe.model"
fi

# Container image (optional)
CONTAINER_ARGS=""
if ls /enroot/nvcr.io_nvidia_pytorch_*.sqsh >/dev/null 2>&1; then
    LATEST=$(ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh | sort -V | tail -1)
    CONTAINER_ARGS="--container-image=${LATEST}"
    CONTAINER_ARGS="${CONTAINER_ARGS} --container-mounts=/netscratch/${USER}:/netscratch/${USER}"
    if [[ "${DATA_PATH}" == /fscratch/* ]]; then
        CONTAINER_ARGS="${CONTAINER_ARGS},/fscratch/${USER}:/fscratch/${USER}"
    fi
    CONTAINER_ARGS="${CONTAINER_ARGS} --container-workdir=${REPO_PATH}"
    echo "Using container: $(basename $LATEST)"
fi

echo "=== Smoke Test ==="
echo "Script:    ${SCRIPT_PATH}"
echo "Partition: ${PARTITION}"
echo "Data:      ${DATA_PATH}"
echo "Log:       ${LOG_PATH}"
echo "==================="
echo ""

srun -K \
    -p "${PARTITION}" \
    --ntasks=1 \
    --gpus-per-task=1 \
    --cpus-per-task=6 \
    --time=00:20:00 \
    ${CONTAINER_ARGS} \
    bash -c '
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 USE_OPENMP=1
ulimit -n 4000 2>/dev/null || true

# Install missing deps inside container (fast via Pegasus PyPI cache)
pip install --quiet --no-cache --index-url http://pypi-cache/index --trusted-host pypi-cache \
    sentencepiece zstandard 2>/dev/null \
  || pip install --quiet sentencepiece zstandard 2>/dev/null \
  || true

cd '"${REPO_PATH}"'

MAX_WALLCLOCK_SECONDS=90 \
ITERATIONS=200 \
DATA_PATH='"${DATA_PATH}"' \
TOKENIZER_PATH='"${TOKENIZER_PATH}"' \
VOCAB_SIZE=1024 \
AMP_DTYPE=auto \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=25 \
RUN_ID=smoke_test \
python3 -u '"${SCRIPT_PATH}"'
' 2>&1 | tee "${LOG_PATH}"

echo ""
echo "=== Smoke Test Validation ==="

# Check for failures
if grep -q "NaN" "${LOG_PATH}"; then
    echo "FAIL: NaN detected in training"
elif grep -q "Error\|Traceback\|RuntimeError" "${LOG_PATH}"; then
    echo "FAIL: Errors detected — check log"
elif grep -q "final_int6" "${LOG_PATH}"; then
    echo "PASS: int6 export completed"
    grep "model_params\|final_int6\|sliding\|submission.*bytes\|ema:" "${LOG_PATH}" || true
else
    echo "WARN: Could not verify int6 export — check log manually"
fi

echo ""
echo "Full log: ${LOG_PATH}"
