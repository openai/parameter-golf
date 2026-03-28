#!/usr/bin/env bash
# Optimized Pegasus launcher for parameter-golf anchor runs
#
# This script incorporates all known hardware wins:
#   1. NGC container (matching RunPod CUDA/cuDNN stack)
#   2. /fscratch data path (lower latency than /netscratch)
#   3. CPU thread pinning (prevent contention)
#   4. NCCL tuning for NVSwitch-only H100 partition
#   5. kill-on-bad-exit for clean failure handling
#
# Usage:
#   # First time setup:
#   bash scripts/pegasus_setup_fscratch.sh
#
#   # Then allocate and run:
#   salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 \
#     --gpu-bind=none --cpus-per-task=6 --time=02:00:00
#   bash scripts/pegasus_optimized_launcher.sh <run_id> [script_path]
#
# Or as a one-shot srun (no salloc needed):
#   bash scripts/pegasus_optimized_launcher.sh <run_id> [script_path] --srun

set -euo pipefail

# --- Configuration (edit these) ---
RUN_ID="${1:?Usage: $0 <run_id> [script_path] [--srun]}"
SCRIPT_PATH="${2:-records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py}"
MODE="${3:-salloc}"  # "salloc" (default, assumes you already allocated) or "--srun" (one-shot)

PARTITION="${PARTITION:-H100}"
REPO_PATH="/netscratch/${USER}/parameter-golf"
LOG_PATH="/netscratch/${USER}/${RUN_ID}.log"

# Data paths: prefer /fscratch if available, fall back to /netscratch
if [ -d "/fscratch/${USER}/parameter-golf-data/datasets/fineweb10B_sp1024" ]; then
    DATA_PATH="/fscratch/${USER}/parameter-golf-data/datasets/fineweb10B_sp1024"
    TOKENIZER_PATH="/fscratch/${USER}/parameter-golf-data/tokenizers/fineweb_1024_bpe.model"
    echo "Using /fscratch for data (low-latency path)"
else
    DATA_PATH="${REPO_PATH}/data/datasets/fineweb10B_sp1024"
    TOKENIZER_PATH="${REPO_PATH}/data/tokenizers/fineweb_1024_bpe.model"
    echo "Using /netscratch for data (run pegasus_setup_fscratch.sh for faster I/O)"
fi

# Container image: use latest NGC PyTorch if available
CONTAINER_IMAGE=""
if ls /enroot/nvcr.io_nvidia_pytorch_*.sqsh >/dev/null 2>&1; then
    CONTAINER_IMAGE=$(ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh | sort -V | tail -1)
    echo "Using NGC container: $(basename $CONTAINER_IMAGE)"
elif [ -f "/netscratch/${USER}/pgolf.sqsh" ]; then
    CONTAINER_IMAGE="/netscratch/${USER}/pgolf.sqsh"
    echo "Using custom container: pgolf.sqsh"
else
    echo "WARNING: No container image found. Running bare metal."
    echo "  This may produce slower results than RunPod."
fi

# --- Build the inner command ---
INNER_CMD='
# Rank mapping from Slurm vars
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# CPU thread pinning — prevent contention (Pegasus known issue)
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1

# NCCL tuning for NVSwitch-only H100 partition (no InfiniBand)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond,eth
export NCCL_P2P_LEVEL=NVL

# Increase file descriptor limit (Pegasus OMP shared memory workaround)
ulimit -n 4000 2>/dev/null || true

# Install missing deps inside container (fast via Pegasus PyPI cache)
pip install --quiet --no-cache --index-url http://pypi-cache/index --trusted-host pypi-cache \
    sentencepiece zstandard 2>/dev/null \
  || pip install --quiet sentencepiece zstandard 2>/dev/null \
  || true

cd '"${REPO_PATH}"'

RUN_ID='"${RUN_ID}"' \
DATA_PATH='"${DATA_PATH}"' \
TOKENIZER_PATH='"${TOKENIZER_PATH}"' \
VOCAB_SIZE=1024 \
AMP_DTYPE=auto \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
python3 -u '"${SCRIPT_PATH}"'
'

# --- Launch ---
echo ""
echo "=== Pegasus Optimized Launch ==="
echo "Run ID:    ${RUN_ID}"
echo "Script:    ${SCRIPT_PATH}"
echo "Partition: ${PARTITION}"
echo "Data:      ${DATA_PATH}"
echo "Log:       ${LOG_PATH}"
echo "Container: ${CONTAINER_IMAGE:-bare-metal}"
echo "Node:      ${SLURM_NODELIST:-not-yet-allocated}"
echo "================================"
echo ""

# Build container args if image is available
CONTAINER_ARGS=""
if [ -n "$CONTAINER_IMAGE" ]; then
    CONTAINER_ARGS="--container-image=${CONTAINER_IMAGE}"
    CONTAINER_ARGS="${CONTAINER_ARGS} --container-mounts=/netscratch/${USER}:/netscratch/${USER}"
    # Mount fscratch if we're using it
    if [[ "${DATA_PATH}" == /fscratch/* ]]; then
        CONTAINER_ARGS="${CONTAINER_ARGS},/fscratch/${USER}:/fscratch/${USER}"
    fi
    CONTAINER_ARGS="${CONTAINER_ARGS} --container-workdir=${REPO_PATH}"
fi

if [ "$MODE" = "--srun" ]; then
    # One-shot mode: allocate + run
    srun -K \
        -p "${PARTITION}" \
        --nodes=1 \
        --ntasks=8 \
        --gpus-per-task=1 \
        --gpu-bind=none \
        --cpus-per-task=6 \
        --time=02:00:00 \
        ${CONTAINER_ARGS} \
        bash -c "${INNER_CMD}" \
        2>&1 | tee "${LOG_PATH}"
else
    # salloc mode: assumes you already have an allocation
    srun -K \
        --gpu-bind=none \
        ${CONTAINER_ARGS} \
        bash -c "${INNER_CMD}" \
        2>&1 | tee "${LOG_PATH}"
fi

echo ""
echo "=== Run complete ==="
echo "Log saved to: ${LOG_PATH}"
echo "Copy to submission folder:"
echo "  cp ${LOG_PATH} ${REPO_PATH}/records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train.log"
