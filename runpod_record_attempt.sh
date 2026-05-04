#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/runpod_workdir}"
REPO_URL="https://github.com/openai/parameter-golf.git"
REPO_DIR="$WORK_DIR/parameter-golf"
TARGET_REL_PATH="records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
TARGET_DIR="$REPO_DIR/$TARGET_REL_PATH"
TRAIN_SCRIPT="$TARGET_DIR/train_gpt.py"

SEEDS=(42 314 999)
NUM_GPUS="${NUM_GPUS:-8}"

log "Starting RunPod record attempt workflow"
log "Using work directory: $WORK_DIR"
mkdir -p "$WORK_DIR"

log "Cloning/syncing latest $REPO_URL"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" checkout main
  git -C "$REPO_DIR" pull --ff-only origin main
else
  git clone "$REPO_URL" "$REPO_DIR"
  git -C "$REPO_DIR" checkout main
fi

log "Installing required Python dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch sentencepiece

log "Attempting to install flash_attn (non-fatal if this fails)"
if python -m pip install --no-build-isolation flash_attn; then
  log "flash_attn installed successfully"
else
  log "WARNING: flash_attn installation failed; continuing without it"
fi

log "Preparing environment for multi-GPU training (${NUM_GPUS}x H100 expected)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
log "Detected CUDA devices: ${GPU_COUNT}"

if [[ ! -d "$TARGET_DIR" ]]; then
  log "ERROR: target directory not found: $TARGET_DIR"
  log "Available recent record directories:"
  find "$REPO_DIR/records/track_10min_16mb" -maxdepth 1 -type d | sort | tail -n 30
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  log "ERROR: training script not found: $TRAIN_SCRIPT"
  exit 1
fi

log "Running official SOTA training script: $TARGET_REL_PATH/train_gpt.py"
cd "$TARGET_DIR"

for seed in "${SEEDS[@]}"; do
  log "========================================"
  log "Launching training with seed=${seed}"
  log "Command: torchrun --standalone --nproc_per_node=${NUM_GPUS} train_gpt.py --seed ${seed}"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" train_gpt.py --seed "$seed" 2>&1 | tee "runpod_seed${seed}.log"
  log "Completed seed=${seed}"
done

log "All requested seeds completed successfully"
