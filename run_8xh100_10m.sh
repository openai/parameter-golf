#!/bin/bash
# 8×H100 SXM 10-min run using DECODED SOTA code
# All SOTA hyperparams are defaults in train_gpt_sota.py
# Checkpoints + logs saved to network volume for persistence
#
# Pod requirements:
#   - Template: parameter-golf (runpod/parameter-golf:latest)
#   - ENV: PUBLIC_KEY must be set for SSH access
#   - pip install brotli --break-system-packages (not in template)
set -e

# Install brotli if missing (not on network volume, needs per-pod install)
python3 -c "import brotli" 2>/dev/null || pip install brotli --break-system-packages -q

EXP_NAME="${EXP_NAME:-exp_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="/workspace/runs/$(date +%Y-%m-%d)-${EXP_NAME}"
LOG_FILE="${RUN_DIR}/train.log"
CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "$CKPT_DIR"

echo "=== Run: $EXP_NAME ==="
echo "Run dir: $RUN_DIR"
echo "Log: $LOG_FILE"
echo "Checkpoints: $CKPT_DIR"

cd /workspace/parameter-golf

CKPT_DIR="$CKPT_DIR" \
CKPT_STEPS=1000,2000,3000,4000,5000 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py 2>&1 | tee "$LOG_FILE"

# Copy final artifacts to run dir
cp -f final_model.pt final_model.int6.ptz "$RUN_DIR/" 2>/dev/null || true

echo "=== Training done. Files saved to $RUN_DIR ==="
echo "=== Stopping pod in 30s (Ctrl+C to cancel) ==="
sleep 30
runpodctl stop pod $RUNPOD_POD_ID
