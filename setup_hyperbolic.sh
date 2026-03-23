#!/usr/bin/env bash
# === Hyperbolic.ai 8x H100 Setup Script ===
# Run this after SSHing into your instance
#
# Usage:
#   wget https://raw.githubusercontent.com/User123331/runpod-testing/main/setup_hyperbolic.sh
#   chmod +x setup_hyperbolic.sh
#   ./setup_hyperbolic.sh
#
# Or paste directly from clipboard

set -euo pipefail

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
log "Detected ${GPU_COUNT} GPU(s)"

if [ "${GPU_COUNT}" -lt 8 ]; then
    log "WARNING: Expected 8 GPUs, found ${GPU_COUNT}"
fi

# Keep-alive to prevent timeout during long builds
(while true; do sleep 60; nvidia-smi > /dev/null 2>&1; done) &
KEEPALIVE_PID=$!
trap "kill ${KEEPALIVE_PID} 2>/dev/null" EXIT

# 1. Clone the competition repo (already in image, but verify)
if [ ! -d "/workspace/parameter-golf" ]; then
    log "Cloning parameter-golf repo..."
    cd /workspace
    git clone https://github.com/openai/parameter-golf.git
    cd parameter-golf
else
    log "parameter-golf repo already exists"
    cd /workspace/parameter-golf
fi

# 2. Clone our MoS-enhanced training scripts
log "Cloning runpod-testing repo with MoS implementation..."
if [ ! -d "/workspace/runpod-testing" ]; then
    cd /workspace
    git clone https://github.com/User123331/runpod-testing.git
else
    cd /workspace/runpod-testing
    git pull || true
fi

# 3. Build Flash Attention 3 (selective, ~5 min)
log "Building Flash Attention 3 (selective kernels only)..."
if python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "FA3 already installed"
else
    FA3_DIR="/workspace/flash-attention"
    if [ ! -d "${FA3_DIR}" ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git "${FA3_DIR}"
    fi
    cd "${FA3_DIR}/hopper"
    rm -rf build/
    mkdir -p flash_attn_3

    # Only build bf16 hdim64 SM90 causal — skip everything else
    export FLASH_ATTENTION_DISABLE_FP16=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    export FLASH_ATTENTION_DISABLE_SM80=TRUE
    export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
    export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
    export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
    export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE

    log "Starting FA3 selective build (~5 min)..."
    pip install --no-build-isolation -e .
    log "FA3 build complete"
fi

# 4. Download dataset (80 train shards = 8B tokens)
cd /workspace/parameter-golf
log "Downloading FineWeb dataset (8B tokens)..."
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# 5. Quick sanity check
log ""
log "=== Setup Complete ==="
log "GPU Count: ${GPU_COUNT}"
log "FA3 Status: $(python3 -c 'from flash_attn_interface import flash_attn_func; print("OK")' 2>/dev/null || echo 'FAILED')"
log "Dataset: $(ls -1 data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) train shards"
log ""
log "Ready to run experiments!"
log "See: /workspace/runpod-testing/run_mos_sota.sh"