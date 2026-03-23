#!/usr/bin/env bash
# === Hyperbolic.ai Quick Start (Paste Into SSH) ===
# Paste this entire block after SSHing into your 8x H100 instance

set -euo pipefail
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
(while true; do sleep 60; nvidia-smi > /dev/null 2>&1; done) &
trap "kill $! 2>/dev/null" EXIT

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
log "Detected ${GPU_COUNT} GPUs"

# Clone repos
cd /workspace
[ ! -d "parameter-golf" ] && git clone https://github.com/openai/parameter-golf.git
[ ! -d "runpod-testing" ] && git clone https://github.com/User123331/runpod-testing.git

# Build FA3 selectively
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "Building FA3 (~5 min)..."
    [ ! -d "flash-attention" ] && git clone https://github.com/Dao-AILab/flash-attention.git
    cd /workspace/flash-attention/hopper
    rm -rf build/ && mkdir -p flash_attn_3
    export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM128=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    export FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_LOCAL=TRUE
    export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
    pip install --no-build-isolation -e .
fi

# Download dataset
cd /workspace/parameter-golf
log "Downloading dataset (8B tokens)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

log "Setup complete! Ready for MoS experiments."