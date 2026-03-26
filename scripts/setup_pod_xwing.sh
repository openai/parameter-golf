#!/usr/bin/env bash
# setup_pod_green_v2.sh — Prepare RunPod for X-WING
#
# Pipe via SSH:
#   cat scripts/setup_pod_green_v2.sh | ssh -tt -i ~/.ssh/id_ed25519_apollo root@POD
set -euo pipefail

WORKSPACE="/workspace/parameter-golf"
FLASH_DIR="${WORKSPACE}/flash-attention/hopper"

echo "============================================"
echo "  X-WING — Pod Setup"
echo "  $(date)"
echo "============================================"

# --- Repo ---
cd /workspace
if [ ! -d "$WORKSPACE" ]; then
    echo "==> Cloning repo..."
    git clone https://github.com/openai/parameter-golf.git
fi
cd "$WORKSPACE"
echo "==> Pulling latest..."
git fetch origin 2>&1 | tail -3
git checkout experiments/pr374-edge 2>/dev/null || git checkout -b experiments/pr374-edge origin/experiments/pr374-edge
git pull --ff-only origin experiments/pr374-edge 2>&1 | tail -3 || git reset --hard origin/experiments/pr374-edge
echo "    HEAD: $(git log --oneline -1)"

# --- Deps ---
echo ""
echo "==> Installing deps..."
pip install -q zstandard sentencepiece 2>&1 | tail -3
python3 -c "
import torch, sentencepiece, zstandard, numpy
print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'  sentencepiece OK, zstandard OK, numpy OK')
"

# --- Flash Attention ---
echo ""
echo "==> Flash Attention (Hopper)..."
if [ -d "$FLASH_DIR" ]; then
    export PYTHONPATH="${FLASH_DIR}:${PYTHONPATH:-}"
    python3 -c "from flash_attn_interface import flash_attn_func; print('  flash_attn_interface OK')" 2>&1 || {
        echo "  Rebuilding..."
        cd "$FLASH_DIR" && pip install -e . 2>&1 | tail -5
        cd "$WORKSPACE"
    }
else
    echo "  No hopper dir, building from pip..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -5
    python3 -c "
import sys, os
shim = 'from flash_attn.flash_attn_interface import flash_attn_func\n'
site = [p for p in sys.path if 'site-packages' in p and os.path.isdir(p)][0]
with open(os.path.join(site, 'flash_attn_interface.py'), 'w') as f:
    f.write(shim)
print('  flash_attn_interface shim created')
"
fi

# --- Data check ---
echo ""
echo "==> Data check..."
VAL_COUNT=$(ls ${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
TRAIN_COUNT=$(ls ${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "  Train shards: ${TRAIN_COUNT}, Val shards: ${VAL_COUNT}"
[ -f "${WORKSPACE}/data/tokenizers/fineweb_1024_bpe.model" ] && echo "  Tokenizer: OK" || echo "  WARNING: tokenizer missing!"

# --- Dirs ---
mkdir -p "${WORKSPACE}/logs" "${WORKSPACE}/checkpoints"

echo ""
echo "============================================"
echo "  SETUP COMPLETE — Ready to race"
echo "============================================"
echo ""
echo "  Run X-WING (shared tables + cubric):"
echo ""
echo "    cd ${WORKSPACE}"
echo "    SEED=1337 NPROC_PER_NODE=8 bash concepts/xwing/run.sh"
echo ""
echo "  Additional seeds:"
echo "    SEED=42 NPROC_PER_NODE=8 bash concepts/xwing/run.sh"
echo "    SEED=2024 NPROC_PER_NODE=8 bash concepts/xwing/run.sh"
echo ""
