#!/usr/bin/env bash
# One-shot setup for a fresh RunPod (template y5cejece4j) for the Opus push.
# Idempotent — safe to re-run.
#
# Usage: bash setup_pod.sh
#
# Assumes:
#   - You're on a RunPod GPU pod with the Parameter Golf template
#   - /workspace is the persistent volume

set -euo pipefail

REPO_URL="https://github.com/GodlyDonuts/parameter-golf.git"
BRANCH="claude/busy-thompson-9c94f9"
WORKDIR="/workspace/parameter-golf"

echo "[1/5] Cloning repo into $WORKDIR (or pulling if exists)"
if [ -d "$WORKDIR/.git" ]; then
    cd "$WORKDIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

echo "[2/5] Installing Python deps (brotli, sentencepiece, FA3)"
pip install --quiet brotli sentencepiece
pip install --quiet flash_attn_3 --no-deps \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ \
    || echo "WARNING: FA3 wheel install failed — script will fall back to SDPA"

echo "[3/5] Downloading sp8192 dataset (cached fineweb)"
if [ -d "data/datasets/fineweb10B_sp8192" ]; then
    echo "  Dataset already present — skipping download"
else
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python3 data/cached_challenge_fineweb.py --variant sp8192
fi

echo "[4/5] Sanity check — GPU"
nvidia-smi | head -20
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}, version: {torch.__version__}')"

echo "[5/5] Sanity check — flash-attn-3"
python3 -c "
try:
    from flash_attn_interface import flash_attn_func
    print('flash_attn_3 available')
except ImportError:
    try:
        from flash_attn import flash_attn_func
        print('flash_attn (v2) available — SOTA may run slower')
    except ImportError:
        print('NO flash-attn — SDPA fallback (slow)')
"

echo
echo "Setup complete. Next: bash Opus/scripts/run_repro.sh"
