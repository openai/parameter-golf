#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# RunPod Setup — Cadence Ablation Science (H1 + H2)
# ═══════════════════════════════════════════════════════════════════════
#
# Usage (on fresh RunPod 8xH100 with PyTorch 2.9+/CUDA 12.8):
#   cd /workspace
#   git clone https://github.com/newjordan/parameter-golf.git
#   cd parameter-golf
#   git checkout experiments/pr374-edge
#   bash experiments/setup_runpod.sh
#
# Then launch:
#   bash experiments/run_all.sh          # sequential, all 8 arms
#   bash experiments/run_all.sh H1       # just H1 (4 arms)
#   bash experiments/run_all.sh H2       # just H2 (4 arms)
#
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "CADENCE ABLATION SETUP — H1 (4x2) + H2 (6x2), 8xH100"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── [1/6] System info ──
echo "=== [1/6] System info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, got $GPU_COUNT."
    echo "Set NPROC=$GPU_COUNT when launching."
fi
echo ""

# ── [2/6] Core deps ──
echo "=== [2/6] Core deps ==="
pip install -q sentencepiece numpy zstandard 2>&1 | tail -1
python3 -c "import sentencepiece; import zstandard; print('sentencepiece + zstandard OK')"
echo ""

# ── [3/6] Flash Attention 3 ──
echo "=== [3/6] Flash Attention 3 ==="
if python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "FA3 already installed, skipping build"
else
    if [ ! -d "flash-attention" ]; then
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
    fi
    cd flash-attention/hopper
    mkdir -p flash_attn_3

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

    echo "Building FA3 (selective, ~5 min)..."
    python3 -m pip install --no-build-isolation -e . 2>&1 | tail -5
    cd ../..
    echo "FA3 build complete"
fi
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 import OK')"
echo ""

# ── [4/6] Data check ──
echo "=== [4/6] Data check ==="
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_COUNT, Val shards: $VAL_COUNT"
if [ "$TRAIN_COUNT" -eq 0 ] || [ "$VAL_COUNT" -eq 0 ]; then
    echo "ERROR: Missing data shards!"
    echo "Run: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi
ls -lh data/tokenizers/fineweb_1024_bpe.model
echo ""

# ── [5/6] Preflight — training script + experiments ──
echo "=== [5/6] Preflight ==="
export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
python3 -c "
import torch, sys, ast

# CUDA
assert torch.cuda.is_available(), 'No CUDA'
cap = torch.cuda.get_device_capability()
assert cap[0] >= 9, f'Need SM90+ (Hopper), got SM{cap[0]}{cap[1]}'
print(f'CUDA: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
print(f'Memory per GPU: {torch.cuda.get_device_properties(0).total_mem // 1024**3} GB')

# Imports
from flash_attn_interface import flash_attn_func
import sentencepiece, zstandard, numpy
print('All imports OK')

# Parse the diagnostic training script
ast.parse(open('train_gpt_diag_ts_polar.py').read())
print('train_gpt_diag_ts_polar.py parses OK')
"

# Verify experiment scripts exist
echo ""
echo "Experiment scripts:"
H1_COUNT=$(ls experiments/H1_cadence_characterization/*.sh 2>/dev/null | wc -l)
H2_COUNT=$(ls experiments/H2_cadence_x_architecture/*.sh 2>/dev/null | wc -l)
echo "  H1 (4f2cx2 cadence sweep): $H1_COUNT arms"
echo "  H2 (3f3cx2 cadence sweep): $H2_COUNT arms"
if [ "$H1_COUNT" -lt 4 ] || [ "$H2_COUNT" -lt 4 ]; then
    echo "ERROR: Missing experiment scripts!"
    exit 1
fi

# Verify runner exists
if [ ! -f "experiments/run_all.sh" ]; then
    echo "ERROR: experiments/run_all.sh not found!"
    exit 1
fi
echo ""

# ── [6/6] Summary ──
echo "═══════════════════════════════════════════════════════════════"
echo "PREFLIGHT PASSED"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "8 ablation arms ready (4 × H1 + 4 × H2), 0.25 scale (150s each)"
echo ""
echo "Launch commands:"
echo "  bash experiments/run_all.sh          # all 8 arms (~20 min)"
echo "  bash experiments/run_all.sh H1       # H1 only: 4f2cx2 cadence sweep"
echo "  bash experiments/run_all.sh H2       # H2 only: 3f3cx2 cadence sweep"
echo ""
echo "Results will be in:"
echo "  experiments/H1_cadence_characterization/results/"
echo "  experiments/H2_cadence_x_architecture/results/"
echo ""
