#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Pod setup — Micro Crawler 4f+2cx2 on 8xH100
# ═══════════════════════════════════════════════════════════════════════
#
# Usage (on fresh RunPod 8xH100 with PyTorch 2.9+/CUDA 12.8):
#   cd /workspace
#   git clone https://github.com/newjordan/parameter-golf.git
#   cd parameter-golf
#   git checkout experiments/pr374-edge
#   bash setup_pod_micro_crawler.sh
#   ./run_micro_crawler_h100.sh
#
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "MICRO CRAWLER POD SETUP — 4flat + 2crawl×2, dim=640, 8xH100"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── [1/6] System info ──
echo "=== [1/6] System info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, got $GPU_COUNT. torchrun may fail."
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

# ── [5/6] Preflight — parse + CUDA + imports ──
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

# Parse the micro crawler script
ast.parse(open('train_gpt_micro_crawler_h100.py').read())
print('train_gpt_micro_crawler_h100.py parses OK')

# Quick architecture sanity check
print()
print('Architecture config (from env or defaults):')
import os
nf = int(os.environ.get('NUM_FLAT_LAYERS', 4))
nc = int(os.environ.get('NUM_CRAWLER_LAYERS', 2))
cl = int(os.environ.get('CRAWLER_LOOPS', 2))
dim = int(os.environ.get('MODEL_DIM', 640))
cad = int(os.environ.get('CRAWLER_CADENCE', 5))
print(f'  {nf}flat + {nc}crawl x{cl} = {nf + nc*cl} effective depth')
print(f'  dim={dim}, stored_blocks={nf+nc}')
print(f'  cadence={cad} (N/N/N/N/C)')
print(f'  estimated params: ~{(nf+nc) * 11 * dim**2 / 1e6:.1f}M')
"
echo ""

# ── [6/6] Export PYTHONPATH ──
echo "=== [6/6] PYTHONPATH ==="
echo "PYTHONPATH=$PYTHONPATH"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "READY"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Run the micro crawler:"
echo "  ./run_micro_crawler_h100.sh"
echo ""
echo "Or manually:"
echo "  export PYTHONPATH=$(pwd)/flash-attention/hopper:\$PYTHONPATH"
echo "  torchrun --nproc_per_node=8 train_gpt_micro_crawler_h100.py"
echo ""
echo "Debug (if torchrun hides traceback):"
echo "  WORLD_SIZE=1 RANK=0 python3 train_gpt_micro_crawler_h100.py 2>&1 | head -80"
echo ""
echo "Common issues:"
echo "  FA3 import fails → export PYTHONPATH=$(pwd)/flash-attention/hopper:\$PYTHONPATH"
echo "  OOM              → reduce TRAIN_BATCH_TOKENS (default 786432, try 524288)"
echo "  Data missing     → python3 data/cached_challenge_fineweb.py --variant sp1024"
echo "  Parse error      → python3 -c \"import ast; ast.parse(open('train_gpt_micro_crawler_h100.py').read())\""
echo ""
