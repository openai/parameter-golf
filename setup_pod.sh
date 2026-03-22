#!/bin/bash
# Pod setup script for leapfrog variants (v1/v2/v3)
# Run from repo root after SSH into a fresh RunPod 8xH100 instance
set -euo pipefail

echo "=== [1/6] System info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
echo ""

echo "=== [2/6] Core deps ==="
pip install -q sentencepiece numpy zstandard 2>&1 | tail -1
python3 -c "import sentencepiece; import zstandard; print('sentencepiece + zstandard OK')"
echo ""

echo "=== [3/6] Flash Attention 3 — selective build (bf16, hdim64, SM90, causal only) ==="
if python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "FA3 already installed, skipping build"
else
    if [ ! -d "flash-attention" ]; then
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
    fi
    cd flash-attention/hopper

    # Disable everything we don't need — builds ~2 kernels instead of 451
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
    pip install -e . 2>&1 | tail -5
    cd ../..
    echo "FA3 build complete"
fi
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 import OK')"
echo ""

echo "=== [4/6] Data check ==="
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_COUNT, Val shards: $VAL_COUNT"
if [ "$TRAIN_COUNT" -eq 0 ] || [ "$VAL_COUNT" -eq 0 ]; then
    echo "ERROR: Missing data shards! Check data/datasets/fineweb10B_sp1024/"
    exit 1
fi
ls -lh data/tokenizers/fineweb_1024_bpe.model
echo ""

echo "=== [5/6] Preflight — CUDA + imports + parse all variants ==="
PYTHONPATH=flash-attention/hopper:${PYTHONPATH:-} python3 -c "
import torch, sys
assert torch.cuda.is_available(), 'No CUDA'
cap = torch.cuda.get_device_capability()
assert cap[0] >= 9, f'Need SM90+ (Hopper), got SM{cap[0]}{cap[1]}'
print(f'CUDA devices: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
print(f'Memory per GPU: {torch.cuda.get_device_properties(0).total_mem // 1024**3} GB')
from flash_attn_interface import flash_attn_func
import sentencepiece, zstandard, numpy
print('All imports OK')
import ast
for v in ['train_gpt_v1.py', 'train_gpt_v2.py', 'train_gpt_v3.py']:
    ast.parse(open(v).read())
    print(f'{v} parses OK')
"
echo ""

echo "=== [6/6] Export PYTHONPATH ==="
export PYTHONPATH=flash-attention/hopper:${PYTHONPATH:-}
echo "PYTHONPATH=$PYTHONPATH"
echo ""

echo "=== READY ==="
echo ""
echo "Run variants (one at a time, or on separate pods):"
echo ""
echo "  # v3 — Control (unmodified PR#414 baseline)"
echo "  PYTHONPATH=flash-attention/hopper:\$PYTHONPATH SEED=1337 torchrun --nproc_per_node=8 train_gpt_v3.py"
echo ""
echo "  # v1 — TTT Burst (2 epochs on last 100 batches at 10% LR)"
echo "  PYTHONPATH=flash-attention/hopper:\$PYTHONPATH SEED=1337 torchrun --nproc_per_node=8 train_gpt_v1.py"
echo ""
echo "  # v2 — Self-Distillation (50 steps KL+CE against EMA teacher)"
echo "  PYTHONPATH=flash-attention/hopper:\$PYTHONPATH SEED=1337 torchrun --nproc_per_node=8 train_gpt_v2.py"
echo ""
echo "Compare: grep 'final_int6_sliding_window_s64_exact' on each log"
