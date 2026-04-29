#!/bin/bash
# Parameter Golf — H100 RunPod Run Script (v3 — LZMA compressed code)
# SOTA architecture + 1-sqrt warmdown + compressed code wrapper
# Pod requirements: 8xH100 SXM, 100GB volume disk, ON-DEMAND (not spot!)
# Expected: ~1.08 BPB, artifact under 16MB

set -e

# === Cache redirects (prevent /tmp disk full) ===
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_cache
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/.torch_cache /workspace/.cache/huggingface

echo "=== Step 1: Install PyTorch 2.9.1 (matches FA3 wheel) ==="
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

echo "=== Step 2: Install dependencies ==="
pip install sentencepiece brotli huggingface_hub
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

echo "=== Step 3: Verify setup ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')
from flash_attn_interface import flash_attn_func
print('FlashAttention 3: OK')
"

echo "=== Step 4: Clone repo ==="
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/vikrant-akavaram/parameter-golf.git
    cd parameter-golf
    git checkout submission
else
    cd parameter-golf
    git checkout submission
    git pull origin submission
fi

echo "=== Step 5: Download SP8192 data ==="
mkdir -p data/datasets data/tokenizers
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('kevclark/parameter-golf', 'datasets/manifest.json', repo_type='dataset')
shutil.copy(path, 'data/manifest.json')
print('SP8192 manifest ready')
"
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

echo "=== Step 6: Fix data paths ==="
find /workspace -name "fineweb10B_sp8192" -type d | head -1 | xargs -I{} ln -sf {} data/datasets/fineweb10B_sp8192
find /workspace -name "fineweb_8192_bpe.model" | head -1 | xargs -I{} ln -sf {} data/tokenizers/fineweb_8192_bpe.model
find /workspace -name "fineweb_8192_bpe.vocab" | head -1 | xargs -I{} ln -sf {} data/tokenizers/fineweb_8192_bpe.vocab
echo "Data files:"
ls data/datasets/fineweb10B_sp8192/ | head -3
ls data/tokenizers/fineweb_8192_bpe.*

echo "=== Step 7: Run training (compressed script with 1-sqrt + f-string fixes) ==="
# Uses LZMA-compressed code wrapper (16KB vs 49KB = saves 32KB for artifact budget)
SEED=1337 RUN_ID=h100_final \
  torchrun --standalone --nproc_per_node=8 train_run_compressed.py

echo "=== DONE ==="
echo "Check logs/h100_final.txt for results"
echo "Look for: Total submission size, final_int6_sliding_window val_bpb"
