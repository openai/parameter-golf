#!/bin/bash
# Parameter Golf — H100 RunPod Run Script
# SOTA architecture + 1-sqrt warmdown schedule
# Expected: ~1.06-1.08 BPB on 8xH100

set -e

echo "=== Step 1: Upgrade PyTorch ==="
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade

echo "=== Step 2: Install dependencies ==="
pip install sentencepiece brotli
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

echo "=== Step 3: Verify setup ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
from flash_attn_interface import flash_attn_func
print('FlashAttention 3: OK')
"

echo "=== Step 4: Clone repo and download SP8192 data ==="
cd /workspace
git clone https://github.com/vikrant-akavaram/parameter-golf.git
cd parameter-golf
git checkout submission

# Download SP8192 data from kevclark dataset repo
# First get the manifest
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('kevclark/parameter-golf', 'datasets/manifest.json', repo_type='dataset')
shutil.copy(path, 'data/manifest.json')
print('SP8192 manifest ready')
"
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

echo "=== Step 5: Verify data ==="
ls data/datasets/fineweb10B_sp8192/ | head -5
ls data/tokenizers/fineweb_8192_bpe.*

echo "=== Step 6: Run training (SOTA + 1-sqrt warmdown) ==="
# Using the decoded v4 script with SDPA fallback + 1-sqrt patches
# FA3 will be auto-detected and used natively on H100
# torch.compile is ENABLED (no TORCHDYNAMO_DISABLE)
SEED=1337 RUN_ID=h100_v4_sqrt \
  torchrun --standalone --nproc_per_node=8 train_gpt_v4_sqrt.py

echo "=== DONE ==="
echo "Check logs/h100_v4_sqrt.txt for results"
echo "Look for: final_int6_sliding_window val_bpb"
