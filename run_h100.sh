#!/bin/bash
# Parameter Golf — H100 RunPod Run Script (v2 — fixed)
# SOTA architecture + 1-sqrt warmdown schedule
# Pod requirements: 8xH100 SXM, 100GB volume disk, web terminal
# Expected: ~1.06-1.08 BPB

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
print(f'CUDA: {torch.version.cuda}')
from flash_attn_interface import flash_attn_func
print('FlashAttention 3: OK')
"

echo "=== Step 4: Download SOTA script and patch ==="
cd /workspace
curl -sL "https://raw.githubusercontent.com/openai/parameter-golf/main/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py" -o train_sota.py

# Decompress the LZMA-packed SOTA script, patch for Python 3.11 + 1-sqrt warmdown
python3 << 'PYEOF'
import lzma as L, base64 as B, re

with open("train_sota.py") as f:
    wrapper = f.read()

m = re.search(r'B\.b85decode\("(.+?)"\)', wrapper, re.DOTALL)
blob = B.b85decode(m.group(1))
code = L.decompress(blob, format=L.FORMAT_RAW, filters=[{"id": L.FILTER_LZMA2}]).decode()

# Fix 1: Python 3.11 f-string compat — nested quotes in glob
code = code.replace(
    '.glob("fineweb_train_*.bin")',
    ".glob('fineweb_train_*.bin')"
)

# Fix 2: Python 3.11 f-string compat — category join
code = code.replace(
    'log(f"  {cat}: {", ".join(sorted(categories[cat]))}")',
    'log(f"  {cat}: " + ", ".join(sorted(categories[cat])))'
)

# Fix 3: 1-sqrt warmdown schedule (our novel contribution)
code = code.replace(
    "if frac>=1.-h.warmdown_frac:return max((1.-frac)/h.warmdown_frac,h.min_lr)",
    "if frac>=1.-h.warmdown_frac:\n\t\t\tt=(frac-(1.-h.warmdown_frac))/h.warmdown_frac\n\t\t\treturn max(1.-__import__('math').sqrt(t),h.min_lr)"
)

# Scan for any remaining nested f-string issues (Python 3.12+ syntax)
# Pattern: f"...{something("inner")}..." — replace inner " with '
import re as RE
# Fix read_text encoding
code = code.replace(
    "read_text(encoding='utf-8')",
    "read_text(encoding='utf-8')"
)

with open("train_run.py", "w") as f:
    f.write(code)

print(f"Patched: {len(code)} bytes")
print("  - Python 3.11 f-string compat")
print("  - 1-sqrt warmdown schedule")
PYEOF

# Verify syntax
python3 -c "import py_compile; py_compile.compile('train_run.py', doraise=True); print('Syntax OK')"

echo "=== Step 5: Download SP8192 data ==="
# Need the data download script from the repo
curl -sL "https://raw.githubusercontent.com/openai/parameter-golf/main/data/cached_challenge_fineweb.py" -o data_download.py
mkdir -p data/datasets data/tokenizers

# Download manifest from kevclark dataset repo
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('data', exist_ok=True)
path = hf_hub_download('kevclark/parameter-golf', 'datasets/manifest.json', repo_type='dataset')
shutil.copy(path, 'data/manifest.json')
print('SP8192 manifest ready')
"

# Download SP8192 data shards + tokenizer
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data_download.py --variant sp8192

echo "=== Step 6: Verify data ==="
ls data/datasets/fineweb10B_sp8192/ | head -5
ls data/tokenizers/fineweb_8192_bpe.*

echo "=== Step 7: Run training ==="
# torch.compile ENABLED (H100 sm_90 supports it)
# FA3 used natively (no SDPA fallback)
SEED=1337 RUN_ID=h100_v4_sqrt \
  torchrun --standalone --nproc_per_node=8 train_run.py

echo "=== DONE ==="
echo "Check logs/h100_v4_sqrt.txt for results"
echo "Look for: final_int6_sliding_window val_bpb"
