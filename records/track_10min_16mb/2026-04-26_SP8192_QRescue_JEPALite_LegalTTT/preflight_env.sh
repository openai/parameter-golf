#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"

echo "record_dir=${SCRIPT_DIR}"
echo "repo_root=${REPO_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "VOCAB_SIZE=${VOCAB_SIZE}"

python - <<'PY'
import os, sys, subprocess, torch
print('python', sys.version)
print('torch', torch.__version__)
print('torch_cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
for mod in ('brotli', 'sentencepiece', 'numpy', 'triton'):
    try:
        __import__(mod)
        print(mod, 'available')
    except Exception as e:
        raise SystemExit(f'{mod} unavailable: {e}')
try:
    from flash_attn_interface import flash_attn_func  # noqa: F401
    print('flash_attn_interface', 'available')
except Exception as e:
    raise SystemExit(f'flash_attn_interface unavailable: {e}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'gpu[{i}]', p.name, 'sm', f'{p.major}.{p.minor}', 'mem_GB', round(p.total_memory/1e9, 2))
print('nvidia-smi:')
try:
    subprocess.run(['nvidia-smi'], check=False)
except FileNotFoundError:
    print('nvidia-smi not found')
PY

python - <<'PY'
import os
from pathlib import Path

data_dir = Path(os.environ['DATA_DIR'])
vocab = int(os.environ.get('VOCAB_SIZE', '8192'))
if vocab != 8192:
    raise SystemExit(f'Expected SP8192 only; got VOCAB_SIZE={vocab}')

dataset_dir = data_dir / 'datasets' / 'fineweb10B_sp8192'
tokenizer = data_dir / 'tokenizers' / 'fineweb_8192_bpe.model'
train = sorted(dataset_dir.glob('fineweb_train_*.bin'))
val = sorted(dataset_dir.glob('fineweb_val_*.bin'))

print('dataset_dir', dataset_dir)
print('tokenizer', tokenizer)
print('train_shards_found', len(train))
print('val_shards_found', len(val))

missing = []
if not tokenizer.is_file():
    missing.append(str(tokenizer))
if not train:
    missing.append(str(dataset_dir / 'fineweb_train_*.bin'))
if not val:
    missing.append(str(dataset_dir / 'fineweb_val_*.bin'))
if missing:
    print('Missing SP8192 cached dataset files:')
    for item in missing:
        print('  ', item)
    print('Create them from the repo root with:')
    print('  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192')
    raise SystemExit(2)

print('SP8192 dataset/tokenizer check PASS')
PY

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
q = torch.randn(1, 8, 4, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 4, 4, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 4, 4, 64, device='cuda', dtype=torch.bfloat16)
try:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    print('enable_gqa supported', tuple(y.shape))
except Exception as e:
    raise SystemExit(f'enable_gqa check failed: {e}')
PY
