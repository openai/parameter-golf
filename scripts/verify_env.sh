#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

python - <<'PY'
import datasets
import sentencepiece
import torch
import mlx.core as mx

print("torch", torch.__version__)
print("datasets", datasets.__version__)
print("sentencepiece", sentencepiece.__version__)
print("mlx_sum", mx.sum(mx.array([1.0, 2.0, 3.0])).item())
PY

python data/cached_challenge_fineweb.py --help >/dev/null

echo
echo "Environment verification passed."
