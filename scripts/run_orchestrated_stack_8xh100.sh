#!/usr/bin/env bash
# Run the orchestrated 10L Int5 competitive stack on 8×GPU (intended: 8×H100 SXM, Linux + CUDA).
#
# Prerequisites:
#   source .venv/bin/activate
#   uv pip install -r requirements.txt
#   uv pip install zstandard
#
# If `torchrun` is not on PATH (common when using a venv), this script falls back to:
#   python -m torch.distributed.run
#
# Local macOS: PyTorch CUDA training is not supported — use train_gpt_mlx.py (see root README).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer project venv Python so `python -m torch.distributed.run` finds torch.
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "ERROR: PyTorch is not installed for: $PYTHON" >&2
  echo "Install with:  source .venv/bin/activate && uv pip install -r requirements.txt" >&2
  exit 1
fi

if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA is not available. This script is for multi-GPU Linux (e.g. Runpod 8×H100)." >&2
  echo "On Apple Silicon, use:  python train_gpt_mlx.py  (see README Getting Started)" >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-orchestrated_10l_int5}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export SEED="${SEED:-42}"

TRAIN="${ROOT}/records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py"

echo "RUN_ID=$RUN_ID DATA_PATH=$DATA_PATH"
echo "Using Python: $PYTHON"

if command -v torchrun >/dev/null 2>&1; then
  exec torchrun --standalone --nproc_per_node=8 "$TRAIN"
else
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAIN"
fi
