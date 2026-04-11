#!/usr/bin/env bash
# Budget smoke: 8×GPU, short wallclock, no val during train, no export/sliding eval.
# See docs/PLAN-h100-novel-budget.md — metrics from this run are NOT valid for leaderboard.
#
# Usage (repo root, Linux + CUDA):
#   bash scripts/smoke_orchestrated_8xh100.sh
# Optional:
#   LEAKY_RELU_SLOPE=0.5 bash scripts/smoke_orchestrated_8xh100.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA required (RunPod 8×H100, etc.)." >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-orchestrated_smoke}"
export SEED="${SEED:-42}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"
export ITERATIONS="${ITERATIONS:-5000}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export SMOKE_MODE="${SMOKE_MODE:-1}"
# Faster smoke: no SWA collection/apply
export SWA_ENABLED="${SWA_ENABLED:-0}"

TRAIN="${ROOT}/records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py"

echo "SMOKE: RUN_ID=$RUN_ID MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS LEAKY_RELU_SLOPE=${LEAKY_RELU_SLOPE:-0}"
echo "Python: $PYTHON"

if command -v torchrun >/dev/null 2>&1; then
  exec torchrun --standalone --nproc_per_node=8 "$TRAIN"
else
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAIN"
fi
