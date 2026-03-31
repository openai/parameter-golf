#!/usr/bin/env bash
# Full competitive run (600s cap, production eval + export). Use for leaderboard A/B.
# See docs/PLAN-h100-novel-budget.md
#
# Usage (repo root):
#   bash scripts/run_orchestrated_full_ab.sh baseline   # LEAKY_RELU_SLOPE=0
#   bash scripts/run_orchestrated_full_ab.sh leaky      # LEAKY_RELU_SLOPE=0.5
#
# Override RUN_ID, SEED, etc. in the environment before calling.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

variant="${1:-}"
if [[ "$variant" != "baseline" && "$variant" != "leaky" ]]; then
  echo "Usage: $0 baseline|leaky" >&2
  exit 1
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA required." >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-42}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export SMOKE_MODE="${SMOKE_MODE:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export SWA_ENABLED="${SWA_ENABLED:-1}"

if [[ "$variant" == "baseline" ]]; then
  export LEAKY_RELU_SLOPE="${LEAKY_RELU_SLOPE:-0}"
  export RUN_ID="${RUN_ID:-orchestrated_full_baseline}"
else
  export LEAKY_RELU_SLOPE="${LEAKY_RELU_SLOPE:-0.5}"
  export RUN_ID="${RUN_ID:-orchestrated_full_leaky05}"
fi

TRAIN="${ROOT}/records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py"

echo "FULL ($variant): RUN_ID=$RUN_ID SEED=$SEED LEAKY_RELU_SLOPE=$LEAKY_RELU_SLOPE SMOKE_MODE=$SMOKE_MODE"
echo "Python: $PYTHON"

if command -v torchrun >/dev/null 2>&1; then
  exec torchrun --standalone --nproc_per_node=8 "$TRAIN"
else
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAIN"
fi
