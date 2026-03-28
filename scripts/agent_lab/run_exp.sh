#!/usr/bin/env bash
# Run agent_lab training from repo root with consistent defaults.
# Usage (from repo root):
#   RUN_ID=my_run ./scripts/agent_lab/run_exp.sh > agent_lab/run.log 2>&1
# Override any env var (DATA_PATH, SEED, MAX_WALLCLOCK_SECONDS, NUM_KV_HEADS, …) as usual.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export RUN_ID="${RUN_ID:-agent_lab_$(date +%Y%m%d_%H%M%S)}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export SEED="${SEED:-1337}"

NPROC="${NPROC:-1}"

exec torchrun --standalone --nproc_per_node="${NPROC}" agent_lab/train_gpt.py "$@"
