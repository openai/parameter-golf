#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/environment/vars.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/environment/vars.env"
  set +a
fi

: "${SEED:=1337}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${NPROC_PER_NODE:=8}"
: "${PYTHON_BIN:=python3}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-xwing_red_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
