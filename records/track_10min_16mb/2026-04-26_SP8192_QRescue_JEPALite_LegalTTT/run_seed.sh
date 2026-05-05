#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: ./run_seed.sh <seed>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

SEED="$1"
RUN_ID="submission_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs artifacts

export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
export NCCL_NET="${NCCL_NET:-Socket}"
export SEED="${SEED}"
export RUN_ID="${RUN_ID}"
export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_EPOCHS="${TTT_EPOCHS:-4}"
export TTT_LR="${TTT_LR:-0.005}"
export TTT_LORA_ENABLED="${TTT_LORA_ENABLED:-0}"
export QRESCUE_ENABLED="${QRESCUE_ENABLED:-1}"
export HESSIAN_LAYERWISE_CLIP="${HESSIAN_LAYERWISE_CLIP:-1}"
export JEPA_LITE_ENABLED="${JEPA_LITE_ENABLED:-1}"
export COMPRESSOR="${COMPRESSOR:-pergroup}"
export LQER_ENABLED="${LQER_ENABLED:-1}"
export LQER_BUDGET_BYTES="${LQER_BUDGET_BYTES:-140000}"
export LQER_MAX_RANK="${LQER_MAX_RANK:-4}"
export LQER_TARGETS="${LQER_TARGETS:-loop_mlp_proj,late_mlp_proj,attn_proj}"
export AWQ_RESCale_ENABLED="${AWQ_RESCale_ENABLED:-0}"
export HADAMARD_PRECOND_ENABLED="${HADAMARD_PRECOND_ENABLED:-0}"
export LAYERWISE_PRECISION_ENABLED="${LAYERWISE_PRECISION_ENABLED:-0}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

python -m py_compile train_gpt.py
bash ./preflight_env.sh | tee "logs/${RUN_ID}_preflight.log"

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "logs/${RUN_ID}.log"
cp "logs/${RUN_ID}.log" "train_seed${SEED}.log"

python parse_run_logs.py "logs/${RUN_ID}.log" --json "logs/${RUN_ID}.summary.json"
python validate_submission_artifacts.py --log "logs/${RUN_ID}.log" --summary "logs/${RUN_ID}.summary.json"
