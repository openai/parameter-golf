#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_NET="${NCCL_NET:-Socket}"
export SEED="${SEED:-42}"
export RUN_ID="${RUN_ID:-smoke_1xh100}"
export SMOKE_TEST=1
export ITERATIONS="${ITERATIONS:-2}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-1}"
export VAL_LOSS_EVERY=0
export EVAL_FRACTION="${EVAL_FRACTION:-0.01}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export TTT_ENABLED="${TTT_ENABLED:-0}"
export JEPA_LITE_ENABLED=0
export AWQ_RESCale_ENABLED=0
export HADAMARD_PRECOND_ENABLED=0
export LAYERWISE_PRECISION_ENABLED=0

python -m py_compile train_gpt.py
bash ./preflight_env.sh

torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee smoke_1xh100.log

python parse_run_logs.py smoke_1xh100.log --smoke
python validate_submission_artifacts.py --log smoke_1xh100.log --smoke
