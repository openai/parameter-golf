#!/usr/bin/env bash
set -euo pipefail

# Locked baseline launcher: exact Rascal record lane (no GPTQ).
# Usage:
#   bash scripts/run_rascal_8x_baseline_locked.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SRC="records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py"
COPY_DIR="analysis/pr1120_racecar_lab/copies"
RUN_DIR="analysis/pr1120_racecar_lab/runs_8x_econ"
TRAIN_COPY="${COPY_DIR}/train_gpt_rascal_sota_local.py"
mkdir -p "${COPY_DIR}" "${RUN_DIR}"

if [ ! -f "${SRC}" ]; then
  echo "FATAL: missing locked source ${SRC}"
  exit 1
fi
cp -f "${SRC}" "${TRAIN_COPY}"
echo "[bootstrap] copied ${SRC} -> ${TRAIN_COPY}"

: "${PYTHON_BIN:=python3}"
: "${NPROC_PER_NODE:=8}"
: "${SEED:=444}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${FA3_REQUIRED:=1}"

echo "[preflight] torch/cuda/gpu:"
"${PYTHON_BIN}" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

if "${PYTHON_BIN}" -c "from flash_attn_interface import flash_attn_func; print('FA3_OK')" >/tmp/rascal_fa3_check.txt 2>&1; then
  echo "[preflight] $(cat /tmp/rascal_fa3_check.txt)"
else
  echo "[preflight] flash_attn_interface import failed:"
  sed -n '1,4p' /tmp/rascal_fa3_check.txt || true
  if [ "${FA3_REQUIRED}" = "1" ]; then
    echo "FATAL: FA3 required for competitive baseline speed."
    exit 1
  fi
fi

LOG="${RUN_DIR}/baseline_seed${SEED}.log"
echo "[run] baseline_seed${SEED} (SKIP_GPTQ=1)"
env \
  SEED="${SEED}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
  SKIP_GPTQ=1 \
  LOADER_MODE=coprime \
  COPRIME_MAX_LOADED_SHARDS=1 \
  COPRIME_SHARDS_PER_BATCH=1 \
  COPRIME_SHARD_HOLD_STEPS=64 \
  XSA_LAST_N=11 \
  BIGRAM_VOCAB_SIZE=2048 \
  BIGRAM_DIM=128 \
  ROPE_DIMS=16 \
  SWA_EVERY=50 \
  NGRAM_EVAL_ORDER=0 \
  MTP_NUM_HEADS=0 \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_COPY}" \
  2>&1 | tee "${LOG}"

echo "[done] ${LOG}"
grep -nE "step:500/|step:1000/|step:1500/|step:2000/|step:2500/|step:6500|stopping_early|final_sliding_window_exact" "${LOG}" | tail -n 20 || true

