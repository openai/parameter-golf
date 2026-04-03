#!/usr/bin/env bash
set -euo pipefail

# Quick 8x launcher for Rascal baseline speed or GPTQ stream vs insta-cache.
# Usage:
#   bash scripts/run_pr1120_8x_quick_ab.sh
# Optional overrides:
#   RUN_MODE=baseline|ab SEED=444 NPROC_PER_NODE=8 GPTQ_RESERVE_MS=9000 GPTQ_CALIB_SAMPLES=256 bash scripts/run_pr1120_8x_quick_ab.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COPY_DIR="analysis/pr1120_racecar_lab/copies"
RUN_DIR="analysis/pr1120_racecar_lab/runs_8x_econ"
TRAIN_COPY="${COPY_DIR}/train_gpt_rascal_sota_local.py"
BASELINE_SRC="records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py"
AB_SRC="scripts/train_gpt_rascal_insta_cache.py"
mkdir -p "${COPY_DIR}" "${RUN_DIR}"

# Run mode:
#   baseline -> speed lane (defaults SKIP_GPTQ=1, one run)
#   ab       -> GPTQ A/B lane (defaults SKIP_GPTQ=0, stream + insta if supported)
: "${RUN_MODE:=baseline}"
if [ "${RUN_MODE}" != "baseline" ] && [ "${RUN_MODE}" != "ab" ]; then
  echo "FATAL: RUN_MODE must be baseline or ab (got: ${RUN_MODE})"
  exit 1
fi

# Always refresh trainer copy from explicit source to avoid stale/mixed lanes.
if [ "${RUN_MODE}" = "baseline" ]; then
  if [ ! -f "${BASELINE_SRC}" ]; then
    echo "FATAL: missing locked baseline source: ${BASELINE_SRC}"
    exit 1
  fi
  cp -f "${BASELINE_SRC}" "${TRAIN_COPY}"
  echo "[bootstrap] copied locked baseline ${BASELINE_SRC} -> ${TRAIN_COPY}"
else
  if [ ! -f "${AB_SRC}" ]; then
    echo "FATAL: missing AB source with insta-cache hook: ${AB_SRC}"
    exit 1
  fi
  cp -f "${AB_SRC}" "${TRAIN_COPY}"
  echo "[bootstrap] copied AB source ${AB_SRC} -> ${TRAIN_COPY}"
fi

: "${PYTHON_BIN:=python3}"
: "${NPROC_PER_NODE:=8}"
: "${SEED:=444}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${FA3_REQUIRED:=1}"
: "${GPTQ_RESERVE_MS:=9000}"
: "${GPTQ_CALIB_SAMPLES:=256}"
: "${GPTQ_CACHE_SEQS_PER_STEP:=1}"

echo "[preflight] torch/cuda/gpu:"
"${PYTHON_BIN}" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

if "${PYTHON_BIN}" -c "from flash_attn_interface import flash_attn_func; print('FA3_OK')" >/tmp/pr1120_fa3_check.txt 2>&1; then
  echo "[preflight] $(cat /tmp/pr1120_fa3_check.txt)"
else
  echo "[preflight] flash_attn_interface import failed (likely no FA3)."
  echo "[preflight] detail:"
  sed -n '1,3p' /tmp/pr1120_fa3_check.txt || true
  if [ "${FA3_REQUIRED}" = "1" ]; then
    echo "FATAL: FA3 is required for competitive speed. Re-run with FA3 installed, or set FA3_REQUIRED=0 to force-run."
    exit 1
  fi
fi

: "${SKIP_GPTQ:=}"
if [ -z "${SKIP_GPTQ}" ]; then
  if [ "${RUN_MODE}" = "baseline" ]; then
    SKIP_GPTQ=1
  else
    SKIP_GPTQ=0
  fi
fi
echo "[mode] RUN_MODE=${RUN_MODE} SKIP_GPTQ=${SKIP_GPTQ} NPROC_PER_NODE=${NPROC_PER_NODE} SEED=${SEED}"

COMMON_ENV=(
  "SEED=${SEED}"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
  "SKIP_GPTQ=${SKIP_GPTQ}"
  "GPTQ_RESERVE_MS=${GPTQ_RESERVE_MS}"
  "GPTQ_CALIB_SAMPLES=${GPTQ_CALIB_SAMPLES}"
  "LOADER_MODE=coprime"
  "COPRIME_MAX_LOADED_SHARDS=1"
  "COPRIME_SHARDS_PER_BATCH=1"
  "COPRIME_SHARD_HOLD_STEPS=64"
  "XSA_LAST_N=11"
  "BIGRAM_VOCAB_SIZE=2048"
  "BIGRAM_DIM=128"
  "ROPE_DIMS=16"
  "SWA_EVERY=50"
  "NGRAM_EVAL_ORDER=0"
  "MTP_NUM_HEADS=0"
)

run_case() {
  local name="$1"
  shift
  local log="${RUN_DIR}/${name}.log"
  echo "[run] ${name}"
  env "${COMMON_ENV[@]}" "$@" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_COPY}" \
    2>&1 | tee "${log}"
}

if [ "${RUN_MODE}" = "baseline" ]; then
  run_case "baseline_seed${SEED}" "GPTQ_INSTA_CACHE=0"
elif grep -q "GPTQ_INSTA_CACHE" "${TRAIN_COPY}"; then
  run_case "stream_seed${SEED}" "GPTQ_INSTA_CACHE=0"
  run_case "insta_seed${SEED}" "GPTQ_INSTA_CACHE=1" "GPTQ_CACHE_SEQS_PER_STEP=${GPTQ_CACHE_SEQS_PER_STEP}"
else
  echo "[info] trainer has no GPTQ_INSTA_CACHE hook; running stream-only"
  run_case "stream_seed${SEED}" "GPTQ_INSTA_CACHE=0"
fi

echo "[done] logs: ${RUN_DIR}"
for f in "${RUN_DIR}"/*.log; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  grep -nE "step:6500|stopping_early|gptq:calibrated|gptq:insta_cache|final_sliding_window_exact" "$f" | tail -n 20 || true
done
