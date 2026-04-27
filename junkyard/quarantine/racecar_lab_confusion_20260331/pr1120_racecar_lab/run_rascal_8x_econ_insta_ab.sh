#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="analysis/pr1120_racecar_lab/copies/train_gpt_rascal_sota_local.py"
LOG_DIR="analysis/pr1120_racecar_lab/runs_8x_econ"
mkdir -p "${LOG_DIR}"

: "${NPROC_PER_NODE:=8}"
: "${SEED:=444}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${GPTQ_RESERVE_MS:=9000}"
: "${GPTQ_CALIB_SAMPLES:=256}"
: "${GPTQ_CACHE_SEQS_PER_STEP:=1}"

common_env=(
  "SEED=${SEED}"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
  "SKIP_GPTQ=0"
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
  local log="${LOG_DIR}/${name}.log"
  echo "[8x-econ] ${name}"
  env "${common_env[@]}" "$@" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
    2>&1 | tee "${log}"
}

run_case "stream_seed${SEED}" "GPTQ_INSTA_CACHE=0"
run_case "insta_seed${SEED}" "GPTQ_INSTA_CACHE=1" "GPTQ_CACHE_SEQS_PER_STEP=${GPTQ_CACHE_SEQS_PER_STEP}"

echo "[8x-econ] done: ${LOG_DIR}"
