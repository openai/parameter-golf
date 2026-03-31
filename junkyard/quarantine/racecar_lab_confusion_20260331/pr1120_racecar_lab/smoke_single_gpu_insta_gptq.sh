#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="analysis/pr1120_racecar_lab/copies/train_gpt_rascal_sota_local.py"
OUT_DIR="analysis/pr1120_racecar_lab/smoke_runs"
mkdir -p "${OUT_DIR}"

: "${SEED:=444}"
: "${NPROC_PER_NODE:=1}"
: "${MAX_WALLCLOCK_SECONDS:=75}"
: "${GPTQ_RESERVE_MS:=8000}"
: "${GPTQ_CALIB_SAMPLES:=64}"
: "${GPTQ_CACHE_SEQS_PER_STEP:=1}"
: "${PYTHON_BIN:=python3}"
: "${TRAIN_BATCH_TOKENS:=131072}"
: "${VAL_BATCH_SIZE:=65536}"
: "${WARMUP_STEPS:=0}"
: "${COMPILE_ENABLED:=0}"
: "${SMOKE_SKIP_VAL:=1}"
: "${SMOKE_SKIP_QUANT_EVAL:=1}"

COMMON_ENV=(
  "SEED=${SEED}"
  "DATA_PATH=${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
  "TOKENIZER_PATH=${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
  "ITERATIONS=20000"
  "VAL_LOSS_EVERY=0"
  "TRAIN_LOG_EVERY=50"
  "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS}"
  "VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
  "WARMUP_STEPS=${WARMUP_STEPS}"
  "COMPILE_ENABLED=${COMPILE_ENABLED}"
  "COMPILE_FULLGRAPH=0"
  "SMOKE_SKIP_VAL=${SMOKE_SKIP_VAL}"
  "SMOKE_SKIP_QUANT_EVAL=${SMOKE_SKIP_QUANT_EVAL}"
  "SKIP_FINAL_EVAL=1"
  "POST_EMA_DIAGNOSTIC=0"
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
  local log="${OUT_DIR}/${name}.log"
  echo "[smoke] case=${name}"
  env "${COMMON_ENV[@]}" "$@" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
    > "${log}" 2>&1

  local last_train last_stop last_gptq last_cache_collect last_cache_used
  last_train=$(rg -n "step:[0-9]+/[0-9]+ train_loss:.*step_avg:" "${log}" | tail -n 1 || true)
  last_stop=$(rg -n "stopping_early:|step:[0-9]+/[0-9]+ val_loss:" "${log}" | tail -n 1 || true)
  last_gptq=$(rg -n "gptq:calibrated|gptq:SKIPPED" "${log}" | tail -n 1 || true)
  last_cache_collect=$(rg -n "gptq:insta_cache_collected" "${log}" | tail -n 1 || true)
  last_cache_used=$(rg -n "gptq:insta_cache_used" "${log}" | tail -n 1 || true)

  echo "  train: ${last_train:-N/A}"
  echo "  stop : ${last_stop:-N/A}"
  echo "  gptq : ${last_gptq:-N/A}"
  echo "  cache_collected: ${last_cache_collect:-N/A}"
  echo "  cache_used     : ${last_cache_used:-N/A}"
}

run_case "A_nogptq" "SKIP_GPTQ=1"
run_case "B_gptq_stream" \
  "SKIP_GPTQ=0" \
  "GPTQ_RESERVE_MS=${GPTQ_RESERVE_MS}" \
  "GPTQ_CALIB_SAMPLES=${GPTQ_CALIB_SAMPLES}" \
  "GPTQ_INSTA_CACHE=0"
run_case "C_gptq_insta" \
  "SKIP_GPTQ=0" \
  "GPTQ_RESERVE_MS=${GPTQ_RESERVE_MS}" \
  "GPTQ_CALIB_SAMPLES=${GPTQ_CALIB_SAMPLES}" \
  "GPTQ_INSTA_CACHE=1" \
  "GPTQ_CACHE_SEQS_PER_STEP=${GPTQ_CACHE_SEQS_PER_STEP}"

echo "[smoke] done. logs in ${OUT_DIR}"
