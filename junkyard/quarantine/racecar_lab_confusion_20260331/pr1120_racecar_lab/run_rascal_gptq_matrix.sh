#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="analysis/pr1120_racecar_lab/copies/train_gpt_rascal_sota_local.py"
LOG_DIR="analysis/pr1120_racecar_lab/runs"
mkdir -p "${LOG_DIR}"

: "${NPROC_PER_NODE:=8}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${SEEDS:=42 300 444}"
: "${GPTQ_RESERVES:=9000 12000 14000}"
: "${GPTQ_CALIB_SAMPLES:=256}"
: "${GPTQ_CACHE_SEQS_PER_STEP:=1}"

# Control (PR1120 behavior)
for seed in ${SEEDS}; do
  run_id="R0_seed${seed}_nogptq"
  echo "[run] ${run_id}"
  SEED="${seed}" \
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
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
    2>&1 | tee "${LOG_DIR}/${run_id}.log"
done

# GPTQ reserve sweep
for reserve in ${GPTQ_RESERVES}; do
  for seed in ${SEEDS}; do
    run_id="R1_seed${seed}_gptq_reserve${reserve}"
    echo "[run] ${run_id}"
    SEED="${seed}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
    SKIP_GPTQ=0 \
    GPTQ_RESERVE_MS="${reserve}" \
    GPTQ_CALIB_SAMPLES="${GPTQ_CALIB_SAMPLES}" \
    GPTQ_INSTA_CACHE=1 \
    GPTQ_CACHE_SEQS_PER_STEP="${GPTQ_CACHE_SEQS_PER_STEP}" \
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
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
      2>&1 | tee "${LOG_DIR}/${run_id}.log"
  done
done
