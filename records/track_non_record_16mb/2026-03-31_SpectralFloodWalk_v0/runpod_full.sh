#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

sfw_run_profile full "${SEED}" \
  --train-tokens "${SFW_FULL_TRAIN_TOKENS:-262144}" \
  --val-tokens "${SFW_FULL_VAL_TOKENS:-65536}" \
  --train-steps "${SFW_FULL_TRAIN_STEPS:-400}" \
  --batch-size "${SFW_FULL_BATCH_SIZE:-64}" \
  --bank-capacity "${SFW_FULL_BANK_CAPACITY:-8192}" \
  --bank-min-entries "${SFW_FULL_BANK_MIN_ENTRIES:-512}" \
  --bank-writes-per-step "${SFW_FULL_BANK_WRITES_PER_STEP:-32}" \
  --warmup-steps "${SFW_FULL_WARMUP_STEPS:-20}" \
  --retrieval-dropout-start "${SFW_FULL_DROPOUT_START:-0.60}" \
  --retrieval-dropout-end "${SFW_FULL_DROPOUT_END:-0.05}" \
  --eval-samples "${SFW_FULL_EVAL_SAMPLES:-512}" \
  --report-every "${SFW_FULL_REPORT_EVERY:-10}" \
  "$@"
