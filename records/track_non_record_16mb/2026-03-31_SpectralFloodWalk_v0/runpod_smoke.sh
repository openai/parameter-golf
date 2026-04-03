#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

sfw_run_profile smoke "${SEED}" \
  --train-tokens "${SFW_SMOKE_TRAIN_TOKENS:-65536}" \
  --val-tokens "${SFW_SMOKE_VAL_TOKENS:-16384}" \
  --train-steps "${SFW_SMOKE_TRAIN_STEPS:-32}" \
  --batch-size "${SFW_SMOKE_BATCH_SIZE:-16}" \
  --bank-capacity "${SFW_SMOKE_BANK_CAPACITY:-1024}" \
  --bank-min-entries "${SFW_SMOKE_BANK_MIN_ENTRIES:-64}" \
  --bank-writes-per-step "${SFW_SMOKE_BANK_WRITES_PER_STEP:-8}" \
  --warmup-steps "${SFW_SMOKE_WARMUP_STEPS:-4}" \
  --retrieval-dropout-start "${SFW_SMOKE_DROPOUT_START:-0.60}" \
  --retrieval-dropout-end "${SFW_SMOKE_DROPOUT_END:-0.10}" \
  --eval-samples "${SFW_SMOKE_EVAL_SAMPLES:-128}" \
  --report-every "${SFW_SMOKE_REPORT_EVERY:-4}" \
  "$@"
