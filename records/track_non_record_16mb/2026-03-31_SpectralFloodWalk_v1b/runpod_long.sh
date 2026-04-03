#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

SFW_FULL_TRAIN_STEPS="${SFW_FULL_TRAIN_STEPS:-${SFW_TRAIN_STEPS:-2000}}" \
SFW_FULL_EVAL_BATCHES="${SFW_FULL_EVAL_BATCHES:-${SFW_EVAL_BATCHES:-512}}" \
SFW_FULL_REPORT_EVERY="${SFW_FULL_REPORT_EVERY:-${SFW_REPORT_EVERY:-100}}" \
"$(dirname "${BASH_SOURCE[0]}")/runpod_full.sh" "$@"
