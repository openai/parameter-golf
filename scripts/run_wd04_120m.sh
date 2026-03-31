#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-120}"
export MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-1800}"
export AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
export AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.388}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-wd04pack}"

bash scripts/run_autogate_budget.sh \
  wd04_locked \
  wd04_warm3500 \
  wd04_warm4000 \
  wd04_lr18 \
  wd04_lr19 \
  wd04_scalar18 \
  wd04_tied28 \
  wd04_ema_swa \
  wd04_warm3500_lr18 \
  wd04_zloss
