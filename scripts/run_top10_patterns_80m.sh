#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-80}"
export MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-900}"
export AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
export AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.395}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-top10}"

bash scripts/run_autogate_budget.sh \
  winner_locked \
  winner_ema_swa \
  winner_wd03 \
  winner_wd04 \
  winner_warm3500 \
  winner_lr18 \
  winner_wd03_ema \
  winner_mlp3
