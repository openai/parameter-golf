#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-120}"
export MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-1500}"
export AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
export AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.385}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-archhunt}"

bash scripts/run_autogate_budget.sh \
  hunt_locked_best \
  hunt_11l_wd04 \
  hunt_11l_wd04_warm4000 \
  hunt_11l_mlp3_leaky \
  hunt_11l_mlp3_leaky_ema \
  hunt_11l_layerwise \
  hunt_11l_fullheads \
  hunt_11l_untied \
  hunt_shared11_mid \
  hunt_11l_mtp
