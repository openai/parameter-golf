#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-120}"
MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-900}"
AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.388}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-bestbet}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

bash scripts/run_autogate_budget.sh \
  wd04_locked \
  winner_wd04 \
  hunt_11l_fullheads \
  t5_11l_fullheads \
  t5_control_wd04 \
  t5_stack_leaky \
  t5_stack_fullheads_toplr \
  hunt_11l_layerwise \
  hunt_11l_mtp \
  hunt_shared11_mid
