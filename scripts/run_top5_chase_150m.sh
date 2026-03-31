#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-150}"
export MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-1200}"
export AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
export AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.390}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-top5}"

bash scripts/run_autogate_budget.sh \
  t5_control_wd04 \
  t5_11l_fullheads \
  t5_stack_relu \
  t5_stack_leaky \
  t5_stack_leaky_w4000 \
  t5_stack_leaky_noema \
  t5_stack_relu_noema \
  t5_stack_leaky_zloss \
  t5_stack_leaky_layerwise \
  t5_stack_leaky_untied \
  t5_stack_leaky_shared \
  t5_stack_fullheads_toplr
