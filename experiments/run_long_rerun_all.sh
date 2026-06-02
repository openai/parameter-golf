#!/bin/bash
set -euo pipefail

# Long rerun launcher for all 10 experiments.
#
# Defaults are tuned for an overnight rerun and can be overridden with env vars:
#   LONG_WALLCLOCK_SECONDS (default 1800)
#   LONG_WARMUP_STEPS      (default 20)
#   LONG_MAX_VAL_TOKENS    (default 0, full eval)
#
# Usage:
#   bash experiments/run_long_rerun_all.sh
#   LONG_WALLCLOCK_SECONDS=2400 bash experiments/run_long_rerun_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"

# Select Python interpreter: explicit override > project .venv > system python3.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
else
  PYTHON_BIN="python3"
fi
export PYTHON_BIN

LONG_WALLCLOCK_SECONDS="${LONG_WALLCLOCK_SECONDS:-1800}"
LONG_WARMUP_STEPS="${LONG_WARMUP_STEPS:-20}"
LONG_MAX_VAL_TOKENS="${LONG_MAX_VAL_TOKENS:-0}"

mkdir -p "$LOGS_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SESSION_LOG="$LOGS_DIR/long_rerun_${RUN_TS}.log"

cd "$PROJECT_ROOT"

{
  echo "=== Long Rerun: All Experiments ==="
  echo "Started: $(date)"
  echo "Project root: $PROJECT_ROOT"
  echo "Python: $PYTHON_BIN"
  echo "LONG_WALLCLOCK_SECONDS=$LONG_WALLCLOCK_SECONDS"
  echo "LONG_WARMUP_STEPS=$LONG_WARMUP_STEPS"
  echo "LONG_MAX_VAL_TOKENS=$LONG_MAX_VAL_TOKENS"
  echo ""

  # --rerun ensures every experiment is re-executed even when exp_E*.txt exists.
  env MAX_WALLCLOCK_SECONDS="$LONG_WALLCLOCK_SECONDS" \
      WARMUP_STEPS="$LONG_WARMUP_STEPS" \
      MAX_VAL_TOKENS="$LONG_MAX_VAL_TOKENS" \
      bash "$SCRIPT_DIR/run_m4_experiments.sh" --rerun

  echo ""
  echo "=== Parsing results ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_results.py"

  echo ""
  echo "Completed: $(date)"
  echo "Summary: $SCRIPT_DIR/results_summary.md"
} 2>&1 | tee "$SESSION_LOG"

echo ""
echo "Session log: $SESSION_LOG"
echo "Summary: $SCRIPT_DIR/results_summary.md"
