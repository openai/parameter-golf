#!/usr/bin/env bash
set -euo pipefail

# Runs only the best experiment behind PR 2069:
#   best4x_ttt_disabled_qk525
#
# Expected setup:
#   PARAMETER_GOLF_REF=codex/frontier-deadline-experiments
#   scripts/setup_runpod_parameter_golf_deadline.sh

PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-/workspace/parameter-golf}"
PG_REF="${PG_REF:-codex/frontier-deadline-experiments}"

cd "$PARAMETER_GOLF_DIR"

if [ -f /workspace/pg_deadline_env.sh ]; then
  # shellcheck disable=SC1091
  source /workspace/pg_deadline_env.sh
fi

git fetch origin "$PG_REF" || true
if git rev-parse --verify "origin/$PG_REF" >/dev/null 2>&1; then
  git switch "$PG_REF"
  git pull --ff-only origin "$PG_REF" || true
fi

export PG_QUEUE="${PG_QUEUE:-planning/four_h100_best_experiment_queue.json}"
export PG_GPUS="${PG_GPUS:-8}"
export PG_ONLY="${PG_ONLY:-best4x_ttt_disabled_qk525}"
export PG_STATUS="${PG_STATUS:-ready}"
export PG_MAX_EXPERIMENTS="${PG_MAX_EXPERIMENTS:-1}"
export PG_OUTPUT_ROOT="${PG_OUTPUT_ROOT:-records/pr2069_best_8xh100}"
export PG_EXPORT_ROOT="${PG_EXPORT_ROOT:-records/pr2069_best_8xh100_exports}"
export PG_LOG_ROOT="${PG_LOG_ROOT:-records/pr2069_best_8xh100_logs}"
export PG_TIMEOUT_SECONDS="${PG_TIMEOUT_SECONDS:-900}"
export PG_STOP_ON_FAILURE="${PG_STOP_ON_FAILURE:-1}"
export PG_AUTO_TUNE="${PG_AUTO_TUNE:-0}"

echo "Running PR 2069 best candidate on ${PG_GPUS} GPUs: ${PG_ONLY}"
exec scripts/run_deadline_experiment_batch.sh
