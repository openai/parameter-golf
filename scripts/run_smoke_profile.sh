#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:-base10l}"
shift || true

export RUN_ID="${RUN_ID:-${PROFILE}_smoke}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-90}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"

bash scripts/run_remote_profile.sh "$PROFILE" "$@"
