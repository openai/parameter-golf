#!/usr/bin/env bash
set -euo pipefail

cd /workspace/parameter-golf

git fetch origin codex/frontier-deadline-experiments
git switch codex/frontier-deadline-experiments
git pull --ff-only

source /workspace/pg_deadline_env.sh

ls scripts/run_4x_best_experiment_batch.sh

MAX_WALLCLOCK_SECONDS=520 \
PG_TIMEOUT_SECONDS=700 \
scripts/run_4x_best_experiment_batch.sh
