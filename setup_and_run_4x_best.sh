#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PARAMETER_GOLF_REPO_URL="https://github.com/tenet-diver/parameter-golf.git"
export PARAMETER_GOLF_REF="codex/frontier-deadline-experiments"

curl -fsSL \
  https://raw.githubusercontent.com/tenet-diver/parameter-golf/codex/frontier-deadline-experiments/scripts/setup_runpod_parameter_golf_deadline.sh \
  -o /workspace/setup_pg.sh

bash /workspace/setup_pg.sh

cd /workspace/parameter-golf

git fetch origin codex/frontier-deadline-experiments
git switch codex/frontier-deadline-experiments
git pull --ff-only

source /workspace/pg_deadline_env.sh

tmux new -s pg4 'scripts/run_4x_best_experiment_batch.sh'
