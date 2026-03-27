#!/usr/bin/env bash
# Three seeds for README / statistical evidence (same config as run_comp_8xh100_10min.sh).
# Usage (8×H100 pod, repo root):
#   chmod +x run_leaderboard_3seed.sh
#   ./run_leaderboard_3seed.sh
# First run downloads data; set SKIP_DATA_PREP=1 on reruns if data already exists.
set -euo pipefail
cd "$(dirname "$0")"

export RUN_ID_PREFIX="${RUN_ID_PREFIX:-pg_leaderboard}"
export SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"

if [[ "${SKIP_DATA_PREP}" != "1" ]]; then
  TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
  echo "[data] cached_challenge_fineweb sp1024 train_shards=${TRAIN_SHARDS}"
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
  export SKIP_DATA_PREP=1
fi

for SEED in 42 1337 2024; do
  echo "========== SEED=${SEED} RUN_ID=${RUN_ID_PREFIX}_seed${SEED} =========="
  SEED="${SEED}" RUN_ID="${RUN_ID_PREFIX}_seed${SEED}" ./run_comp_8xh100_10min.sh
done

echo "Done. Collect logs/${RUN_ID_PREFIX}_seed*.txt for the PR."
