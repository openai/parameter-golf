#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Spark queue-friendly defaults: one GPU, short wallclock, architecture-preserving mini lane.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-75}"
export RUN_TAG="${RUN_TAG:-SHROUD_JUNKYARD_MINI_SPARK}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-256}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-256}"
export ITERATIONS="${ITERATIONS:-28}"
export USE_CRAWLER="${USE_CRAWLER:-1}"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-2}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-1}"
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
export INST_DIM="${INST_DIM:-16}"

bash experiments/Shroud/profiles/run_junkyard_rat_mini_shroud.sh
