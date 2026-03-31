#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# GB10 single-GPU proxy for directional signal only (not submission-comparable).
# Keeps the same 4-arm chain:
# baseline -> turbomuon -> engramlite -> combo
#
# Approx "10% strength" defaults relative to the current smoke profile:
# - 1 seed
# - 220 steps (10% of 2200)
# - TRAIN_BATCH_TOKENS=81920 (~10% of 786432)
# - no warmdown

PROFILE=smoke \
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}" \
NPROC="${NPROC:-1}" \
SEEDS="${SEEDS:-444}" \
ITERATIONS="${ITERATIONS:-220}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-0}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-81920}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
EVAL_STRIDE="${EVAL_STRIDE:-128}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}" \
POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-1}" \
COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
"${SCRIPT_DIR}/run_ab_matrix.sh"
