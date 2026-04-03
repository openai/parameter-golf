#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Triton Pop Test"
echo "  wallclock=${MAX_WALLCLOCK_SECONDS:-180}s"
echo "  kernel=${MLP_KERNEL_MODE:-triton_act}"
echo "  mlp_scale=${MLP_SCALE_INIT:-1.0} attn_scale=${ATTN_SCALE_INIT:-1.0}"
echo "  resid_mix=(${RESID_MIX_X_INIT:-1.0},${RESID_MIX_X0_INIT:-0.0}) ln_scale=${LN_SCALE:-1}"
echo "============================================"

exec env \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}" \
    WARMUP_STEPS="${WARMUP_STEPS:-0}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}" \
    TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}" \
    POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}" \
    SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}" \
    MLP_KERNEL_MODE="${MLP_KERNEL_MODE:-triton_act}" \
    COMPILE_MODE="${COMPILE_MODE:-}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}" \
    ATTN_SCALE_INIT="${ATTN_SCALE_INIT:-1.0}" \
    MLP_SCALE_INIT="${MLP_SCALE_INIT:-1.0}" \
    RESID_MIX_X_INIT="${RESID_MIX_X_INIT:-1.0}" \
    RESID_MIX_X0_INIT="${RESID_MIX_X0_INIT:-0.0}" \
    LN_SCALE="${LN_SCALE:-1}" \
    bash "${SCRIPT_DIR}/../run.sh"
