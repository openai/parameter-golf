#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
POP_RUNNER="${SCRIPT_DIR}/../run_pop_test.sh"

SEED="${SEED:-1337}"
POP_SECONDS="${POP_SECONDS:-170}"

run_case() {
    local label="$1"
    shift
    echo
    echo "============================================================"
    echo "  Triton Delta Sequence: ${label}"
    echo "  seed=${SEED} seconds=${POP_SECONDS}"
    echo "============================================================"
    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS="${POP_SECONDS}" \
        "$@" \
        bash "${POP_RUNNER}"
}

# ~6 x 170s = ~17 minutes of train budget, leaving room for startup/validation overhead.
run_case "delta00_base" \
    MLP_KERNEL_MODE=triton_act \
    ATTN_SCALE_INIT=1.0 \
    MLP_SCALE_INIT=1.0 \
    RESID_MIX_X_INIT=1.0 \
    RESID_MIX_X0_INIT=0.0 \
    LN_SCALE=1

run_case "delta01_mlp_scale_098" \
    MLP_KERNEL_MODE=triton_act \
    MLP_SCALE_INIT=0.98

run_case "delta02_mlp_scale_102" \
    MLP_KERNEL_MODE=triton_act \
    MLP_SCALE_INIT=1.02

run_case "delta03_attn_scale_098" \
    MLP_KERNEL_MODE=triton_act \
    ATTN_SCALE_INIT=0.98

run_case "delta04_attn_scale_102" \
    MLP_KERNEL_MODE=triton_act \
    ATTN_SCALE_INIT=1.02

run_case "delta05_residmix_098_002" \
    MLP_KERNEL_MODE=triton_act \
    RESID_MIX_X_INIT=0.98 \
    RESID_MIX_X0_INIT=0.02

echo
echo "============================================================"
echo "  Triton Delta Sequence Complete"
echo "============================================================"
