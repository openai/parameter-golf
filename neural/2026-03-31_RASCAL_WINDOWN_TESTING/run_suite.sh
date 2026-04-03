#!/usr/bin/env bash
# run_suite.sh — 4-arm Legal Window Strategy Gate
# RASCAL_WINDOWN_TESTING
#
# Arms (all: 1-GPU, MAX_WALLCLOCK_SECONDS=120, seed=444):
#   CTRL-00   : no eval-time adaptation
#   SLOT-01   : legal context-only SLOT (8 steps, lr=0.005)
#   SCALE-02  : Score-first Scale TTT (attn_scale+mlp_scale, lr=1e-4, 1 epoch/chunk)
#   SLOT+SCALE-03 : both combined
#
# Usage: bash neural/2026-03-31_RASCAL_WINDOWN_TESTING/run_suite.sh
#
# Results land in: neural/2026-03-31_RASCAL_WINDOWN_TESTING/suite_<arm>.log
# Read the final_sliding_window*_exact lines to compare arms.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${SCRIPT_DIR}/train_gpt.py"

SEED="${SEED:-444}"
WALLCLOCK="${WALLCLOCK:-120}"   # 2-min proxy; bump to 600 for a real run

BASE_ENV="SEED=${SEED} MAX_WALLCLOCK_SECONDS=${WALLCLOCK} SKIP_GPTQ=1 SLOT_MAX_WINDOWS=0"

run_arm() {
    local arm_id="$1"; shift
    local extra_env="$*"
    local logfile="${SCRIPT_DIR}/suite_${arm_id}.log"
    echo ""
    echo "========================================================"
    echo "  ARM ${arm_id}  env: ${extra_env}"
    echo "========================================================"
    env ${BASE_ENV} ${extra_env} \
        python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
        "${TRAIN}" 2>&1 | tee "${logfile}"
    # Print the key BPB line immediately
    echo ""
    grep "final_sliding_window.*_exact" "${logfile}" | tail -3 || true
    echo ""
}

run_arm "CTRL-00"       "SLOT_ENABLED=0  SCALE_TTT_ENABLED=0"
run_arm "SLOT-01"       "SLOT_ENABLED=1  SCALE_TTT_ENABLED=0  SLOT_STEPS=8  SLOT_LR=0.005"
run_arm "SCALE-02"      "SLOT_ENABLED=0  SCALE_TTT_ENABLED=1  SCALE_TTT_LR=1e-4  SCALE_TTT_EPOCHS=1  SCALE_TTT_CHUNK=32768"
run_arm "SLOT+SCALE-03" "SLOT_ENABLED=1  SCALE_TTT_ENABLED=1  SLOT_STEPS=8  SLOT_LR=0.005  SCALE_TTT_LR=1e-4  SCALE_TTT_EPOCHS=1  SCALE_TTT_CHUNK=32768"

echo ""
echo "========================================================"
echo "  SUITE SUMMARY — final_sliding_window*_exact"
echo "========================================================"
for arm in CTRL-00 SLOT-01 SCALE-02 "SLOT+SCALE-03"; do
    logfile="${SCRIPT_DIR}/suite_${arm}.log"
    if [[ -f "${logfile}" ]]; then
        echo "--- ${arm} ---"
        grep "final_sliding_window.*_exact" "${logfile}" | tail -3
    fi
done
echo "========================================================"
