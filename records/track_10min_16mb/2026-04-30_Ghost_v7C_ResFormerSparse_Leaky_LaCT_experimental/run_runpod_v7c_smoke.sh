#!/usr/bin/env bash
# ============================================================
# run_runpod_v7c_smoke.sh
# Ghost v7C — ResFormer Sparse + LeakyReLU² + LaCT TTT
# RunPod validation helper
#
# USAGE:
#   bash run_runpod_v7c_smoke.sh
#
# This script:
#   1. Runs micro_sim.py as a lightweight pre-flight check.
#   2. If micro_sim passes, prints the exact torchrun command
#      for the full 8xH100 validation run.
#
# IMPORTANT: The expensive torchrun command is printed but NOT
# executed automatically. You must copy-paste and run it
# yourself after reviewing the micro_sim output.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment variables for the full 8xH100 run ──────────
export RESFORMER_ENABLED=1
export RESFORMER_MODE=sparse
export RESFORMER_LEARNED=1
export RESFORMER_DETACH_V0=1
export LEAKY_RELU_SLOPE=0.5
export LACT_ENABLED=1
export TTT_ENABLED=1
export TTT_NO_QV=1
export SEED=42

echo "============================================================"
echo "  Ghost v7C — Pre-flight micro_sim check"
echo "============================================================"
echo ""
echo "Environment variables set:"
echo "  RESFORMER_ENABLED=${RESFORMER_ENABLED}"
echo "  RESFORMER_MODE=${RESFORMER_MODE}"
echo "  RESFORMER_LEARNED=${RESFORMER_LEARNED}"
echo "  RESFORMER_DETACH_V0=${RESFORMER_DETACH_V0}"
echo "  LEAKY_RELU_SLOPE=${LEAKY_RELU_SLOPE}"
echo "  LACT_ENABLED=${LACT_ENABLED}"
echo "  TTT_ENABLED=${TTT_ENABLED}"
echo "  TTT_NO_QV=${TTT_NO_QV}"
echo "  SEED=${SEED}"
echo ""

# ── Step 1: Run micro_sim.py ────────────────────────────────
echo "------------------------------------------------------------"
echo "STEP 1: Running micro_sim.py ..."
echo "------------------------------------------------------------"

python3 "${SCRIPT_DIR}/micro_sim.py"
MICRO_SIM_EXIT=$?

if [ "${MICRO_SIM_EXIT}" -ne 0 ]; then
    echo ""
    echo "ERROR: micro_sim.py exited with code ${MICRO_SIM_EXIT}."
    echo "Do NOT proceed to the full 8xH100 run until micro_sim passes."
    exit "${MICRO_SIM_EXIT}"
fi

echo ""
echo "micro_sim.py PASSED (exit code 0)."
echo ""

# ── Step 2: Print the 8xH100 torchrun command ───────────────
echo "============================================================"
echo "  STEP 2: Full 8xH100 validation command (NOT auto-started)"
echo "============================================================"
echo ""
echo "micro_sim passed. When you are ready to run the full"
echo "8xH100 validation, execute the following command:"
echo ""
echo "────────────────────────────────────────────────────────────"
cat <<'EOF'
RESFORMER_ENABLED=1 \
RESFORMER_MODE=sparse \
RESFORMER_LEARNED=1 \
RESFORMER_DETACH_V0=1 \
LEAKY_RELU_SLOPE=0.5 \
LACT_ENABLED=1 \
TTT_ENABLED=1 \
TTT_NO_QV=1 \
SEED=42 \
torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_gpt.py
EOF
echo "────────────────────────────────────────────────────────────"
echo ""
echo "NOTE: This command is printed for your review only."
echo "      It has NOT been started automatically."
echo "      Run it manually on your 8xH100 RunPod instance"
echo "      only after you have explicitly approved."
echo ""
echo "Submission folder: ${SCRIPT_DIR}"
echo "Done."
