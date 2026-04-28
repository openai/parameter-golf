#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  JR-02 — Triton Candidate"
echo "  kernel_mode=triton_act compile_mode=${COMPILE_MODE:-default}"
echo "  loader follows JR-01 winner unless overridden"
echo "============================================"

exec env \
    MLP_KERNEL_MODE="${MLP_KERNEL_MODE:-triton_act}" \
    COMPILE_MODE="${COMPILE_MODE:-}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}" \
    bash "${SCRIPT_DIR}/run.sh"
