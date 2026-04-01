#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  JUNKYARD RAT MLP — Competition Runner"
echo "  compact artifact mode: ON"
echo "  kernel_mode: ${MLP_KERNEL_MODE:-fused_mlp}"
echo "  loader: ${LOADER_MODE:-coprime}"
echo "  final eval: int6 roundtrip + sliding window"
echo "============================================"

exec env \
    COMPETITION_ARTIFACT=1 \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
    LOADER_MODE="${LOADER_MODE:-coprime}" \
    TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS:-1}" \
    COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
    COMPILE_MODE="${COMPILE_MODE:-max-autotune}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}" \
    MLP_KERNEL_MODE="${MLP_KERNEL_MODE:-fused_mlp}" \
    ATTN_SCALE_INIT="${ATTN_SCALE_INIT:-1.0}" \
    TRIGRAM=0 \
    NGRAM_EVAL_ORDER=0 \
    CUBRIC_CADENCE=0 \
    bash "${SCRIPT_DIR}/run.sh"
