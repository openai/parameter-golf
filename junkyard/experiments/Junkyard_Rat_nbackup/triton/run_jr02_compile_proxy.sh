#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env \
    MLP_KERNEL_MODE="" \
    COMPILE_MODE="${COMPILE_MODE:-max-autotune}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}" \
    bash "${SCRIPT_DIR}/../run.sh"
