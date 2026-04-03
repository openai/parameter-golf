#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=smoke \
SEEDS="${SEEDS:-444}" \
ITERATIONS="${ITERATIONS:-2200}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-0}" \
"${SCRIPT_DIR}/run_ab_matrix.sh"
