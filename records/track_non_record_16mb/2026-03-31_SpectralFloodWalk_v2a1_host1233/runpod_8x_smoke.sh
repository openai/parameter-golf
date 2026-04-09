#!/usr/bin/env bash
set -euo pipefail

SFW_TARGET_GPU_COUNT="${SFW_TARGET_GPU_COUNT:-8}" \
SFW_NPROC_PER_NODE="${SFW_NPROC_PER_NODE:-8}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_smoke.sh" "$@"
