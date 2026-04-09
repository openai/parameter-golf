#!/usr/bin/env bash
set -euo pipefail

SFW_TARGET_GPU_COUNT="${SFW_TARGET_GPU_COUNT:-8}" \
SFW_NPROC_PER_NODE="${SFW_NPROC_PER_NODE:-8}" \
SFW_VAL_TOKEN_LIMIT="${SFW_VAL_TOKEN_LIMIT:-0}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runpod_full.sh" "$@"
