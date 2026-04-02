#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runpod_common.sh"

SFW_MEMORY_MIN_READ_COUNT="${SFW_MEMORY_MIN_READ_COUNT:-2}" \
SFW_MAINTENANCE_PASSES="${SFW_MAINTENANCE_PASSES:-0}" \
SFW_MAINTENANCE_MAX_SLOTS="${SFW_MAINTENANCE_MAX_SLOTS:-0}" \
sfw_run_profile gate2_nomaint "${SFW_SEED:-1337}" "$@"
