#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runpod_common.sh"

SFW_SPINE_VARIANT="${SFW_SPINE_VARIANT:-xsa}" \
SFW_XSA_LAST_N="${SFW_XSA_LAST_N:-4}" \
sfw_run_profile spine_b "${SFW_SEED:-1337}" "$@"
