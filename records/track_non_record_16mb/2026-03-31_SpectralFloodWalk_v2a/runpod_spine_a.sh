#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runpod_common.sh"

SFW_SPINE_VARIANT="${SFW_SPINE_VARIANT:-plain}" \
SFW_XSA_LAST_N="${SFW_XSA_LAST_N:-0}" \
sfw_run_profile spine_a "${SFW_SEED:-1337}" "$@"
