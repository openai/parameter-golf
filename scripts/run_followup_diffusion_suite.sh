#!/usr/bin/env zsh
set -u

SCRIPT_DIR=${0:A:h}
ROOT_DIR=${SCRIPT_DIR:h}
MANIFEST=${ROOT_DIR}/configs/followup/manifest.txt

export MANIFEST
if [[ $# -gt 0 ]]; then
  exec "${ROOT_DIR}/scripts/run_overnight_diffusion_suite.sh" "$@"
else
  exec "${ROOT_DIR}/scripts/run_overnight_diffusion_suite.sh" "followup_diffusion_$(date +%Y%m%d_%H%M%S)"
fi
