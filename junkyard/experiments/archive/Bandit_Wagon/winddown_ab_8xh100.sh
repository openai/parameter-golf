#!/usr/bin/env bash
set -euo pipefail

# BANDIT_WAGON 8xH100 winddown sweep launcher.
# Intended to run on top of a finished Bandit Wagon variation checkpoint.
#
# Usage:
#   MODEL_PATH=/abs/path/to/final_model.pt \
#   SEEDS=444 \
#   bash experiments/Bandit_Wagon/winddown_ab_8xh100.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export SEEDS="${SEEDS:-${SEED:-1337}}"
export DRY_RUN="${DRY_RUN:-0}"
export AUTO_ARCH_FROM_CKPT="${AUTO_ARCH_FROM_CKPT:-1}"

# Full matrix by default on 8xH100.
export ARM_FILTER="${ARM_FILTER:-}"

echo "============================================"
echo "  BANDIT_WAGON 8xH100 winddown sweep"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  SEEDS=${SEEDS}"
echo "  AUTO_ARCH_FROM_CKPT=${AUTO_ARCH_FROM_CKPT}"
if [[ -n "${ARM_FILTER}" ]]; then
  echo "  ARM_FILTER=${ARM_FILTER}"
fi
echo "  DRY_RUN=${DRY_RUN}"
echo "============================================"

bash "${SCRIPT_DIR}/winddown_ab.sh"

