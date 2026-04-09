#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${SFW_PROFILE_SCRIPT:-runpod_full.sh}"
SEEDS="${SFW_SEEDS:-1337 42 2025}"

for seed in ${SEEDS}; do
  echo "=== Running ${PROFILE} for seed ${seed} ==="
  SFW_SEED="${seed}" "${SCRIPT_DIR}/${PROFILE}" "$@"
done
