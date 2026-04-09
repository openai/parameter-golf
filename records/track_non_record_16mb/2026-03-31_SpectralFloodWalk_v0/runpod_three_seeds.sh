#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SFW_SEEDS:-1337 42 2025}"

for seed in ${SEEDS}; do
  echo "=== Running full profile for seed ${seed} ==="
  SFW_SEED="${seed}" "${SCRIPT_DIR}/runpod_full.sh" "$@"
done
