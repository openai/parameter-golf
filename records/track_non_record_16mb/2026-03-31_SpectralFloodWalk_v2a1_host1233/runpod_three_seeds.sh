#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in 42 1337 2025; do
  SFW_SEED="${seed}" "${SCRIPT_DIR}/runpod_full.sh"
done
