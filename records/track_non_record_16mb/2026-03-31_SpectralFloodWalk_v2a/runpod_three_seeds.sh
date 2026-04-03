#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="${SFW_PROFILE_SCRIPT:-runpod_spine_b.sh}"

for seed in 1337 2025 42; do
  SFW_SEED="${seed}" "${DIR}/${PROFILE_SCRIPT}"
done
