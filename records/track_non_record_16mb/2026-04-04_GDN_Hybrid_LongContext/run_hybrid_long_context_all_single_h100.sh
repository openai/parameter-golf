#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seq in 16384 32768; do
  RUN_ID="hybrid_gdn_1xh100_600s_seq${seq}" \
    "$ROOT_DIR/run_hybrid_long_context_single_h100.sh" "$seq"
done
