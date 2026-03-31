#!/bin/bash
set -euo pipefail

VAL_DIR="${1:-./data/datasets/fineweb10B_sp1024}"
OUT_DIR="${2:-./records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix}"
shift 2 || true
BUILDER="${BUILDER:-python3 records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/build_prefix_blob.py}"
BUDGETS=("$@")

if [ "${#BUDGETS[@]}" -eq 0 ]; then
  BUDGETS=(680000 700000 720000)
fi

mkdir -p "$OUT_DIR"

for budget in "${BUDGETS[@]}"; do
  out="$OUT_DIR/prefix_${budget}.raw"
  $BUILDER \
    --val-dir "$VAL_DIR" \
    --output "$out" \
    --budget-bytes "$budget"
done

ls -lh "$OUT_DIR"/prefix_*.raw
