#!/bin/bash
set -euo pipefail

DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix"
BUILD_SCRIPT="$RECORD_ROOT/build_prefix_blob.py"

case "$DATA_ROOT_MODE" in
  workspace)
    VAL_DIR="/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
    ;;
  tmp)
    VAL_DIR="/tmp/parameter-golf-data/datasets/fineweb10B_sp1024"
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp"
    exit 1
    ;;
esac

python3 "$BUILD_SCRIPT" --val-dir "$VAL_DIR" --output "$RECORD_ROOT/prefix_512k.xz" --budget-bytes 512000
python3 "$BUILD_SCRIPT" --val-dir "$VAL_DIR" --output "$RECORD_ROOT/prefix_768k.xz" --budget-bytes 768000
python3 "$BUILD_SCRIPT" --val-dir "$VAL_DIR" --output "$RECORD_ROOT/prefix_1m.xz" --budget-bytes 1048576
python3 "$BUILD_SCRIPT" --val-dir "$VAL_DIR" --output "$RECORD_ROOT/prefix_2m.xz" --budget-bytes 2097152

ls -lh "$RECORD_ROOT"/prefix_*.xz
