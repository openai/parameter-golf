#!/usr/bin/env bash
# ============================================================================
# MASTER PERFORMANCE SWEEP: Batch Size & Compiler Modes
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Batch Size Sweep..."
bash "${DIR}/sweep_batch_size_small_skc.sh"

echo "Batch Size Sweep Complete."

echo "Master Performance Sweep Completed."
