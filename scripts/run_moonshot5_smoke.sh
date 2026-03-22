#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNS=(drope_eval yarn_eval mtp_low muon_balance hybrid_delta)
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

for run in "${RUNS[@]}"; do
  echo
  echo "=== Smoke ${run} ==="
  NPROC_PER_NODE="$NPROC_PER_NODE" bash scripts/run_smoke_profile.sh "$run"
  echo
  echo "--- tail ${run}_smoke ---"
  tail -n 10 "logs/${run}_smoke.txt"
done
