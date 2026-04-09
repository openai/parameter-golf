#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== V1a baseline ==="
"${SCRIPT_DIR}/runpod_baseline.sh" "$@"

echo "=== V1a semantic1 ==="
"${SCRIPT_DIR}/runpod_semantic1.sh" "$@"

echo "=== V1a full ==="
"${SCRIPT_DIR}/runpod_full.sh" "$@"

echo
echo "Recent runs:"
find "${SCRIPT_DIR}/runs" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 3
