#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SLOT="${1:-}"

if [[ -z "$SLOT" ]]; then
  echo "usage: $0 <slot> [--nproc-per-node 8] [--label h100] [--dry-run]"
  echo "slots: S2-B0 S2-E1 S2-E2 S2-E3 S2-E4 S2-E5 S2-E6 S2-E6B S2-E7 S2-E8"
  exit 1
fi
shift

source "$ROOT_DIR/.venv/bin/activate"
cd "$ROOT_DIR"

uv run python pgolf/parameter-golf/stage2/h100_matrix_r1/run_family.py --slot "$SLOT" "$@"
