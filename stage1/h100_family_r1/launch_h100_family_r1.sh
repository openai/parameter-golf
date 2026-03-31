#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SLOT="${1:-}"

if [[ -z "$SLOT" ]]; then
  echo "usage: $0 <slot>"
  echo "slots: P0 P1 P2 P3 P4 P5 P6 P7"
  exit 1
fi
shift

source "$ROOT_DIR/.venv/bin/activate"
cd "$ROOT_DIR"

uv run python pgolf/parameter-golf/stage1/h100_family_r1/run_family.py --slot "$SLOT" "$@"
