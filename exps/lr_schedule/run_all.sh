#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"

bash "$SCRIPT_DIR/trapezoid.sh"
bash "$SCRIPT_DIR/trapezoid_cosine.sh"
bash "$SCRIPT_DIR/trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/cosine.sh"
bash "$SCRIPT_DIR/cosine_min10.sh"
bash "$SCRIPT_DIR/linear.sh"
bash "$SCRIPT_DIR/constant.sh"
bash "$SCRIPT_DIR/rsqrt.sh"
