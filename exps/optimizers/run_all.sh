#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"

bash "$SCRIPT_DIR/muon_adam.sh"
bash "$SCRIPT_DIR/muon_adamw.sh"
bash "$SCRIPT_DIR/muon_steps3.sh"
bash "$SCRIPT_DIR/muon_steps10.sh"
bash "$SCRIPT_DIR/muon_mom90.sh"
bash "$SCRIPT_DIR/muon_mom98.sh"
bash "$SCRIPT_DIR/adam.sh"
bash "$SCRIPT_DIR/adamw.sh"
