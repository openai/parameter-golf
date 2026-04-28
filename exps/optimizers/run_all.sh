#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
if [ -f "${SCRIPT_DIR}/../../.env" ]; then
  source "${SCRIPT_DIR}/../../.env"
elif [ -f .env ]; then
  source .env
fi
export COMET_API_KEY="${COMET_API_KEY:-}"

bash "$SCRIPT_DIR/muon_adam.sh"
bash "$SCRIPT_DIR/muon_adamw.sh"
bash "$SCRIPT_DIR/muon_steps3.sh"
bash "$SCRIPT_DIR/muon_steps10.sh"
bash "$SCRIPT_DIR/muon_mom90.sh"
bash "$SCRIPT_DIR/muon_mom98.sh"
bash "$SCRIPT_DIR/adam.sh"
bash "$SCRIPT_DIR/adamw.sh"
