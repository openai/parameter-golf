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

bash "$SCRIPT_DIR/muon_adam__trapezoid.sh"
bash "$SCRIPT_DIR/muon_adam__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/muon_adam__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/muon_adam__cosine.sh"
bash "$SCRIPT_DIR/muon_adam__cosine_min10.sh"
bash "$SCRIPT_DIR/muon_adam__linear.sh"
bash "$SCRIPT_DIR/muon_adam__constant.sh"
bash "$SCRIPT_DIR/muon_adam__rsqrt.sh"
bash "$SCRIPT_DIR/muon_adamw__trapezoid.sh"
bash "$SCRIPT_DIR/muon_adamw__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/muon_adamw__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/muon_adamw__cosine.sh"
bash "$SCRIPT_DIR/muon_adamw__cosine_min10.sh"
bash "$SCRIPT_DIR/muon_adamw__linear.sh"
bash "$SCRIPT_DIR/muon_adamw__constant.sh"
bash "$SCRIPT_DIR/muon_adamw__rsqrt.sh"
bash "$SCRIPT_DIR/muon_steps3__trapezoid.sh"
bash "$SCRIPT_DIR/muon_steps3__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/muon_steps3__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/muon_steps3__cosine.sh"
bash "$SCRIPT_DIR/muon_steps3__cosine_min10.sh"
bash "$SCRIPT_DIR/muon_steps3__linear.sh"
bash "$SCRIPT_DIR/muon_steps3__constant.sh"
bash "$SCRIPT_DIR/muon_steps3__rsqrt.sh"
bash "$SCRIPT_DIR/muon_steps10__trapezoid.sh"
bash "$SCRIPT_DIR/muon_steps10__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/muon_steps10__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/muon_steps10__cosine.sh"
bash "$SCRIPT_DIR/muon_steps10__cosine_min10.sh"
bash "$SCRIPT_DIR/muon_steps10__linear.sh"
bash "$SCRIPT_DIR/muon_steps10__constant.sh"