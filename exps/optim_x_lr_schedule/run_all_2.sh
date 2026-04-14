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

bash "$SCRIPT_DIR/muon_steps10__rsqrt.sh"
bash "$SCRIPT_DIR/muon_mom90__trapezoid.sh"
bash "$SCRIPT_DIR/muon_mom90__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/muon_mom90__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/muon_mom90__cosine.sh"
bash "$SCRIPT_DIR/muon_mom90__cosine_min10.sh"
bash "$SCRIPT_DIR/muon_mom90__linear.sh"
bash "$SCRIPT_DIR/muon_mom90__constant.sh"
bash "$SCRIPT_DIR/muon_mom90__rsqrt.sh"
bash "$SCRIPT_DIR/muon_mom98__trapezoid.sh"
bash "$SCRIPT_DIR/muon_mom98__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/muon_mom98__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/muon_mom98__cosine.sh"
bash "$SCRIPT_DIR/muon_mom98__cosine_min10.sh"
bash "$SCRIPT_DIR/muon_mom98__linear.sh"
bash "$SCRIPT_DIR/muon_mom98__constant.sh"
bash "$SCRIPT_DIR/muon_mom98__rsqrt.sh"
bash "$SCRIPT_DIR/adam__trapezoid.sh"
bash "$SCRIPT_DIR/adam__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/adam__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/adam__cosine.sh"
bash "$SCRIPT_DIR/adam__cosine_min10.sh"
bash "$SCRIPT_DIR/adam__linear.sh"
bash "$SCRIPT_DIR/adam__constant.sh"
bash "$SCRIPT_DIR/adam__rsqrt.sh"
bash "$SCRIPT_DIR/adamw__trapezoid.sh"
bash "$SCRIPT_DIR/adamw__trapezoid_cosine.sh"
bash "$SCRIPT_DIR/adamw__trapezoid_cosine_min10.sh"
bash "$SCRIPT_DIR/adamw__cosine.sh"
bash "$SCRIPT_DIR/adamw__cosine_min10.sh"
bash "$SCRIPT_DIR/adamw__linear.sh"
bash "$SCRIPT_DIR/adamw__constant.sh"
bash "$SCRIPT_DIR/adamw__rsqrt.sh"
