#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"


bash "$SCRIPT_DIR/beta2_0.99.sh"
bash "$SCRIPT_DIR/beta1_0.0.sh"
bash "$SCRIPT_DIR/beta1_0.95.sh"
bash "$SCRIPT_DIR/tied_emb_lr_0.02.sh"
bash "$SCRIPT_DIR/tied_emb_lr_0.1.sh"
bash "$SCRIPT_DIR/tied_emb_lr_0.2.sh"
bash "$SCRIPT_DIR/matrix_lr_0.02.sh"
bash "$SCRIPT_DIR/matrix_lr_0.06.sh"
bash "$SCRIPT_DIR/matrix_lr_0.08.sh"
bash "$SCRIPT_DIR/baseline.sh"
