#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running Ternary Experiments..."

bash "$SCRIPT_DIR/ternary_10L_d768.sh"
bash "$SCRIPT_DIR/ternary_12L_d768.sh"
bash "$SCRIPT_DIR/ternary_12L_d768_fp8.sh"
bash "$SCRIPT_DIR/ternary_12L_d768_muon5.sh"
bash "$SCRIPT_DIR/ternary_ptq_only_12L_d768.sh"

echo "All ternary experiments completed."
