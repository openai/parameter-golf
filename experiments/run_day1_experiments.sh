#!/bin/bash
set -euo pipefail

# Day 1 M4 shortlist sweep (fast mode)
# Shortlist mapping:
# - E05: Combo #2 (Foundation Stack)
# - E03: Combo #7 (QK-Gain 4.0)
# - E06: Combo #10 (Focal Loss)
# - E07: Combo #9 (Stride-16 Eval)
# - E02: Combo #6 (Warmdown 5000)
# - E10: Combo #50 partial (Sanath Stack partial)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Select Python interpreter: explicit override > project .venv > system python3.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
else
  PYTHON_BIN="python3"
fi
export PYTHON_BIN

DAY1_EXPERIMENTS=(
  E05
  E03
  E06
  E07
  E02
  E10
)

cd "$PROJECT_ROOT"

echo "=== Day 1 Experiment Sweep (FAST) ==="
echo "Project root: $PROJECT_ROOT"
echo "Experiments: ${DAY1_EXPERIMENTS[*]}"
echo ""

for exp_id in "${DAY1_EXPERIMENTS[@]}"; do
  echo "--- Running $exp_id ---"
  bash "$SCRIPT_DIR/run_m4_experiments.sh" --fast --single "$exp_id"
  echo ""
done

echo "=== Parsing results ==="
"$PYTHON_BIN" "$SCRIPT_DIR/analyze_results.py"

echo "Done. See logs in logs/ and summary in experiments/results_summary.md"
