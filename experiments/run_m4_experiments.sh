#!/bin/bash
set -euo pipefail

# Parameter Golf M4 Experiment Runner
# Runs all 10 experiments with optional --fast flag for quick iteration.
#
# Environment variables can override train budget for longer reruns:
#   MAX_WALLCLOCK_SECONDS=1800 WARMUP_STEPS=20 MAX_VAL_TOKENS=0
#
# Usage:
#   bash run_m4_experiments.sh                  # Run all experiments (full eval, ~60 min each)
#   bash run_m4_experiments.sh --fast           # Fast mode: truncated eval (~12 min each)
#   bash run_m4_experiments.sh --single E01     # Run just E01
#   bash run_m4_experiments.sh --rerun          # Re-run all even if logs already exist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"

# Select Python interpreter: explicit override > project .venv > system python3.
if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

# Defaults
FAST_MODE=false
SINGLE_EXP=""
FORCE_RERUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            shift
            ;;
        --single)
            SINGLE_EXP="$2"
            shift 2
            ;;
        --rerun)
            FORCE_RERUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_m4_experiments.sh [--fast] [--single E01] [--rerun]"
            exit 1
            ;;
    esac
done

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Preflight: this runner targets train_gpt_mlx.py, so MLX must be importable.
if ! "$PYTHON_BIN" -c "import mlx" >/dev/null 2>&1; then
    echo "ERROR: Python package 'mlx' is not available in the active environment."
    echo "Interpreter: $PYTHON_BIN"
    echo "Use your MLX-enabled Python env (or set PYTHON_BIN), then re-run this script."
    exit 1
fi

# Read experiments from Python config
echo "=== Parameter Golf M4 Experiment Runner ==="
echo "Mode: $([ "$FAST_MODE" = true ] && echo 'FAST (MAX_VAL_TOKENS=1000000)' || echo 'FULL EVAL')"
echo "Project root: $PROJECT_ROOT"
echo "Logs dir: $LOGS_DIR"
echo ""

# Get baseline BPB for comparison
BASELINE_BPB="3.2178"
echo "Baseline M4 BPB: $BASELINE_BPB"
echo ""

# Run experiments
RESULTS=()
count=0
total=10

while IFS='|' read -r exp_id exp_name exp_desc exp_vars; do
    # Skip empty lines
    [ -z "$exp_id" ] && continue

    # Check if we're filtering to single experiment
    if [ -n "$SINGLE_EXP" ] && [ "$exp_id" != "$SINGLE_EXP" ]; then
        continue
    fi

    count=$((count + 1))

    # Parse env vars
    env_cmd=""
    IFS=',' read -r -a env_pairs <<< "$exp_vars"
    for pair in "${env_pairs[@]}"; do
        [ -z "$pair" ] && continue
        env_cmd="$env_cmd $pair"
    done

    # Check if log exists
    log_file="$LOGS_DIR/exp_${exp_id}_${exp_name// /-}.txt"
    if [ -f "$log_file" ]; then
        if [ "$FORCE_RERUN" = true ]; then
            echo "[RERUN] $exp_id: $exp_name (overwriting existing log)"
            rm -f "$log_file"
        else
            echo "[SKIP] $exp_id: $exp_name (log exists)"
            # Still try to extract results from existing log
            final_bpb=$(awk '
                /final_int8_zlib_roundtrip_exact/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i ~ /^val_bpb:/) {
                            split($i, a, ":")
                            bpb = a[2]
                        }
                    }
                }
                END {
                    if (bpb != "") print bpb; else print "N/A"
                }
            ' "$log_file")
            delta="N/A"
            if [ "$final_bpb" != "N/A" ]; then
                delta=$($PYTHON_BIN -c "print(f'{float($final_bpb) - float($BASELINE_BPB):.6f}')")
            fi
            RESULTS+=("$exp_id|$exp_name|$final_bpb|$delta")
            continue
        fi
    fi

    # Print header
    echo ""
    echo "=============================================="
    echo "[$count/$total] Running Experiment: $exp_id"
    echo "  Name: $exp_name"
    echo "  Desc: $exp_desc"
    echo "  Log:  $log_file"
    echo "=============================================="
    echo ""

    # Add fast mode env var if needed
    if [ "$FAST_MODE" = true ]; then
        env_cmd="$env_cmd MAX_VAL_TOKENS=1000000 WARMUP_STEPS=5"
    fi

    # Run the training
    cd "$PROJECT_ROOT"
    eval "$env_cmd $PYTHON_BIN train_gpt_mlx.py" 2>&1 | tee -a "$log_file"

    # Extract final BPB from log
    final_bpb=$(awk '
        /final_int8_zlib_roundtrip_exact/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^val_bpb:/) {
                    split($i, a, ":")
                    bpb = a[2]
                }
            }
        }
        END {
            if (bpb != "") print bpb; else print "N/A"
        }
    ' "$log_file")
    delta="N/A"
    if [ "$final_bpb" != "N/A" ]; then
        delta=$($PYTHON_BIN -c "print(f'{float($final_bpb) - float($BASELINE_BPB):.6f}')")
    fi

    RESULTS+=("$exp_id|$exp_name|$final_bpb|$delta")

    echo ""
    echo "✓ $exp_id completed. Final BPB: $final_bpb (Δ: $delta)"
    echo ""
done < <($PYTHON_BIN "$SCRIPT_DIR/experiment_configs.py" --machine 2>/dev/null | grep 'E[0-9][0-9]' || echo "")

# Summary table
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
echo "| ID   | Experiment             | BPB     | Delta vs Baseline |"
echo "|------|------------------------|---------|------------------|"

if [ ${#RESULTS[@]} -gt 0 ]; then
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r exp_id exp_name bpb delta <<< "$result"
        printf "| %-4s | %-22s | %-7s | %-16s |\n" "$exp_id" "$exp_name" "$bpb" "$delta"
    done
else
    echo "| (no new results - all experiments skipped) |"
fi

echo ""
echo "Baseline BPB: $BASELINE_BPB"
echo "Mode: $([ "$FAST_MODE" = true ] && echo 'FAST (truncated eval)' || echo 'FULL EVAL')"
echo ""
echo "Next: Run 'python experiments/analyze_results.py' to generate detailed report"
