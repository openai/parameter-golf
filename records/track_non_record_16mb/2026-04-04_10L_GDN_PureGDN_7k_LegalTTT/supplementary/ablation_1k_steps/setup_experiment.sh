#!/usr/bin/env bash
# setup_experiment.sh — Create a clean experiment directory with code copies.
#
# Usage: ./setup_experiment.sh <experiment_name>
# Example: ./setup_experiment.sh exp00_baseline
#
# Creates: experiments/<experiment_name>/code/ with copies of the base code files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CODE_DIR="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments"
EXPERIMENTS_DIR="$SCRIPT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 exp00_baseline"
    exit 1
fi

EXPERIMENT_NAME="$1"
EXPERIMENT_DIR="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}"

if [ -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Experiment directory already exists: $EXPERIMENT_DIR"
    echo "       Remove it first or choose a different name."
    exit 1
fi

# Create directory structure
mkdir -p "$EXPERIMENT_DIR/code"
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$EXPERIMENT_DIR/checkpoints"

# Copy base code files
for f in architectures.py configs.py train_gdn_7k.py eval_ttt.py; do
    if [ -f "$BASE_CODE_DIR/$f" ]; then
        cp "$BASE_CODE_DIR/$f" "$EXPERIMENT_DIR/code/$f"
    else
        echo "WARNING: Base file not found: $BASE_CODE_DIR/$f"
    fi
done

echo "Created experiment: $EXPERIMENT_DIR"
echo "  code/          — copied base code (modify as needed)"
echo "  logs/          — training and eval logs"
echo "  checkpoints/   — model checkpoints and artifacts"
echo ""
echo "Next steps:"
echo "  1. Edit $EXPERIMENT_DIR/code/ files for your experiment"
echo "  2. Run: ./run_experiment.sh $EXPERIMENT_NAME"
echo "  3. Or submit: sbatch slurm_template.sh  (after setting EXPERIMENT_NAME)"
