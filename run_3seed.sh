#!/bin/bash
# Run 3-seed statistical significance test
# Usage: bash run_3seed.sh <script.sh>
# Example: bash run_3seed.sh run_enhanced.sh
#
# WARNING: This uses ~$12 of Runpod credits (3 full runs).
# Only use after you've validated the config with a single seed first.

SCRIPT=${1:-run_enhanced.sh}

echo "=== 3-seed run: $SCRIPT ==="
echo "Starting seed 1337..."
SEED=1337 bash $SCRIPT 2>&1 | tee logs/seed1337.txt

echo "Starting seed 42..."
SEED=42 bash $SCRIPT 2>&1 | tee logs/seed42.txt

echo "Starting seed 2025..."
SEED=2025 bash $SCRIPT 2>&1 | tee logs/seed2025.txt

echo "=== All 3 seeds complete ==="
echo "Results:"
grep "val_bpb" logs/seed1337.txt | tail -1
grep "val_bpb" logs/seed42.txt | tail -1
grep "val_bpb" logs/seed2025.txt | tail -1
