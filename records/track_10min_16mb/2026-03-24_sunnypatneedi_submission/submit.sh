#!/bin/bash
# Run this AFTER 3-seed validation completes on RunPod.
# Collects logs and prepares the submission folder.
#
# Usage (on RunPod):
#   bash submit.sh <run_dir> <seed42_log> <seed1337_log> <seed2024_log>
#
# Example:
#   bash submit.sh . /workspace/run_seed42.log /workspace/run_seed1337.log /workspace/run_seed2024.log

set -e
DIR="${1:-.}"

echo "=== Collecting submission files ==="

# Copy logs
cp "$2" "${DIR}/train_seed42.log" 2>/dev/null && echo "Copied seed 42 log"
cp "$3" "${DIR}/train_seed1337.log" 2>/dev/null && echo "Copied seed 1337 log"
cp "$4" "${DIR}/train_seed2024.log" 2>/dev/null && echo "Copied seed 2024 log"

# Extract key metrics from logs
echo ""
echo "=== Results ==="
for log in "${DIR}"/train_seed*.log; do
  seed=$(basename "$log" | grep -oP '\d+')
  bpb=$(grep "final_int6_sliding_window_exact" "$log" | tail -1 | grep -oP 'val_bpb:\K[\d.]+')
  artifact=$(grep "Total submission size" "$log" | grep -oP '[\d]+')
  steps=$(grep "stopping_early" "$log" | grep -oP 'step:\K[\d]+')
  echo "Seed ${seed}: bpb=${bpb} artifact=${artifact} steps=${steps}"
done

echo ""
echo "=== TODO ==="
echo "1. Update submission.json with actual val_bpb and artifact_bytes"
echo "2. Update README.md with results table and technique description"
echo "3. Rename folder to: 2026-03-24_sunnypatneedi_<TechniqueName>"
echo "4. git add, commit, push to fork main branch"
echo "5. Open PR at: https://github.com/openai/parameter-golf/compare"
