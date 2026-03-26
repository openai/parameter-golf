#!/bin/bash
# Fractal Cadence Experiments — DGX Spark
# Run all 4 tests sequentially, results in logs/

set -e
cd "$(dirname "$0")"

echo "=== Test 1: Cadence 2, fractal on step 1 (F/N/F/N) ==="
python train_fractal_cadence.py \
  --cadence 2 --cadence-offset 0 --gravity \
  --iterations 300 --run-id cadence2_step1

echo ""
echo "=== Test 2: Cadence 3, fractal on step 3 (N/N/F) ==="
python train_fractal_cadence.py \
  --cadence 3 --cadence-offset 2 --gravity \
  --iterations 300 --run-id cadence3_step3

echo ""
echo "=== Control A: Always fractal (old behavior) ==="
python train_fractal_cadence.py \
  --cadence 1 --gravity \
  --iterations 300 --run-id always_fractal

echo ""
echo "=== Control B: Never fractal (pure single-pass) ==="
python train_fractal_cadence.py \
  --cadence 0 \
  --iterations 300 --run-id never_fractal

echo ""
echo "=== All done. Logs in: ==="
ls -la logs/cadence*.tsv logs/always*.tsv logs/never*.tsv 2>/dev/null
