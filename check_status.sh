#!/usr/bin/env bash
cd /mnt/c/Users/wrc02/Desktop/Projects/NanoGPT-Challenge/repo

echo "=== Running python3 processes ==="
ps aux | grep python3 | grep -v grep || echo "(none)"

echo ""
echo "=== sweep_results.txt ==="
cat sweep_results.txt 2>/dev/null || echo "(not yet)"

echo ""
echo "=== Training output from sweep logs (skipping source dump) ==="
for f in $(ls -t logs/sweep_*.txt 2>/dev/null); do
  echo "--- $f ---"
  # Skip the source code block; real output starts after the ==== separator
  awk '/^={80}/{found++} found>=2' "$f" | grep -E "(step:|val_bpb|warmup_step|peak memory|final_ternary)" | head -20
  echo ""
done
