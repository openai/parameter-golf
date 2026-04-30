#!/usr/bin/env bash
cd /mnt/c/Users/wrc02/Desktop/Projects/NanoGPT-Challenge/repo
# Show tail of every sweep log, newest first
for f in $(ls -t logs/sweep_*.txt 2>/dev/null); do
  echo "=== $f ==="
  grep -E "(step:|val_bpb:|train_loss:|EXPERIMENT|final_bpb|WARNING)" "$f" | tail -15
  echo ""
done
echo "=== sweep_results.txt ==="
cat sweep_results.txt 2>/dev/null || echo "(not yet)"
