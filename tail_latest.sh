#!/usr/bin/env bash
cd /mnt/c/Users/wrc02/Desktop/Projects/NanoGPT-Challenge/repo
LATEST=$(ls -t logs/sweep_*.txt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "=== $LATEST (last 40 lines) ==="
  tail -40 "$LATEST"
else
  echo "(no sweep logs yet)"
fi
echo ""
echo "=== sweep_results.txt ==="
cat sweep_results.txt 2>/dev/null || echo "(not yet)"
