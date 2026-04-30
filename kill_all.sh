#!/usr/bin/env bash
echo "=== Killing compile workers and sweep ==="
pkill -f "compile_worker" && echo "Killed compile workers." || echo "(no compile workers)"
pkill -f "sweep_hybrid" && echo "Killed sweep." || echo "(no sweep)"
pkill -f "train_gpt" && echo "Killed train_gpt." || echo "(no train_gpt)"
sleep 2
echo "=== Remaining python3 processes ==="
ps aux | grep python3 | grep -v grep | grep -v unattended || echo "(none)"
