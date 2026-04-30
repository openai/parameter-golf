#!/usr/bin/env bash
echo "=== Killing all python3 training processes ==="
pkill -f "python3 train_gpt.py" && echo "Killed." || echo "No training process found."
ps aux | grep python3 | grep -v grep || echo "(no python3 processes remaining)"
