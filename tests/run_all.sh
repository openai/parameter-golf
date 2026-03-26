#!/bin/bash
# Run all tests. Call before every commit.
set -e
cd "$(dirname "$0")/.."
echo "=== Running Parameter Golf Test Suite ==="
python -m pytest tests/ -v --tb=short 2>&1
echo ""
echo "=== Artifact size check ==="
if [ -f "final_model.int8.ptz" ]; then
    python -c "
import os
code = open('train_gpt.py','rb').read()
model = open('final_model.int8.ptz','rb').read()
total = len(code) + len(model)
print(f'Code:  {len(code):>12,} bytes')
print(f'Model: {len(model):>12,} bytes')
print(f'Total: {total:>12,} / 16,000,000')
status = 'OK' if total < 16_000_000 else 'OVER LIMIT'
print(f'Status: {status}')
exit(0 if total < 16_000_000 else 1)
"
else
    echo "(no final_model.int8.ptz yet — skipping size check)"
fi
