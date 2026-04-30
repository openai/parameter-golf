#!/bin/bash
# Run smearonly first, then smear_gate2d. Each ~15 min.
set -e
cd /parameter-golf
echo "=== START $(date -u) chain run ==="
echo "[1/2] smearonly_s42"
bash run_smearonly.sh 2>&1 | tee /workspace/smearonly_run.log
echo "=== smearonly done $(date -u) ==="
echo "[2/2] smear_gate2d_s42"
bash run_smear_gate2d.sh 2>&1 | tee /workspace/smear_gate2d_run.log
echo "=== smear_gate2d done $(date -u) ==="
echo "=== CHAIN_COMPLETE ==="
