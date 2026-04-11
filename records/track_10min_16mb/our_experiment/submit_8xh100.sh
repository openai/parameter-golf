#!/bin/bash
# 3-seed validation on 8xH100 SXM (RunPod)
# p21 config: XSA-all + LeakyReLU(0.5)² + VR + GA
# Changes vs v9: warmdown=3000 (was 3500), auto zstd/int5 fallback, n-gram auto-on
#
# Built-in features (no env vars needed):
#   - warmdown=3000 default (46.5% ratio at ~6450 steps)
#   - 5-gram cache auto-enabled on multi-GPU (alpha=0.20, order=5)
#   - Artifact auto-downgrade: int6+zstd-16 → try zstd-[1,17,2] → int5 middle layers

set -e
cd "$(dirname "$0")"
mkdir -p logs

export TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export XSA_LAST_N=11 LEAKY_RELU=1
export MAX_WALLCLOCK_SECONDS=600

echo "=== Seed 1337 ==="
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/p21_seed1337.txt

echo "=== Seed 42 ==="
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/p21_seed42.txt

echo "=== Seed 7 ==="
SEED=7 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/p21_seed7.txt

echo ""
echo "========================================="
echo "=== 3-SEED RESULTS ==="
echo "========================================="
for f in logs/p21_seed*.txt; do
    echo ""
    echo "--- $(basename $f) ---"
    grep "quant_try\|quant_fallback" "$f" 2>/dev/null | tail -2
    grep "Total submission size" "$f" 2>/dev/null
    grep "final_int6_sliding_window_exact" "$f" 2>/dev/null
done
