#!/bin/bash
# H100 Smoke Test + Benchmark for train_gpt_mega.py
# Run on Thunder Compute 1x H100 PCIe (~$0.38/hr)
#
# Usage:
#   bash megakernel/h100_test.sh         # full test
#   bash megakernel/h100_test.sh --bench # benchmark only (no training)
#
# After verifying speedup with this script:
#   export FUSED_RMSNORM_MLP=1
#   export FUSED_RMSNORM_QKV=1
# and re-run the training.

set -e
cd "$(dirname "$0")/.."

echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python3 -c "import triton; print(f'Triton: {triton.__version__}')"
echo ""

# Step 1: Benchmark the fused kernels
echo "=== Step 1: Kernel benchmark ==="
cd megakernel
FUSED_RMSNORM_MLP=1 FUSED_RMSNORM_QKV=1 python3 h100_benchmark.py
cd ..
echo ""

if [[ "$1" == "--bench" ]]; then
    echo "Benchmark-only mode, skipping training test."
    exit 0
fi

# Step 2: Import check
echo "=== Step 2: Import check ==="
python3 -c "
import ast, sys
src = open('megakernel/train_gpt_mega.py').read()
ast.parse(src)
print(f'AST clean. Lines: {len(src.splitlines())}')
"

# Step 3: Smoke test (1-GPU, 10 steps, small batch)
echo "=== Step 3: Smoke test (10 steps, fused kernels ON) ==="
FUSED_RMSNORM_MLP=1 FUSED_RMSNORM_QKV=1 \
  TRAIN_STEPS=10 \
  python3 megakernel/train_gpt_mega.py \
    --input_bin "data/fineweb10B_train_000001.bin" \
    --input_val_bin "data/fineweb10B_val_000000.bin" \
    --output_dir /tmp/mega_smoke \
    --num_iterations 10 \
    --sequence_length 512 \
    --batch_size 8 \
    2>&1 | tail -30

echo ""
echo "=== Step 4: Compare with fused OFF ==="
FUSED_RMSNORM_MLP=0 FUSED_RMSNORM_QKV=0 \
  TRAIN_STEPS=10 \
  python3 megakernel/train_gpt_mega.py \
    --input_bin "data/fineweb10B_train_000001.bin" \
    --input_val_bin "data/fineweb10B_val_000000.bin" \
    --output_dir /tmp/mega_smoke_base \
    --num_iterations 10 \
    --sequence_length 512 \
    --batch_size 8 \
    2>&1 | tail -20

echo ""
echo "=== Done. Check step times above. ==="
echo "If FUSED is faster: set FUSED_RMSNORM_MLP=1 FUSED_RMSNORM_QKV=1 for full run."
echo "If FUSED is slower: defaults (0) are correct — kernels don't help on this hardware."
