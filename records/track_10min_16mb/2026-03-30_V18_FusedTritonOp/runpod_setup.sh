#!/bin/bash
# V17 RunPod Setup — PR #1089 (TurboMuon) + PR #1072 (Fused Triton Kernel)
# USAGE:
#   bash runpod_setup.sh          # Setup (PyTorch upgrade, deps)
#   bash runpod_setup.sh run      # Run training
set -e

if [ "$1" = "run" ]; then
    # ---- RUN MODE ----
    echo "=== V17 FusedTurboMuon ==="
    echo "Config: SEED=${SEED:-1337} GPTQ_RESERVE_MS=${GPTQ_RESERVE_MS:-9000}"
    echo "Starting in 3s..."
    sleep 3
    GPTQ_RESERVE_MS=${GPTQ_RESERVE_MS:-9000} \
    SEED=${SEED:-1337} \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    exit 0
fi

# ---- SETUP MODE ----
echo "============================================="
echo "  V17 FUSED TURBOMUON — POD SETUP"
echo "============================================="

# 1. Check CUDA driver
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "CUDA Driver: $DRIVER"

# 2. Check current PyTorch
CURRENT_PT=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "Current PyTorch: $CURRENT_PT"

# 3. Install deps (brotli required for compression, sentencepiece for tokenizer)
pip install brotli sentencepiece 2>&1 | tail -2

# 4. Install FA3 for SDPA backend acceleration (30-second wheel install)
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null && echo "FA3: already installed" || {
    echo "Installing FA3 pre-built wheel..."
    pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>&1 | tail -3
    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')" 2>/dev/null || echo "FA3: not available (SDPA will use FA2 or math backend)"
}

# 5. Symlink data if needed
[ -L data ] || [ -d data ] || ln -sf /workspace/data data
[ -d data/datasets/fineweb10B_sp1024 ] && echo "Data: OK" || echo "WARNING: Data not found at data/datasets/fineweb10B_sp1024"
[ -f data/tokenizers/fineweb_1024_bpe.model ] && echo "Tokenizer: OK" || echo "WARNING: Tokenizer not found"

# 6. Check Triton
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
try:
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor
    print(f'Triton {triton.__version__} + TensorDescriptor: OK → Fused MLP kernel ENABLED')
except Exception as e:
    print(f'Triton not available: {e} → Standard MLP path (slower)')
"

echo ""
echo "============================================="
echo "  SETUP COMPLETE"
echo "============================================="
echo ""
echo "  V17 Stack:"
echo "    Turbo-Muon + EngramLite + Parameter Banking (PR #1089)"
echo "    Fused Triton MLP kernel (PR #1072, if Triton available)"
echo "    Mixed-precision GPTQ int5/int6/int7 + Brotli compression"
echo "    GPTQ reserve optimized to 9s (from 14s default)"
echo ""
echo "RUN COMMANDS:"
echo ""
echo "  # Single seed test:"
echo "  SEED=1337 bash runpod_setup.sh run"
echo ""
echo "  # 3-seed submission:"
echo "  for S in 1337 42 999; do"
echo "    SEED=\$S bash runpod_setup.sh run | tee run_seed\$S.log"
echo "  done"
echo "============================================="
