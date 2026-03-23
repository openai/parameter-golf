#!/bin/bash
# Install FlashAttention-3 (Hopper) on RunPod H100
# Run this BEFORE training on any new pod

set -e

# Install zstandard (for compression)
pip install --break-system-packages -q zstandard

# Install FA3 from Dao-AILab repo (hopper branch)
# This builds the Hopper-optimized CUDA kernels
cd /tmp
if [ ! -d flash-attention ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention

# Install the main package first (includes flash_attn_interface for Hopper)
pip install --break-system-packages -e . --no-build-isolation 2>&1 | tail -5

# Verify
python3 -c "
try:
    from flash_attn_interface import flash_attn_func
    print('FA3 Hopper interface: OK (top-level)')
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        print('FA3 Hopper interface: OK (submodule)')
    except ImportError:
        print('FA3 Hopper interface: NOT FOUND')

from flash_attn import flash_attn_func
print(f'flash_attn: OK')
import flash_attn
print(f'Version: {flash_attn.__version__}')
"

echo "FA3 setup complete."
