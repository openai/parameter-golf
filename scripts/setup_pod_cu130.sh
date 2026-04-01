#!/bin/bash
set -euo pipefail

echo "=== Installing torch cu130 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 -q

echo "=== Verifying torch ==="
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

echo "=== Verifying FA3 ==="
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"

echo "=== READY ==="
