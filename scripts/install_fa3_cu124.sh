#!/bin/bash
set -euo pipefail

PY=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Python tag: ${PY}"

for ABI in FALSE TRUE; do
    URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu124torch2.4cxx11abi${ABI}-${PY}-${PY}-linux_x86_64.whl"
    echo "Trying: ${URL}"
    if pip install --no-cache-dir "${URL}" -q; then
        echo "Installed FA3 (cxx11abi=${ABI})"
        break
    else
        echo "Failed cxx11abi=${ABI}, trying next..."
    fi
done

python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
