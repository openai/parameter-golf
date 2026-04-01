#!/usr/bin/env bash
# setup_cu124_fa3_venv.sh — Create cu124 venv with FA3 bridge from system.
# Based on the original working approach from c6caaad (custom FA3 header install).
#
# Usage: bash scripts/setup_cu124_fa3_venv.sh
#   Then: source .venv-cu124/bin/activate
#         export PYTHONPATH="${FA3_DIR}:${PYTHONPATH:-}"
#         SEED=300 bash neural/2026-03-31_Rascal_III_SLOT/run.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv-cu124"

echo "============================================"
echo "  cu124 + FA3 venv setup"
echo "============================================"

# ── Step 0: Clean up space ──
echo ""
echo "[0/4] Cleaning up disk space..."
pip cache purge 2>/dev/null || true
rm -rf /tmp/pip-* /tmp/ccm* /workspace/flash-attention/hopper/build 2>/dev/null || true
rm -rf "${VENV_DIR}" 2>/dev/null || true
echo "  $(df -h / | awk 'NR==2{print $4}') available"

# ── Step 1: Find system FA3 ──
echo ""
echo "[1/4] Locating system FA3..."
FA3_DIR=""

# Method A: check system pythons for importable flash_attn_interface
for _py in $(which -a python3 2>/dev/null | awk '!seen[$0]++') /usr/bin/python3 /opt/conda/bin/python3 /usr/local/bin/python3; do
    [ -x "${_py}" ] || continue
    _dir=$("${_py}" -c "
import inspect, os
try:
    import flash_attn_interface
    print(os.path.dirname(inspect.getfile(flash_attn_interface)))
except: pass
" 2>/dev/null) || true
    if [ -n "${_dir}" ]; then
        FA3_DIR="${_dir}"
        echo "  Found via ${_py}: ${FA3_DIR}"
        break
    fi
done

# Method B: filesystem scan for flash_attn_interface.py
if [ -z "${FA3_DIR}" ]; then
    echo "  Python import failed, scanning filesystem..."
    while IFS= read -r _f; do
        [ -n "${_f}" ] || continue
        FA3_DIR="$(dirname "${_f}")"
        echo "  Found file: ${_f}"
        break
    done < <(find /workspace /opt/conda /usr/local /root -type f -name "flash_attn_interface.py" 2>/dev/null | head -1)
fi

if [ -z "${FA3_DIR}" ]; then
    echo "  FATAL: cannot find flash_attn_interface anywhere on this pod."
    exit 1
fi
echo "  FA3_DIR=${FA3_DIR}"

# Check what the interface needs
echo ""
echo "  Checking FA3 dependencies..."
if grep -q "import flash_attn_3._C" "${FA3_DIR}/flash_attn_interface.py" 2>/dev/null; then
    echo "  Needs flash_attn_3._C module"
    # Find the .so
    FA3_SO=""
    for _path in /usr/local/lib/python3.*/dist-packages/flash_attn_3/_C*.so \
                 /workspace/flash_attn_3/_C*.so \
                 /opt/conda/lib/python3.*/site-packages/flash_attn_3/_C*.so; do
        if [ -f "${_path}" ]; then
            FA3_SO="${_path}"
            echo "  Found .so: ${FA3_SO}"
            break
        fi
    done
    if [ -z "${FA3_SO}" ]; then
        echo "  WARNING: flash_attn_3._C.so not found"
    fi
fi

# ── Step 2: Create venv ──
echo ""
echo "[2/4] Creating isolated venv..."
if [ -d "${VENV_DIR}" ]; then
    echo "  Removing old venv..."
    rm -rf "${VENV_DIR}"
fi
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install -U pip setuptools wheel -q 2>&1 | tail -1

# ── Step 3: Try torch versions until FA3 works ──
echo ""
echo "[3/4] Finding compatible torch+cu124..."
FA3_WORKS=0

for TORCH_VER in 2.6.0 2.5.1 2.5.0 2.4.1; do
    echo ""
    echo "  Trying torch==${TORCH_VER}+cu124..."
    pip install --index-url https://download.pytorch.org/whl/cu124 "torch==${TORCH_VER}" -q 2>&1 | tail -1

    _installed=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAIL")
    if [[ "${_installed}" == "FAIL" ]]; then
        echo "  torch install failed, skipping"
        continue
    fi
    echo "  Installed: ${_installed}"

    # Set up flash_attn_3 package in venv if needed
    VENV_SITE="${VENV_DIR}/lib/python3.12/site-packages"
    if [ -n "${FA3_SO:-}" ]; then
        mkdir -p "${VENV_SITE}/flash_attn_3"
        ln -sf "${FA3_SO}" "${VENV_SITE}/flash_attn_3/"
        touch "${VENV_SITE}/flash_attn_3/__init__.py"
    fi

    # Test FA3 import via PYTHONPATH bridge
    _torch_lib="${VENV_SITE}/torch/lib"
    if LD_LIBRARY_PATH="${_torch_lib}:${LD_LIBRARY_PATH:-}" \
       PYTHONPATH="${FA3_DIR}:${PYTHONPATH:-}" \
       python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" 2>/dev/null; then
        echo "  ✓ FA3 works with torch==${TORCH_VER}+cu124!"
        FA3_WORKS=1
        break
    else
        echo "  ✗ FA3 failed with torch==${TORCH_VER}"
        # Clean up flash_attn_3 symlink for next attempt
        rm -rf "${VENV_SITE}/flash_attn_3"
    fi
done

if [ "${FA3_WORKS}" -eq 0 ]; then
    echo ""
    echo "FATAL: FA3 bridge failed with all torch versions."
    echo "The system's _C.abi3.so may be incompatible with all cu124 torch builds."
    exit 1
fi

# ── Step 4: Install remaining deps ──
echo ""
echo "[4/4] Installing deps..."
pip install zstandard sentencepiece numpy tqdm huggingface-hub "typing-extensions>=4.15" -q 2>&1 | tail -1
echo "  Done."

# ── Verify ──
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
PYTHONPATH="${FA3_DIR}:${PYTHONPATH:-}" python3 -c "
import torch
print(f'torch:   {torch.__version__}')
print(f'cuda:    {torch.version.cuda}')
print(f'gpus:    {torch.cuda.device_count()}')
from flash_attn_interface import flash_attn_func
print(f'FA3:     OK (flash_attn_interface)')
import zstandard
print(f'zstd:    {zstandard.__version__}')
"

echo ""
echo "============================================"
echo "  READY"
echo "============================================"
echo ""
echo "To run:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export PYTHONPATH=\"${FA3_DIR}:\${PYTHONPATH:-}\""
echo "  SEED=300 bash neural/2026-03-31_Rascal_III_SLOT/run.sh"
echo ""
echo "FA3_DIR=${FA3_DIR}"
