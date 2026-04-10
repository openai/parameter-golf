#!/bin/bash
# Symlink the installed mamba_ssm Mamba-3 Triton files to the repo copies
# under triton_kernels/, so edits in the repo take effect immediately
# without having to re-run setup_mamba3.sh.
#
# Run this ON THE POD after setup_mamba3.sh has installed mamba_ssm once.
#
# Usage:
#   bash triton_kernels/setup_editable_mamba3.sh
#
# Idempotent: safe to run multiple times.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
REPO_KERNELS="${REPO_DIR}/triton_kernels"

if ! python3 -c "import mamba_ssm" 2>/dev/null; then
    echo "ERROR: mamba_ssm not importable. Run 'bash setup_mamba3.sh' first." >&2
    exit 1
fi

PKG=$(python3 -c "import mamba_ssm; print(mamba_ssm.__path__[0])")
TARGET_DIR="${PKG}/ops/triton/mamba3"

echo "==> mamba_ssm package:  ${PKG}"
echo "==> repo kernels:       ${REPO_KERNELS}"
echo

for f in mamba3_siso_bwd.py mamba3_siso_combined.py; do
    src="${REPO_KERNELS}/${f}"
    dst="${TARGET_DIR}/${f}"

    if [[ ! -f "${src}" ]]; then
        echo "ERROR: ${src} missing. Copy it from ${dst} first." >&2
        exit 1
    fi

    rm -f "${dst}"
    ln -s "${src}" "${dst}"
    echo "  linked ${dst} -> ${src}"
done

echo
echo "==> Verifying import..."
python3 -c "
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import mamba3_siso_bwd_kernel_dqkv
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
print('  Mamba-3 + bwd kernels + combined OK')
"
echo "==> Done. Edits to ${REPO_KERNELS}/ now take effect immediately."
