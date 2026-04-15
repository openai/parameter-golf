#!/bin/bash
# Install mamba-ssm and patch in Mamba-3 files (not yet included in the wheel/sdist)
set -euo pipefail

# Install mamba-ssm from source (gets dependencies like transformers, einops, tilelang)
pip install git+https://github.com/state-spaces/mamba.git --break-system-packages --no-build-isolation

# Copy Mamba-3 files that setup.py doesn't package yet
PKG=$(python3 -c "import mamba_ssm; print(mamba_ssm.__path__[0])")
rm -rf /tmp/mamba
git clone --depth 1 https://github.com/state-spaces/mamba.git /tmp/mamba
cp /tmp/mamba/mamba_ssm/modules/mamba3.py "$PKG/modules/"
# angle_cumsum.py was merged into the mamba3/ subdir in newer repo versions — skip it
mkdir -p "$PKG/ops/triton/mamba3"
cp /tmp/mamba/mamba_ssm/ops/triton/mamba3/*.py "$PKG/ops/triton/mamba3/"
rm -rf /tmp/mamba

python3 -c "from mamba_ssm.modules.mamba3 import Mamba3; print('Mamba-3 OK')"
