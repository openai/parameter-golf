#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HOPPER_DIR="${REPO_ROOT}/flash-attention/hopper"

if [[ ! -d "${HOPPER_DIR}" ]]; then
  echo "FATAL: missing ${HOPPER_DIR}"
  exit 1
fi

python3 - <<'PYEOF'
import torch
tv = torch.__version__
cv = torch.version.cuda or ""
assert tv.startswith("2.4.1"), f"wrong torch: {tv}"
assert cv.startswith("12.4"), f"wrong cuda: {cv}"
print(f"torch={tv} cuda={cv}")
PYEOF

cd "${HOPPER_DIR}"

# Historical known-good FA3 trim profile (used across prior RunPod/Vast workflows).
# Keep this conservative and stable: do not add extra disable flags here.
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export MAX_JOBS="${MAX_JOBS:-4}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "${TMPDIR}"

pip install -U ninja packaging
pip install -e . --no-build-isolation

python3 - <<'PYEOF'
import importlib, os, site
importlib.import_module("flash_attn_3._C")
import flash_attn_interface
print(f"flash_attn_interface={flash_attn_interface.__file__}")

cfg_src = os.path.join(os.path.dirname(flash_attn_interface.__file__), "flash_attn_config.py")
sp = site.getsitepackages()[0]
cfg_dst = os.path.join(sp, "flash_attn_config.py")
if os.path.isfile(cfg_src) and not os.path.exists(cfg_dst):
    os.symlink(cfg_src, cfg_dst)
    print(f"linked {cfg_dst} -> {cfg_src}")
print("FA3 OK")
PYEOF

echo "READY: trimmed FA3 installed for H100/cu124."
