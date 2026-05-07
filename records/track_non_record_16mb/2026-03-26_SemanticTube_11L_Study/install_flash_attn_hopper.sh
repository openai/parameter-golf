#!/usr/bin/env bash
set -euo pipefail

# Build and cache a Hopper-only flash-attn wheel for true FA3 kernels.
# This is for H100-class pods. It avoids repeated generic multi-arch builds.

if python3 - <<'PY' >/dev/null 2>&1
mods = ("flash_attn_interface", "hopper.flash_attn_interface")
for mod in mods:
    try:
        __import__(mod, fromlist=["flash_attn_func"])
        raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
then
  echo "flash-attn Hopper kernels already importable"
  exit 0
fi

FLASH_ATTN_REPO_URL="${FLASH_ATTN_REPO_URL:-https://github.com/Dao-AILab/flash-attention.git}"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-main}"
FLASH_ATTN_BUILD_ROOT="${FLASH_ATTN_BUILD_ROOT:-/tmp/flash-attention-hopper}"
FLASH_ATTN_WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-$HOME/.cache/flash-attn-hopper-wheels}"
FLASH_ATTN_ARCH_LIST="${FLASH_ATTN_ARCH_LIST:-9.0}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-4}"

export MAX_JOBS
export NVCC_THREADS
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$FLASH_ATTN_ARCH_LIST}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-TRUE}"
export FLASH_ATTENTION_DISABLE_FP16="${FLASH_ATTENTION_DISABLE_FP16:-TRUE}"
export FLASH_ATTENTION_DISABLE_FP8="${FLASH_ATTENTION_DISABLE_FP8:-TRUE}"
export FLASH_ATTENTION_DISABLE_PAGEDKV="${FLASH_ATTENTION_DISABLE_PAGEDKV:-TRUE}"
export FLASH_ATTENTION_DISABLE_APPENDKV="${FLASH_ATTENTION_DISABLE_APPENDKV:-TRUE}"
export FLASH_ATTENTION_DISABLE_LOCAL="${FLASH_ATTENTION_DISABLE_LOCAL:-TRUE}"
export FLASH_ATTENTION_DISABLE_SOFTCAP="${FLASH_ATTENTION_DISABLE_SOFTCAP:-TRUE}"
export FLASH_ATTENTION_DISABLE_CLUSTER="${FLASH_ATTENTION_DISABLE_CLUSTER:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIM128="${FLASH_ATTENTION_DISABLE_HDIM128:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIM192="${FLASH_ATTENTION_DISABLE_HDIM192:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIM256="${FLASH_ATTENTION_DISABLE_HDIM256:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF64="${FLASH_ATTENTION_DISABLE_HDIMDIFF64:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF192="${FLASH_ATTENTION_DISABLE_HDIMDIFF192:-TRUE}"

mkdir -p "$FLASH_ATTN_WHEEL_DIR"

LATEST_WHEEL="$(ls -t "$FLASH_ATTN_WHEEL_DIR"/flash_attn_3-*.whl 2>/dev/null | head -1 || true)"
if [ -n "$LATEST_WHEEL" ]; then
  python3 -m pip install --no-deps --force-reinstall "$LATEST_WHEEL"
  if python3 - <<'PY' >/dev/null 2>&1
mods = ("flash_attn_interface", "hopper.flash_attn_interface")
for mod in mods:
    try:
        __import__(mod, fromlist=["flash_attn_func"])
        raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
  then
    echo "flash-attn Hopper kernels installed from cached wheel: $(basename "$LATEST_WHEEL")"
    exit 0
  fi
fi

if [ ! -d "$FLASH_ATTN_BUILD_ROOT/.git" ]; then
  rm -rf "$FLASH_ATTN_BUILD_ROOT"
  git clone --depth 1 --branch "$FLASH_ATTN_REF" "$FLASH_ATTN_REPO_URL" "$FLASH_ATTN_BUILD_ROOT"
else
  git -C "$FLASH_ATTN_BUILD_ROOT" fetch --depth 1 origin "$FLASH_ATTN_REF"
  git -C "$FLASH_ATTN_BUILD_ROOT" checkout -f FETCH_HEAD
fi

cd "$FLASH_ATTN_BUILD_ROOT/hopper"
rm -rf build dist
python3 setup.py bdist_wheel

WHEEL_PATH="$(ls -t dist/*.whl | head -1)"
cp "$WHEEL_PATH" "$FLASH_ATTN_WHEEL_DIR/"
python3 -m pip install --no-deps --force-reinstall "$FLASH_ATTN_WHEEL_DIR/$(basename "$WHEEL_PATH")"

python3 - <<'PY'
mods = ("flash_attn_interface", "hopper.flash_attn_interface")
for mod in mods:
    try:
        __import__(mod, fromlist=["flash_attn_func"])
        print(f"flash-attn Hopper install OK via {mod}")
        raise SystemExit(0)
    except Exception:
        pass
raise SystemExit("flash-attn Hopper install failed: FA3 module not importable")
PY
