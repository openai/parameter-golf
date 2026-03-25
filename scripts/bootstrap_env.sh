#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v "${PYTHON_BIN:-python3}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python was not found. Install python3 and rerun."
  exit 1
fi

IS_COLAB="0"
if [[ -d "/content" || -n "${COLAB_RELEASE_TAG:-}" || -n "${COLAB_GPU:-}" ]]; then
  IS_COLAB="1"
fi

if [[ "$IS_COLAB" == "1" ]]; then
  CACHE_ROOT="${CACHE_ROOT:-/content/.cache/parameter-golf}"
else
  CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache/parameter-golf}"
fi

mkdir -p \
  "$CACHE_ROOT/huggingface" \
  "$CACHE_ROOT/hf_datasets" \
  "$CACHE_ROOT/transformers" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/pip"

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

CACHE_ENV_FILE="$ROOT_DIR/.cache_env.sh"
cat > "$CACHE_ENV_FILE" <<EOF
export HF_HOME="$HF_HOME"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export TORCH_HOME="$TORCH_HOME"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
EOF

echo "=== Parameter Golf Environment Report ==="
echo "timestamp_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "repo_root: $ROOT_DIR"
echo "is_colab: $IS_COLAB"
echo "python_bin: $(command -v "$PYTHON_BIN")"
echo "python_version: $("$PYTHON_BIN" --version 2>&1)"

if [[ -f /etc/os-release ]]; then
  echo "os_release: $(grep '^PRETTY_NAME=' /etc/os-release | head -n1 | cut -d= -f2- | tr -d '"')"
else
  echo "os_release: $(uname -a)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_report:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "gpu_report: nvidia-smi not found"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc_version: $(nvcc --version | tail -n1)"
else
  echo "nvcc_version: not found"
fi

set +e
TORCH_REPORT="$(
"$PYTHON_BIN" - <<'PY'
import torch
print(f"torch_version: {torch.__version__}")
print(f"torch_cuda_build: {torch.version.cuda}")
print(f"torch_cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_gpu_count: {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"torch_gpu0_name: {props.name}")
    print(f"torch_gpu0_mem_mib: {props.total_memory // (1024 * 1024)}")
PY
)"
TORCH_STATUS=$?
set -e

if [[ "$TORCH_STATUS" -eq 0 ]]; then
  TORCH_IMPORTABLE="1"
  echo "$TORCH_REPORT"
else
  TORCH_IMPORTABLE="0"
  echo "torch_importable: false"
fi

if [[ "${INSTALL_DEPS:-1}" == "0" ]]; then
  echo "dependency_install: skipped (INSTALL_DEPS=0)"
  echo "cache_exports_file: $CACHE_ENV_FILE"
  exit 0
fi

REQ_FILE="$ROOT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: requirements.txt was not found at $REQ_FILE"
  exit 1
fi

TMP_REQ="$(mktemp)"
if [[ "$TORCH_IMPORTABLE" == "1" ]]; then
  grep -Evi '^torch[[:space:]]*$' "$REQ_FILE" > "$TMP_REQ"
  echo "dependency_install: torch import succeeded, skipping torch reinstall"
else
  cp "$REQ_FILE" "$TMP_REQ"
  echo "dependency_install: torch import failed, installing full requirements (including torch)"
fi

echo "pip_upgrade: start"
"$PYTHON_BIN" -m pip install --upgrade pip
echo "pip_install: start"
if [[ -s "$TMP_REQ" ]]; then
  "$PYTHON_BIN" -m pip install -r "$TMP_REQ"
fi
rm -f "$TMP_REQ"

echo "cache_exports_file: $CACHE_ENV_FILE"
echo "next_step: source .cache_env.sh"
