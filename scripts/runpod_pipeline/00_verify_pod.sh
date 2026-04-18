#!/usr/bin/env bash
# Stage 0: Pod bring-up verification
# Checks torch 2.9.1+cu128, FA3, 8 H100s, disk space, commit SHA
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
REQUIRED_COMMIT="a33191f572430566b88c4d61badb0369e1e6f9a3"
LOG_DIR="${REPO_DIR}/runs"
LOG_FILE="${LOG_DIR}/00_verify_pod.log"
PYTHON="/opt/pg-venv/bin/python"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Stage 0: Pod verification === $(date)"

# 1. Python env: torch version, FA3, CUDA, GPU count
"${PYTHON}" - <<'PY'
import sys, torch
try:
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except ImportError as e:
    print(f"ERROR: FA3 import failed: {e}", flush=True); sys.exit(1)

expected_torch = "2.9.1+cu128"
expected_cuda  = "12.8"
if torch.__version__ != expected_torch:
    print(f"ERROR: torch {torch.__version__} != expected {expected_torch}"); sys.exit(1)
if torch.version.cuda != expected_cuda:
    print(f"ERROR: CUDA {torch.version.cuda} != expected {expected_cuda}"); sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: CUDA not available"); sys.exit(1)
ngpu = torch.cuda.device_count()
if ngpu != 8:
    print(f"ERROR: expected 8 GPUs, got {ngpu}"); sys.exit(1)
for i in range(ngpu):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {ngpu} GPUs: OK")
print("flash_attn_interface (FA3): OK")
PY

# 2. nvidia-smi sanity
echo ""
echo "nvidia-smi:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 3. Disk space >= 80 GB
FREE_GB=$(df -BG /workspace --output=avail 2>/dev/null | tail -1 | tr -d 'G ')
if [ "${FREE_GB:-0}" -lt 80 ]; then
    echo "ERROR: /workspace has ${FREE_GB}G free; need >= 80G" >&2
    exit 1
fi
echo ""
echo "disk: ${FREE_GB}G free in /workspace: OK"

# 4. Commit SHA check
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "ERROR: ${REPO_DIR}/.git not found — repo not cloned yet" >&2
    exit 1
fi
cd "${REPO_DIR}"
CURRENT_COMMIT=$(git rev-parse HEAD)
if ! git merge-base --is-ancestor "${REQUIRED_COMMIT}" HEAD 2>/dev/null; then
    echo "ERROR: HEAD (${CURRENT_COMMIT}) does not include required commit ${REQUIRED_COMMIT}" >&2
    echo "  Run: git fetch origin && git checkout submission/pr1610-corrector" >&2
    exit 1
fi
echo "git HEAD: ${CURRENT_COMMIT}"
echo "required SHA ${REQUIRED_COMMIT}: present in history: OK"

# 4b. Optional exact-SHA pin for Session launches
if [ -n "${EXPECTED_SHA:-}" ]; then
    if [ "${CURRENT_COMMIT}" != "${EXPECTED_SHA}" ]; then
        echo "ERROR: HEAD=${CURRENT_COMMIT} != EXPECTED_SHA=${EXPECTED_SHA}" >&2
        echo "  Session launches must run on the pinned SHA." >&2
        exit 1
    fi
    echo "exact SHA pin: ${CURRENT_COMMIT} == EXPECTED_SHA: OK"
fi

# 5. Train script exists
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"
[ -f "${TRAIN_SCRIPT}" ] || { echo "ERROR: ${TRAIN_SCRIPT} not found"; exit 1; }
echo "train_gpt.py: present: OK"

echo ""
echo "00_verify_pod: PASS"
