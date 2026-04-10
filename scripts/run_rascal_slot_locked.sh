#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-fa3wheel}"
elif [[ -f "${VENV_DIR:-/workspace/venv_cu124}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR:-/workspace/venv_cu124}/bin/activate"
fi

TORCH_LIB="$(python - <<'PYEOF'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

export COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}"
# Preserve caller override if provided; otherwise leave torch default behavior.
if [[ -n "${TORCHDYNAMO_OPTIMIZE_DDP:-}" ]]; then
  export TORCHDYNAMO_OPTIMIZE_DDP
fi
# Strict default: fail fast on compiler issues (do not silently fall back to slow eager).
export TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export SEED="${SEED:-300}"

bash "${REPO_ROOT}/scripts/verify_cu124_fa3_env.sh"

exec bash "${REPO_ROOT}/neural/2026-03-31_Rascal_III_SLOT/run.sh"
