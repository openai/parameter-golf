#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VERIFY_DATA="${VERIFY_DATA:-1}"

if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-fa3wheel}" >/dev/null 2>&1 || true
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

python - <<'PYEOF'
import glob
import importlib
import os
import torch

assert torch.__version__.startswith("2.4.1+cu124"), f"wrong torch: {torch.__version__}"
assert str(torch.version.cuda).startswith("12.4"), f"wrong cuda: {torch.version.cuda}"
importlib.import_module("flash_attn_3._C")
from flash_attn_interface import flash_attn_func  # noqa: F401

verify_data = os.environ.get("VERIFY_DATA", "1") != "0"
train = []
val = []
if verify_data:
    tokenizer = "./data/tokenizers/fineweb_1024_bpe.model"
    assert os.path.isfile(tokenizer), f"missing tokenizer: {tokenizer}"

    train = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    val = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    assert len(train) >= 1, "missing train shards"
    assert len(val) >= 1, "missing val shards"

print("VERIFY_OK")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpus={torch.cuda.device_count()}")
if verify_data:
    print(f"train_shards={len(train)} val_shards={len(val)}")
else:
    print("data_checks=skipped")
PYEOF
