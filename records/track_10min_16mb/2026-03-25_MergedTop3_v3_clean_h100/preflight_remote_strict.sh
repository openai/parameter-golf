#!/usr/bin/env bash
set -euo pipefail

export DATA_PATH="${DATA_PATH:-../../../data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-../../../data/tokenizers/fineweb_1024_bpe.model}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export REMOTE_VENV_DIR="${REMOTE_VENV_DIR:-/workspace/.venvs/parameter-golf-20260325}"
export REQUIRE_FLASH_ATTN="${REQUIRE_FLASH_ATTN:-1}"

bash bootstrap_remote_env.sh
source "${REMOTE_VENV_DIR}/bin/activate"

gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "${gpu_count}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "preflight failed: expected at least ${NPROC_PER_NODE} GPUs, found ${gpu_count}" >&2
  exit 1
fi

python - <<'PY'
import glob
import os
from pathlib import Path

data_path = Path(os.environ["DATA_PATH"]).resolve()
tokenizer_path = Path(os.environ["TOKENIZER_PATH"]).resolve()
train_files = sorted(glob.glob(str(data_path / "fineweb_train_*.bin")))
val_files = sorted(glob.glob(str(data_path / "fineweb_val_*.bin")))
if not tokenizer_path.exists():
    raise SystemExit(f"preflight failed: tokenizer missing: {tokenizer_path}")
if not train_files:
    raise SystemExit(f"preflight failed: no training shards under {data_path}")
if not val_files:
    raise SystemExit(f"preflight failed: no validation shards under {data_path}")
print(f"tokenizer={tokenizer_path}")
print(f"train_shards={len(train_files)}")
print(f"val_shards={len(val_files)}")
PY

python stability_probe.py --checkpoint-dir .stability_probe
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" stability_probe.py --checkpoint-dir .stability_probe
df -h .
