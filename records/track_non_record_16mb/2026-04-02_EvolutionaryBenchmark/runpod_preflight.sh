#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/runpod_common.sh"

echo "repo_root=${ROOT}"
echo "python_bin=${EVO_PYTHON_BIN}"
echo "enwik8_path=${EVO_ENWIK8_PATH}"
echo "output_dir=${EVO_OUTPUT_DIR}"
echo "log_dir=${EVO_LOG_DIR}"
echo "gpus=${EVO_GPUS}"
echo "max_workers=${EVO_MAX_WORKERS}"

"${EVO_PYTHON_BIN}" --version
require_benchmark_python_modules
check_sentencepiece_tokenizer "sp_bpe_1024" 0
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

if [[ ! -f "${EVO_ENWIK8_PATH}" ]]; then
  echo "missing enwik8 at ${EVO_ENWIK8_PATH}" >&2
  exit 1
fi
ls -lh "${EVO_ENWIK8_PATH}"

"${EVO_PYTHON_BIN}" "${ROOT}/tools/generate_evolutionary_matrix.py" --format names | wc -l
"${EVO_PYTHON_BIN}" "${ROOT}/tools/run_evolutionary_matrix.py" \
  --python-bin "${EVO_PYTHON_BIN}" \
  --script-path "${ROOT}/tools/evolutionary_benchmark.py" \
  --output-dir "${EVO_OUTPUT_DIR}" \
  --log-dir "${EVO_LOG_DIR}" \
  --enwik8-path "${EVO_ENWIK8_PATH}" \
  --gpus "${EVO_GPUS}" \
  --max-workers "${EVO_MAX_WORKERS}" \
  --stages 0-throughput \
  --include-tags core \
  --dry-run | sed -n '1,120p'
