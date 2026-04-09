#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"

EVO_PYTHON_BIN="${EVO_PYTHON_BIN:-python3}"
EVO_ENWIK8_PATH="${EVO_ENWIK8_PATH:-/workspace/data/enwik8}"
EVO_OUTPUT_DIR="${EVO_OUTPUT_DIR:-${DIR}/runs}"
EVO_LOG_DIR="${EVO_LOG_DIR:-${EVO_OUTPUT_DIR}/logs}"
EVO_GPUS="${EVO_GPUS:-auto}"
EVO_MAX_WORKERS="${EVO_MAX_WORKERS:-0}"
EVO_SKIP_EXISTING="${EVO_SKIP_EXISTING:-1}"
EVO_FAIL_FAST="${EVO_FAIL_FAST:-0}"

require_python_modules() {
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  "${EVO_PYTHON_BIN}" - "$@" <<'PY'
from __future__ import annotations

import importlib.util
import sys

modules = sys.argv[1:]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing python modules: {', '.join(missing)}")
print(f"python_modules_ok={','.join(modules)}")
PY
}

require_benchmark_python_modules() {
  require_python_modules numpy torch
}

check_sentencepiece_tokenizer() {
  local tokenizer_name="${1:-sp_bpe_1024}"
  local strict="${2:-0}"
  "${EVO_PYTHON_BIN}" - "${ROOT}" "${tokenizer_name}" "${strict}" <<'PY'
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tokenizer_name = sys.argv[2]
strict = sys.argv[3] == "1"

specs_path = root / "data" / "tokenizer_specs.json"
if importlib.util.find_spec("sentencepiece") is None:
    message = f"missing python module: sentencepiece (required for tokenizer {tokenizer_name})"
    if strict:
        raise SystemExit(message)
    print(f"sentencepiece_status=missing ({message})")
    raise SystemExit(0)

if not specs_path.exists():
    message = f"missing tokenizer specs: {specs_path}"
    if strict:
        raise SystemExit(message)
    print(f"sentencepiece_status=missing ({message})")
    raise SystemExit(0)

payload = json.loads(specs_path.read_text(encoding="utf-8"))
tokenizers = {
    str(entry["name"]): entry
    for entry in payload.get("tokenizers", [])
    if isinstance(entry, dict) and entry.get("name")
}
spec = tokenizers.get(tokenizer_name)
if spec is None:
    raise SystemExit(f"unknown tokenizer name: {tokenizer_name}")

model_path_value = spec.get("model_path")
if not model_path_value:
    raise SystemExit(f"tokenizer spec {tokenizer_name} is missing model_path")

model_path = Path(model_path_value)
if not model_path.is_absolute():
    model_path = (root / model_path).resolve()
if not model_path.exists():
    raise SystemExit(f"missing tokenizer model: {model_path}")

print(f"sentencepiece_status=ok tokenizer={tokenizer_name} model={model_path}")
PY
}

require_sentencepiece_tokenizer() {
  local tokenizer_name="${1:-sp_bpe_1024}"
  check_sentencepiece_tokenizer "${tokenizer_name}" 1
}

evo_queue() {
  local extra_args=("$@")
  local cmd=(
    "${EVO_PYTHON_BIN}"
    "${ROOT}/tools/run_evolutionary_matrix.py"
    --python-bin "${EVO_PYTHON_BIN}"
    --script-path "${ROOT}/tools/evolutionary_benchmark.py"
    --output-dir "${EVO_OUTPUT_DIR}"
    --log-dir "${EVO_LOG_DIR}"
    --enwik8-path "${EVO_ENWIK8_PATH}"
    --gpus "${EVO_GPUS}"
    --max-workers "${EVO_MAX_WORKERS}"
  )
  if [[ "${EVO_SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi
  if [[ "${EVO_FAIL_FAST}" == "1" ]]; then
    cmd+=(--fail-fast)
  fi
  cmd+=("${extra_args[@]}")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

evo_summary() {
  local section="${1:-all}"
  "${EVO_PYTHON_BIN}" \
    "${ROOT}/tools/summarize_evolutionary_runs.py" \
    "${EVO_OUTPUT_DIR}" \
    --section "${section}"
}
