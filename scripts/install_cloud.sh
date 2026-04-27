#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

before_json="$(mktemp)"
after_json="$(mktemp)"
final_json="$(mktemp)"
cleanup() {
  rm -f "$before_json" "$after_json" "$final_json"
}
trap cleanup EXIT

echo "[install_cloud] Inspecting the existing torch stack"
"$PYTHON" scripts/check_frontier_env.py --allow-missing-flash-attn --json > "$before_json"

echo "[install_cloud] Installing common cloud dependencies without torch"
"$PYTHON" -m pip install -r requirements-cloud.txt

echo "[install_cloud] Verifying that torch did not change"
"$PYTHON" scripts/check_frontier_env.py --allow-missing-flash-attn --json > "$after_json"
"$PYTHON" - "$before_json" "$after_json" <<'PY'
import json
import sys

before = json.load(open(sys.argv[1], "r", encoding="utf-8"))
after = json.load(open(sys.argv[2], "r", encoding="utf-8"))
changed = []
for key in ("torch_version", "torch_cuda_version", "torch_path"):
    if before.get(key) != after.get(key):
        changed.append((key, before.get(key), after.get(key)))
if changed:
    print("STOP: the torch stack changed during requirements-cloud.txt installation.", file=sys.stderr)
    for key, before_value, after_value in changed:
        print(f"  {key}: before={before_value!r} after={after_value!r}", file=sys.stderr)
    print("Do not continue on this pod. Recreate it and avoid reinstalling torch on top of the image.", file=sys.stderr)
    raise SystemExit(1)
print("Torch stack unchanged after requirements-cloud.txt install.")
PY

if "$PYTHON" -c "import flash_attn_interface" >/dev/null 2>&1; then
  echo "[install_cloud] flash_attn_interface already imports"
else
  echo "[install_cloud] Installing flash-attn against the existing torch stack"
  "$PYTHON" -m pip install flash-attn --no-build-isolation
fi

echo "[install_cloud] Verifying that torch still matches the original image stack"
"$PYTHON" scripts/check_frontier_env.py --allow-missing-flash-attn --json > "$final_json"
"$PYTHON" - "$before_json" "$final_json" <<'PY'
import json
import sys

before = json.load(open(sys.argv[1], "r", encoding="utf-8"))
after = json.load(open(sys.argv[2], "r", encoding="utf-8"))
changed = []
for key in ("torch_version", "torch_cuda_version", "torch_path"):
    if before.get(key) != after.get(key):
        changed.append((key, before.get(key), after.get(key)))
if changed:
    print("STOP: the torch stack changed during flash-attn installation.", file=sys.stderr)
    for key, before_value, after_value in changed:
        print(f"  {key}: before={before_value!r} after={after_value!r}", file=sys.stderr)
    print("Do not continue on this pod. Recreate it and keep the image torch unchanged.", file=sys.stderr)
    raise SystemExit(1)
print("Torch stack still matches the original image after flash-attn installation.")
PY

echo "[install_cloud] Final frontier readiness check"
"$PYTHON" scripts/check_frontier_env.py
