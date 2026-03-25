#!/usr/bin/env bash
set -euo pipefail

python3 -m py_compile train_gpt.py
python3 - <<'PY'
import json
from pathlib import Path
json.loads(Path("submission.json").read_text())
print("submission.json ok")
PY
bash -n run_8xh100_one_shot.sh
bash -n bootstrap_remote_env.sh
bash -n preflight_remote_strict.sh
bash -n collect_artifacts.sh
echo "local prep ok"
