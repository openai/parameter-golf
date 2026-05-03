#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export PYTHONUNBUFFERED=1
set +e
python3 base_train_gpt.py > results/train.log 2>&1
rc=$?
set -e
printf '{"returncode": %s, "finished_at": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/runner_status.json
exit $rc
