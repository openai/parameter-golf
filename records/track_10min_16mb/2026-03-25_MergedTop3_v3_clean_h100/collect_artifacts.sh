#!/usr/bin/env bash
set -euo pipefail

tar -czf mergedtop3_one_shot_artifacts.tgz \
  README.md \
  submission.json \
  requirements.txt \
  requirements-remote.txt \
  train_gpt.py \
  bootstrap_remote_env.sh \
  preflight_remote_strict.sh \
  stability_probe.py \
  run_8xh100_one_shot.sh \
  collect_artifacts.sh \
  preflight_local_prep.sh \
  logs \
  train_seed*.log \
  final_model*.pt* 2>/dev/null || true

echo "wrote mergedtop3_one_shot_artifacts.tgz"
