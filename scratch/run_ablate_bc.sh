#!/usr/bin/env bash
set -euo pipefail
cd /workspace
run_one() {
  local rid="$1"; shift
  echo "=== START ${rid} $(date -Is) ===" | tee -a /workspace/logs/cuda/${rid}.launcher.log
  env "$@" RUN_ID="$rid" bash /workspace/run_god_tier_skc.sh >> /workspace/logs/cuda/${rid}.txt 2>&1
  echo "=== END ${rid} $(date -Is) ===" | tee -a /workspace/logs/cuda/${rid}.launcher.log
}
common=(AUTO_TUNE=0 DIAGNOSTICS_ENABLED=1 COMPILE_MODE=none DDP_FIND_UNUSED_PARAMETERS=1 TRAIN_LOG_EVERY=1 TRAIN_LOG_EVERY_FRACTION=0 VAL_LOSS_EVERY=50 VAL_LOSS_EVERY_FRACTION=0 EXPORT_ALIGNED_TRAIN=0 EXPORT_PROXY_EVAL=0 RUNTIME_PATH_POLICY=legacy TRAIN_BATCH_TOKENS=10128 COMPILER_WARMUP_BATCH_TOKENS=8192)
run_one ablateB_legacy_wd000_r35 "${common[@]}" ADAM_WD=0.0 RECURRENCE_START_FRACTION=0.35
run_one ablateC_legacy_wd000_r65 "${common[@]}" ADAM_WD=0.0 RECURRENCE_START_FRACTION=0.65
