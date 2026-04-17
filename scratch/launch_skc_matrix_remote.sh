#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p /workspace/logs/cuda /workspace/logs/max_vram_10min

run_one() {
  local rid="$1"; shift
  echo "=== START ${rid} $(date -Is) ===" | tee -a /workspace/logs/cuda/${rid}.launcher.log
  env "$@" RUN_ID="$rid" bash /workspace/run_god_tier_skc.sh >> /workspace/logs/cuda/${rid}.txt 2>&1
  echo "=== END ${rid} $(date -Is) ===" | tee -a /workspace/logs/cuda/${rid}.launcher.log
}

common=(
  AUTO_TUNE=0
  SKIP_BUILD_SUBMISSION=1
  DIAGNOSTICS_ENABLED=1
  COMPILE_MODE=none
  DDP_FIND_UNUSED_PARAMETERS=1
  TRAIN_LOG_EVERY=1
  TRAIN_LOG_EVERY_FRACTION=0
  VAL_LOSS_EVERY=0
  VAL_LOSS_EVERY_FRACTION=0
  EXPORT_ALIGNED_TRAIN=0
  EXPORT_PROXY_EVAL=0
  TRAINING_DYNAMICS_ONLY=1
  ITERATIONS=200000
  MAX_WALLCLOCK_SECONDS=570
  RUN_TIMEOUT_SECONDS=2100
  TRAIN_BATCH_TOKENS=10128
  TRITON_ENGRAM_ENABLED=0
  SLIDING_EVAL=0
  FINAL_EVAL_SEQUENTIAL_CARRY=0
  RUNTIME_PATH_POLICY=legacy
  ADAM_WD=0.0
)

run_one E1v2_control_legacy_wd0_sc1_r35_eng1 "${common[@]}" SCALES_LR_MULT=1.0 RECURRENCE_START_FRACTION=0.35 BIGRAM_HASH_ENABLED=1
run_one E2v2_scales_legacy_wd0_sc3_r35_eng1 "${common[@]}" SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.35 BIGRAM_HASH_ENABLED=1
run_one E3v2_recur_legacy_wd0_sc3_r20_eng1  "${common[@]}" SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.20 BIGRAM_HASH_ENABLED=1
run_one E4v2_engoff_legacy_wd0_sc3_r20_eng0 "${common[@]}" SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.20 BIGRAM_HASH_ENABLED=0
run_one E5v2_strict_wd0_sc3_r20_eng1         "${common[@]}" RUNTIME_PATH_POLICY=strict SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.20 BIGRAM_HASH_ENABLED=1

echo "=== MATRIX DONE ==="
