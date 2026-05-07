#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# NON-CLAIMING benchmark command templates
# ==========================================================
# This script provides concrete templates for two phases:
#   1) local smoke path (sanity only, non-claiming)
#   2) target-GPU path template (for real benchmark execution)
#
# IMPORTANT:
# - Running this script as-is does NOT create a benchmark claim.
# - Claims remain disabled until evidence files are produced and verified.
# - Do not label outputs as leaderboard-ready without full benchmark evidence.
# ==========================================================

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_DIR="${RUN_DIR}/evidence"
mkdir -p "${EVIDENCE_DIR}"

REPO_ROOT_DEFAULT="/Users/ever/Documents/GitHub/parameter-golf"
REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"

PHASE="${1:-help}"  # help | local-smoke | target-gpu-template

log_info() {
  printf '[info] %s\n' "$*"
}

write_environment_snapshot() {
  cat > "${EVIDENCE_DIR}/environment.json" <<JSON
{
  "captured_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "non_claiming": true,
  "repo_root": "${REPO_ROOT}",
  "host": "$(hostname)",
  "uname": "$(uname -a | sed 's/"/\\"/g')",
  "python_version": "$(python3 --version 2>/dev/null | sed 's/"/\\"/g' || true)",
  "phase": "${PHASE}"
}
JSON
}

local_smoke() {
  log_info "Running local smoke template (non-claiming sanity path)."
  log_info "Repo root: ${REPO_ROOT}"

  cd "${REPO_ROOT}"

  # Optional lightweight setup check
  {
    echo "[smoke] timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[smoke] pwd: $(pwd)"
    echo "[smoke] python: $(python3 --version 2>&1 || true)"
    echo "[smoke] note: this is not a benchmark claim run"
  } | tee "${EVIDENCE_DIR}/train.log"

  # Example smoke command from repo docs (short local sanity only).
  # This checks codepath health, not challenge competitiveness.
  {
    echo "[smoke] command: RUN_ID=vita_smoke ITERATIONS=200 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py"
    echo "[smoke] (command intentionally not auto-executed here; uncomment if desired)"
    # RUN_ID=vita_smoke ITERATIONS=200 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py
  } | tee -a "${EVIDENCE_DIR}/eval.log"

  write_environment_snapshot

  cat > "${EVIDENCE_DIR}/artifact_sizes.json" <<JSON
{
  "captured_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "non_claiming": true,
  "artifact_size_verified": false,
  "note": "Smoke path only. No challenge-grade artifact accounting yet."
}
JSON

  cat > "${EVIDENCE_DIR}/submission.json" <<JSON
{
  "status": "non-claiming-smoke",
  "benchmark_ready": false,
  "note": "Placeholder only; replace after real benchmark execution."
}
JSON

  log_info "Local smoke template completed. Evidence placeholders written to ${EVIDENCE_DIR}."
}

target_gpu_template() {
  log_info "Printing target-GPU benchmark template commands (non-executing)."
  log_info "Use these as a runbook on target hardware."

  cat <<'TEMPLATE' | tee "${EVIDENCE_DIR}/target_gpu_template.txt"
# ==========================================================
# TARGET-GPU BENCHMARK TEMPLATE (NON-CLAIMING UNTIL EXECUTED)
# ==========================================================
# 1) Prepare benchmark environment
cd /workspace
# git clone https://github.com/<your-fork>/parameter-golf.git
cd parameter-golf

# 2) Download challenge dataset/tokenizer cache (example)
python3 data/cached_challenge_fineweb.py --variant sp1024

# 3) Run benchmark training path (example baseline-style command)
# NOTE: Replace with your adapted candidate command once mapping is implemented.
RUN_ID=vita_benchmark_attempt \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train.log

# 4) Capture exact final metric lines
# grep final_int8_zlib_roundtrip_exact train.log

# 5) Capture artifact-size accounting from produced artifacts
# (e.g., final model artifact bytes + code bytes)

# 6) Copy evidence files into run scaffold
# evidence/train.log
# evidence/eval.log (or metric-extracted log)
# evidence/submission.json
# evidence/artifact_sizes.json
# evidence/environment.json

# 7) Only after all evidence + checks: evaluate claim eligibility.
# ==========================================================
TEMPLATE

  write_environment_snapshot
  log_info "Template written: ${EVIDENCE_DIR}/target_gpu_template.txt"
}

case "${PHASE}" in
  help)
    cat <<EOF
Usage:
  $(basename "$0") local-smoke
  $(basename "$0") target-gpu-template

Env overrides:
  REPO_ROOT=/path/to/parameter-golf

Notes:
  - Both modes are NON-CLAIMING by default.
  - local-smoke writes sanity placeholders.
  - target-gpu-template writes command templates only.
EOF
    ;;
  local-smoke)
    local_smoke
    ;;
  target-gpu-template)
    target_gpu_template
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
