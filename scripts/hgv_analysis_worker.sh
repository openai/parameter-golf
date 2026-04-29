#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HGV_HOST="${HGV_HOST:-192.168.179.163}"
HGV_USER="${HGV_USER:-hgv}"
HGV_PASS="${HGV_PASS:-}"
INTERVAL_SECS="${INTERVAL_SECS:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY_SECS="${RETRY_DELAY_SECS:-4}"
SSH_OPTS=(-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o ConnectTimeout=8)

if [[ -z "$HGV_PASS" ]]; then
  echo "HGV_PASS is required. Example: HGV_PASS='***' bash scripts/hgv_analysis_worker.sh --once"
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass is required for password-based non-interactive runs."
  exit 1
fi

OUT_DIR="$REPO_ROOT/logs/sweep"
OUT_LATEST="$OUT_DIR/hgv_signal_latest.txt"
OUT_HISTORY="$OUT_DIR/hgv_signal_history.log"
OUT_RECO="$OUT_DIR/hgv_recommendations.tsv"
OUT_SWEEP="$OUT_DIR/hgv_recommendations_sweep.tsv"
OUT_HEALTH="$OUT_DIR/hgv_analysis_health.json"
mkdir -p "$OUT_DIR"

ANALYZER_SRC="$REPO_ROOT/scripts/hgv_smart_crunch.py"
LOCK_DIR="/tmp/hgv_analysis_worker.lock"

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  if [[ -f "$LOCK_DIR/pid" ]]; then
    local old_pid
    old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[hgv-analysis] lock held by pid=$old_pid; exiting"
      exit 0
    fi
  fi
  rm -rf "$LOCK_DIR" 2>/dev/null || true
  mkdir "$LOCK_DIR"
  echo "$$" > "$LOCK_DIR/pid"
}

release_lock() {
  rm -rf "$LOCK_DIR" 2>/dev/null || true
}

write_health() {
  local status="$1" mode="$2" detail="$3"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "$OUT_HEALTH.tmp" <<JSON
{
  "timestamp": "$ts",
  "status": "$status",
  "mode": "$mode",
  "detail": "$detail",
  "interval_secs": $INTERVAL_SECS,
  "max_retries": $MAX_RETRIES,
  "host": "$HGV_USER@$HGV_HOST"
}
JSON
  mv "$OUT_HEALTH.tmp" "$OUT_HEALTH"
}

retry_run() {
  local n=1
  until "$@"; do
    if [[ "$n" -ge "$MAX_RETRIES" ]]; then
      return 1
    fi
    n=$((n + 1))
    sleep "$RETRY_DELAY_SECS"
  done
}

validate_outputs() {
  local signal_file="$1" reco_file="$2" sweep_file="$3"

  [[ -s "$signal_file" ]] || return 1
  [[ -s "$reco_file" ]] || return 1
  [[ -s "$sweep_file" ]] || return 1

  grep -q '^=== ' "$signal_file" || return 1
  grep -q 'smart_arm_rank_top8' "$signal_file" || return 1

  # recommendations must keep strict tab-separated header
  local reco_header
  reco_header="$(sed -n '1p' "$reco_file")"
  [[ "$reco_header" == $'label\toverrides\treason' ]] || return 1
  local reco_rows
  reco_rows="$(tail -n +2 "$reco_file" | grep -cv '^[[:space:]]*$' || true)"
  [[ "$reco_rows" -ge 1 ]] || return 1
  awk -F'\t' 'NR>1 && NF<3 {exit 1}' "$reco_file"

  # sweep must contain at least one runnable row with label + overrides
  local sweep_rows
  sweep_rows="$(grep -Ev '^(#|[[:space:]]*$)' "$sweep_file" | wc -l | tr -d ' ')"
  [[ "$sweep_rows" -ge 1 ]] || return 1
  awk -F'\t' '!/^#/ && NF>0 && NF<2 {exit 1}' "$sweep_file"
}

run_remote_once() {
  local stamp signal_tmp reco_tmp sweep_tmp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  signal_tmp="$OUT_LATEST.tmp"
  reco_tmp="$OUT_RECO.tmp"
  sweep_tmp="$OUT_SWEEP.tmp"

  retry_run sshpass -p "$HGV_PASS" scp "${SSH_OPTS[@]}" "$ANALYZER_SRC" "$REPO_ROOT/logs/sweep/results.csv" "$HGV_USER@$HGV_HOST:/tmp/" >/dev/null

  {
    echo "=== $stamp ==="
    retry_run sshpass -p "$HGV_PASS" ssh "${SSH_OPTS[@]}" "$HGV_USER@$HGV_HOST" "python3 /tmp/hgv_smart_crunch.py --csv /tmp/results.csv --recommend-out /tmp/hgv_recommendations.tsv --sweep-out /tmp/hgv_recommendations_sweep.tsv"
    echo
  } > "$signal_tmp"

  retry_run sshpass -p "$HGV_PASS" scp "${SSH_OPTS[@]}" "$HGV_USER@$HGV_HOST:/tmp/hgv_recommendations.tsv" "$reco_tmp" >/dev/null
  retry_run sshpass -p "$HGV_PASS" scp "${SSH_OPTS[@]}" "$HGV_USER@$HGV_HOST:/tmp/hgv_recommendations_sweep.tsv" "$sweep_tmp" >/dev/null

  validate_outputs "$signal_tmp" "$reco_tmp" "$sweep_tmp" || {
    write_health "degraded" "remote" "remote_output_validation_failed"
    return 1
  }

  mv "$signal_tmp" "$OUT_LATEST"
  mv "$reco_tmp" "$OUT_RECO"
  mv "$sweep_tmp" "$OUT_SWEEP"
  cat "$OUT_LATEST" >> "$OUT_HISTORY"
  write_health "ok" "remote" "remote_sync_and_analysis_success"
  echo "[hgv-analysis] updated $OUT_LATEST (remote)"
}

run_local_fallback() {
  local stamp signal_tmp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  signal_tmp="$OUT_LATEST.tmp"
  {
    echo "=== $stamp ==="
    python3 "$ANALYZER_SRC" --csv "$REPO_ROOT/logs/sweep/results.csv" --recommend-out "$OUT_RECO.tmp" --sweep-out "$OUT_SWEEP.tmp"
    echo
  } > "$signal_tmp"

  validate_outputs "$signal_tmp" "$OUT_RECO.tmp" "$OUT_SWEEP.tmp" || {
    write_health "error" "local-fallback" "local_output_validation_failed"
    return 1
  }

  mv "$signal_tmp" "$OUT_LATEST"
  mv "$OUT_RECO.tmp" "$OUT_RECO"
  mv "$OUT_SWEEP.tmp" "$OUT_SWEEP"
  cat "$OUT_LATEST" >> "$OUT_HISTORY"
  write_health "degraded" "local-fallback" "remote_unavailable_local_analysis_used"
  echo "[hgv-analysis] updated $OUT_LATEST (local fallback)"
}

run_once() {
  if run_remote_once; then
    return 0
  fi
  echo "[hgv-analysis] remote cycle failed after retries; switching to local fallback"
  run_local_fallback || {
    write_health "error" "none" "both_remote_and_local_failed"
    return 1
  }
}

acquire_lock
trap release_lock EXIT INT TERM

if [[ "${1:-}" == "--once" ]]; then
  run_once || true
  exit 0
fi

echo "[hgv-analysis] starting loop interval=${INTERVAL_SECS}s host=$HGV_USER@$HGV_HOST"
while true; do
  run_once || true
  sleep "$INTERVAL_SECS"
done
