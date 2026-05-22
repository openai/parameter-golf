#!/usr/bin/env bash
# squid_monitor.sh — reliable remote monitor with clean connection semantics.
#
# Core fix:
# - "connection_failed" is only emitted when SSH transport/auth fails.
# - "no_matching_training_process" is emitted when SSH works but process grep finds none.

set -u

HOST_ALIAS="${1:-${REMOTE_HOST_ALIAS:-squid}}"
KEY_PATH="${REMOTE_SSH_KEY:?Set REMOTE_SSH_KEY to the SSH private key path}"
CONNECT_TIMEOUT="${REMOTE_CONNECT_TIMEOUT:-5}"
REPO_DIR="${REMOTE_REPO_DIR:-~/parameter-golf-main}"
PROC_REGEX="${REMOTE_PROC_REGEX:-sweep_runner.sh|run_experiment.sh|train_gpt_sota_decoded.py}"
MAX_RETRIES="${REMOTE_MAX_RETRIES:-3}"
RETRY_SLEEP_SECS="${REMOTE_RETRY_SLEEP_SECS:-1}"
FAIL_STREAK_THRESHOLD="${REMOTE_FAIL_STREAK_THRESHOLD:-3}"
FAIL_STREAK_FILE="${REMOTE_FAIL_STREAK_FILE:-/tmp/squid_monitor_${HOST_ALIAS}_fail_streak}"

SSH_BASE=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout="$CONNECT_TIMEOUT"
  -o StrictHostKeyChecking=accept-new
  -i "$KEY_PATH"
  "$HOST_ALIAS"
)

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

read_fail_streak() {
  if [[ -f "$FAIL_STREAK_FILE" ]]; then
    cat "$FAIL_STREAK_FILE" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

write_fail_streak() {
  printf "%s\n" "$1" >"$FAIL_STREAK_FILE" 2>/dev/null || true
}

clear_fail_streak() {
  rm -f "$FAIL_STREAK_FILE" 2>/dev/null || true
}

echo "[$(timestamp)] squid_monitor: checking transport"
attempt=1
while true; do
  if "${SSH_BASE[@]}" "echo connected" >/dev/null 2>&1; then
    clear_fail_streak
    break
  fi
  if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
    streak="$(read_fail_streak)"
    if ! [[ "$streak" =~ ^[0-9]+$ ]]; then
      streak=0
    fi
    streak=$((streak + 1))
    write_fail_streak "$streak"

    # Only escalate to hard failure after repeated failed polls.
    if [[ "$streak" -ge "$FAIL_STREAK_THRESHOLD" ]]; then
      echo "status=connection_failed host=$HOST_ALIAS attempts=$attempt consecutive_failures=$streak"
      exit 2
    fi

    echo "status=connection_degraded host=$HOST_ALIAS attempts=$attempt consecutive_failures=$streak"
    exit 0
  fi
  echo "[$(timestamp)] transport_retry attempt=$attempt"
  sleep "$RETRY_SLEEP_SECS"
  attempt=$((attempt + 1))
done

echo "status=connected host=$HOST_ALIAS"

# Process check must never cause a connection failure classification.
RUNNING_COUNT="$(${SSH_BASE[@]} "ps -ef | grep -E '$PROC_REGEX' | grep -v grep | wc -l" 2>/dev/null || echo 0)"
RUNNING_COUNT="${RUNNING_COUNT//[[:space:]]/}"

if [[ -z "$RUNNING_COUNT" ]]; then
  RUNNING_COUNT=0
fi

echo "running_processes=$RUNNING_COUNT"

if [[ "$RUNNING_COUNT" -gt 0 ]]; then
  echo "[$(timestamp)] active_processes"
  "${SSH_BASE[@]}" "ps -ef | grep -E '$PROC_REGEX' | grep -v grep" 2>/dev/null || true
else
  echo "[$(timestamp)] no_matching_training_process"
fi

echo "[$(timestamp)] recent_sweep_logs"
"${SSH_BASE[@]}" "cd '$REPO_DIR' 2>/dev/null && ls -lt logs/sweep 2>/dev/null | head -n 12" 2>/dev/null || true

exit 0
