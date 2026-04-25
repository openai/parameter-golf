#!/usr/bin/env bash
# Usage: ./await_steps.sh <experiment_dir> [n=10]
#
# Blocks until <experiment_dir>/run.log contains N training-step log lines, then
# prints them. Used to gate on a healthy trajectory before committing to wait
# the full ~5 minutes for a run to finish, and also to peek mid-run at any
# threshold the agent cares about (e.g. "show me steps through 100").
#
# Robustness:
#   - If the log file doesn't exist yet (script called before run_experiment.sh
#     has started writing), waits up to MAX_WAIT_SECONDS for it to appear.
#   - Exits early if the python training process dies before reaching N
#     (pgrep "train_gpt.py", case-insensitive).
#   - Exits early if the log file goes stale for >LOG_STALE_SECONDS, BUT only
#     while training is still in progress. Once the final training step
#     (step:M/M train_loss:...) has been logged, the stale-mtime check is
#     suppressed because eval_val runs as a single computation that doesn't
#     log incrementally — minutes of legitimate silence then.
#   - Hard ceiling of MAX_WAIT_SECONDS (default 600) so the script never hangs
#     a Bash background task forever. Override if you need to wait through a
#     full-val eval (which can take >10 min on MPS).
#
# Override defaults via env vars: MAX_WAIT_SECONDS, LOG_STALE_SECONDS.

set -uo pipefail

EXP_DIR="${1:-}"
N="${2:-10}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
LOG_STALE_SECONDS="${LOG_STALE_SECONDS:-60}"

if [[ -z "$EXP_DIR" || "$EXP_DIR" == "-h" || "$EXP_DIR" == "--help" ]]; then
  cat <<EOF
Usage: $0 <experiment_dir> [n=10]

Blocks until <experiment_dir>/run.log contains N training-step log lines, then
prints them. Useful for gating on a healthy trajectory before committing to a
full ~5 min wait, and also for peeking mid-run at any step threshold (e.g.
n=100 to inspect through step 100).

Exits early on:
  - python process gone (pgrep "train_gpt.py")
  - log mtime stale > LOG_STALE_SECONDS (default 60) — handles hung Python.
    Suppressed once the final training step has been logged, because eval_val
    is a single long computation that doesn't log incrementally.
  - hard timeout > MAX_WAIT_SECONDS (default 600). Bump this if you're
    waiting through a full-val eval (>10 min on MPS).

Env overrides: MAX_WAIT_SECONDS, LOG_STALE_SECONDS.
EOF
  exit 0
fi
if [[ ! -d "$EXP_DIR" ]]; then
  echo "Error: experiment dir not found: $EXP_DIR" >&2
  exit 1
fi

LOG="${EXP_DIR}/run.log"
START_EPOCH=$(date +%s)

# Portable log-mtime helper (BSD stat on macOS, GNU stat on Linux).
log_mtime() {
  stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo 0
}
elapsed() { echo $(( $(date +%s) - START_EPOCH )); }

# Did training finish? Final-step line has the form `step:M/M train_loss:...`
# where the iteration index equals the iteration total. After that, eval can
# run silently for minutes and that's expected.
training_done() {
  grep -qE '^step:([0-9]+)/\1 train_loss:' "$LOG" 2>/dev/null
}

# Phase 1: wait for the log file to be created.
while [[ ! -f "$LOG" ]]; do
  if (( $(elapsed) > MAX_WAIT_SECONDS )); then
    echo "(timed out after ${MAX_WAIT_SECONDS}s waiting for ${LOG} to be created)" >&2
    exit 2
  fi
  sleep 0.5
done

# Phase 2: wait until we have N step lines, or the run is clearly done/stuck.
while true; do
  count=$(grep -cE '^step:[0-9]+/[0-9]+ train_loss:' "$LOG" 2>/dev/null || echo 0)
  if (( count >= N )); then break; fi

  # Cap N at the run's actual iteration count: if training has already logged
  # its final step, count won't grow further. Caller asked for more steps
  # than the run does — give them everything we have and bail out.
  if training_done; then
    echo "(training finished at step ${count}; N=${N} > iterations, returning ${count})" >&2
    break
  fi

  # Crash signal 1: python process gone. Match the script name only (not the
  # binary), case-insensitive, so we catch both .../Python and .../python and
  # any pyenv/uv variant.
  if ! pgrep -if "train_gpt.py" > /dev/null; then
    echo "(python exited before reaching N=${N}; ${count} step lines so far)" >&2
    break
  fi

  # Crash signal 2: log not growing — process hung or pgrep is matching the
  # wrong python. Suppressed during eval (post-final-training-step) since
  # eval_val runs for minutes without incremental logging — but training_done
  # is already an exit condition above, so this branch only runs during the
  # training phase where the log is expected to grow steadily.
  log_age=$(( $(date +%s) - $(log_mtime) ))
  if (( log_age > LOG_STALE_SECONDS )); then
    echo "(log idle for ${log_age}s — assuming run is hung; ${count} step lines so far)" >&2
    break
  fi

  # Hard ceiling.
  if (( $(elapsed) > MAX_WAIT_SECONDS )); then
    echo "(hard timeout ${MAX_WAIT_SECONDS}s reached; ${count}/${N} steps logged)" >&2
    break
  fi

  sleep 1
done

grep -E '^step:[0-9]+/[0-9]+ train_loss:' "$LOG" 2>/dev/null | head -"$N"
