#!/usr/bin/env bash
# run_experiment.sh — single experiment runner with hard timeout, log capture,
# failure classification, and CSV result append. Never blocks the caller:
# always exits 0 even if the training itself failed.
#
# Usage:
#   bash scripts/run_experiment.sh <LABEL> [ENV_KEY=VALUE ...]
#
# Example:
#   bash scripts/run_experiment.sh qk525_ttt QK_GAIN_INIT=5.25 TTT_ENABLED=1
#
# Writes:
#   logs/sweep/<LABEL>.log            full stdout/stderr
#   logs/sweep/results.csv            one appended row per run
#
# Env overrides (set before calling):
#   SCRIPT=train_gpt_sota_decoded.py  # training script to use
#   TIMEOUT_SECS=650                  # hard kill timeout (wallclock 600 + slack)
#   RESULTS_CSV=logs/sweep/results.csv

set -u  # -e intentionally OFF: we must always record & return

LABEL="${1:?label required}"
shift || true

TRAIN_PID=""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="${SCRIPT:-train_gpt_sota_decoded.py}"
# Pre-scan positional KEY=VAL args so TIMEOUT_SECS / SCRIPT specified in the
# sweep TSV row take effect BEFORE the defaults below are applied. Without this,
# a TSV line like `... TIMEOUT_SECS=5400 ...` is silently ignored (the outer
# `timeout --kill-after=10 $TIMEOUT_SECS` below reads the default 1800).
for kv in "$@"; do
  case "$kv" in
    TIMEOUT_SECS=*|SCRIPT=*) export "$kv" ;;
  esac
done
# TIMEOUT_SECS must cover: train (~6-7m on cold cache) + torch.compile eval warmup (~1-2m) +
# pre-quant eval (~1-2m eager) + GPTQ calib/quant/compress (~2-3m) +
# dequant + 2nd compile (~1-2m) + quant eval (~1-2m) + optional sliding/TTT.
# 1800s covers baseline cleanly with slack; bump to 2400 if TTT=1.
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
SWEEP_DIR="logs/sweep"
RESULTS_CSV="${RESULTS_CSV:-$SWEEP_DIR/results.csv}"
LOG="$SWEEP_DIR/${LABEL}.log"

mkdir -p "$SWEEP_DIR"

# CSV header (written once). Extended schema captures run context so results
# remain interpretable months later without cross-referencing TSV/config files.
CSV_HEADER="label,status,exit_code,wall_s,train_loss,pre_quant_bpb,quant_bpb,sliding_bpb,ttt_bpb,delta_bpb,tok_s,peak_mem_gb,failure_class,timestamp,hostname,script,script_sha8,seed,iterations,sliding_enabled,ttt_enabled,fast_smoke,overrides,notes"
if [[ ! -s "$RESULTS_CSV" ]]; then
  echo "$CSV_HEADER" > "$RESULTS_CSV"
fi

# csv_escape: quote a value if it contains comma, quote, or newline; double any quotes.
csv_escape() {
  local v="${1:-}"
  if [[ "$v" == *,* || "$v" == *\"* || "$v" == *$'\n'* ]]; then
    v="${v//\"/\"\"}"
    printf '"%s"' "$v"
  else
    printf '%s' "$v"
  fi
}

# Always append a row — even on kill -9 — via trap.
append_row() {
  local status="$1" code="$2" wall="$3"
  local parsed
  if [[ -f "$LOG" ]]; then
    parsed="$(python3 scripts/parse_log.py "$LOG" 2>/dev/null || echo ",,,,,,,,parse_error")"
  else
    parsed=",,,,,,,,no_log"
  fi
  local train_loss pre_quant quant sliding ttt delta tok_s peak_mem fail_class
  IFS=',' read -r train_loss pre_quant quant sliding ttt delta tok_s peak_mem fail_class <<< "$parsed"
  # Refine status from failure_class if we had a crash
  if [[ "$fail_class" == "oversize" ]]; then
    status="oversize"
  fi
  if [[ "$status" == "ok" && -n "$fail_class" ]]; then
    status="$fail_class"
  fi
  if [[ "$status" == "ok" && -z "$quant" && -z "$pre_quant" ]]; then
    status="no_bpb"
  fi
  local ts host sha ovr notes
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  host="$(hostname 2>/dev/null || echo unknown)"
  if [[ -f "$SCRIPT" ]]; then
    sha="$(sha256sum "$SCRIPT" 2>/dev/null | cut -c1-8)"
  else
    sha=""
  fi
  ovr="${RUN_OVERRIDES:-}"
  notes="${NOTES:-}"
  {
    printf '%s,' "$(csv_escape "$LABEL")"
    printf '%s,' "$(csv_escape "$status")"
    printf '%s,%s,' "$code" "$wall"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$train_loss" "$pre_quant" "$quant" "$sliding" "$ttt" "$delta" "$tok_s" "$peak_mem"
    printf '%s,' "$(csv_escape "$fail_class")"
    printf '%s,' "$ts"
    printf '%s,' "$(csv_escape "$host")"
    printf '%s,%s,' "$(csv_escape "$SCRIPT")" "$sha"
    printf '%s,%s,%s,%s,%s,' \
      "${SEED:-}" "${ITERATIONS:-}" "${SLIDING_WINDOW_ENABLED:-}" "${TTT_ENABLED:-}" "${FAST_SMOKE:-}"
    printf '%s,' "$(csv_escape "$ovr")"
    printf '%s\n' "$(csv_escape "$notes")"
  } >> "$RESULTS_CSV"
  echo "[run_experiment] ${LABEL} => status=${status} code=${code} wall=${wall}s train_loss=${train_loss:-?} pre_quant_bpb=${pre_quant:-?} quant_bpb=${quant:-?} delta=${delta:-?}"
}

# Append a lightweight stage row so interrupted runs still preserve progress.
# This keeps train/eval timeout domains debuggable and makes eval-only resume
# flows trivial to stitch by run_id.
append_stage_row() {
  local stage="$1" status="$2" code="$3" wall="$4" ckpt_path="${5:-}"
  local parsed
  if [[ -f "$LOG" ]]; then
    parsed="$(python3 scripts/parse_log.py "$LOG" 2>/dev/null || echo ",,,,,,,,parse_error")"
  else
    parsed=",,,,,,,,no_log"
  fi
  local train_loss pre_quant quant sliding ttt delta tok_s peak_mem fail_class
  IFS=',' read -r train_loss pre_quant quant sliding ttt delta tok_s peak_mem fail_class <<< "$parsed"
  local ts host sha notes_with_meta ovr stage_fail
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  host="$(hostname 2>/dev/null || echo unknown)"
  if [[ -f "$SCRIPT" ]]; then
    sha="$(sha256sum "$SCRIPT" 2>/dev/null | cut -c1-8)"
  else
    sha=""
  fi
  ovr="${RUN_OVERRIDES:-} RUN_ID=${RUN_ID:-}"
  notes_with_meta="${NOTES:-} | stage=${stage} run_id=${RUN_ID:-} checkpoint_path=${ckpt_path}"
  stage_fail="stage:${stage}"
  {
    printf '%s,' "$(csv_escape "$LABEL")"
    printf '%s,' "$(csv_escape "$status")"
    printf '%s,%s,' "$code" "$wall"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$train_loss" "$pre_quant" "$quant" "$sliding" "$ttt" "$delta" "$tok_s" "$peak_mem"
    printf '%s,' "$(csv_escape "$stage_fail")"
    printf '%s,' "$ts"
    printf '%s,' "$(csv_escape "$host")"
    printf '%s,%s,' "$(csv_escape "$SCRIPT")" "$sha"
    printf '%s,%s,%s,%s,%s,' \
      "${SEED:-}" "${ITERATIONS:-}" "${SLIDING_WINDOW_ENABLED:-}" "${TTT_ENABLED:-}" "${FAST_SMOKE:-}"
    printf '%s,' "$(csv_escape "$ovr")"
    printf '%s\n' "$(csv_escape "$notes_with_meta")"
  } >> "$RESULTS_CSV"
  echo "[run_experiment] ${LABEL} stage=${stage} status=${status} wall=${wall}s run_id=${RUN_ID:-}"
}

START_TS=$(date +%s)
handle_stop_signal() {
  local signal="$1"
  local code=143
  if [[ "$signal" == "INT" ]]; then
    code=130
  fi
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    pkill -TERM -P "$TRAIN_PID" 2>/dev/null
    kill -TERM "$TRAIN_PID" 2>/dev/null
    sleep 2
    pkill -KILL -P "$TRAIN_PID" 2>/dev/null
    kill -KILL "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
  fi
  WALL=$(( $(date +%s) - START_TS ))
  append_row "stopped" "$code" "$WALL"
  trap - INT TERM
  exit "$code"
}
trap 'handle_stop_signal INT' INT
trap 'handle_stop_signal TERM' TERM

# Default smoke-test env (caller overrides wins)
# Single GB10, ~19k tok/s with 11L/512d SOTA model.
# 150 iters * 32K tokens / 19k tok/s = ~260s train.
# MAX_WALLCLOCK_SECONDS caps TRAINING ONLY (eval is unbounded).
# Full pipeline at SLIDING=0 TTT=0: train 5m + eval/quant 5-7m = ~12m.
export ITERATIONS="${ITERATIONS:-150}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export VAL_BATCH_TOKENS="${VAL_BATCH_TOKENS:-131072}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-400}"
# GPTQ: 4 calib batches is half of default 8 — still produces comparable quant_bpb
# across runs since all use the same value. Saves ~30-60s per run at smoke scale.
export GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-4}"
export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-5}"
# Smoke defaults: OFF for speed. Turn on per-run when specifically testing them.
export SLIDING_WINDOW_ENABLED="${SLIDING_WINDOW_ENABLED:-0}"
export TTT_ENABLED="${TTT_ENABLED:-0}"
export SEED="${SEED:-1337}"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
export RUN_ID="sweep_${LABEL}"
export PYTHONUNBUFFERED=1
# NPROC_PER_NODE>1 implies torchrun multi-GPU launch. Compile must be explicitly
# enabled (caller passes TORCHDYNAMO_DISABLE=0) because torch.compile is broken
# on aarch64/Spark; safe on H100.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
# NCCL fixes for H100 SXM on RunPod containers (NCCL 2.20+):
# cuMemCreate API (CUMEM) and NVLS cause cuda error 1 in restricted containers.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

# Apply caller-supplied KEY=VALUE overrides
# Preserve the raw override string for CSV provenance
RUN_OVERRIDES="$*"
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    export "$kv"
  fi
done

# Stable run identity for stage rows and future eval-only resume flow.
RUN_ID="${RUN_ID:-sweep_${LABEL}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
export RUN_ID

# Stage marker guards (prevent duplicate partial rows).
STAGE_STARTED_WRITTEN=0
STAGE_TRAIN_COMPLETE_WRITTEN=0
STAGE_EVAL_STARTED_WRITTEN=0

# Probe log for stage markers emitted by the training script.
maybe_write_stage_markers() {
  local wall_now="$1"
  local ckpt_final="$REPO_ROOT/final_model.pt"
  if [[ "$STAGE_TRAIN_COMPLETE_WRITTEN" == "0" ]] && grep -q "stage:train_complete" "$LOG" 2>/dev/null; then
    append_stage_row "train_complete" "train_complete" "0" "$wall_now" "$ckpt_final"
    STAGE_TRAIN_COMPLETE_WRITTEN=1
  fi
  if [[ "$STAGE_EVAL_STARTED_WRITTEN" == "0" ]] && grep -q "stage:eval_started" "$LOG" 2>/dev/null; then
    append_stage_row "eval_started" "eval_started" "0" "$wall_now" "$ckpt_final"
    STAGE_EVAL_STARTED_WRITTEN=1
  fi
}

# Activate venv if present
if [[ -f "$REPO_ROOT/.venv-spark/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-spark/bin/activate"
fi

{
  echo "=============================================="
  echo "SWEEP RUN: $LABEL"
  echo "  script=$SCRIPT"
  echo "  timeout=${TIMEOUT_SECS}s"
  echo "  overrides: $*"
  echo "  iters=$ITERATIONS seq=$TRAIN_SEQ_LEN bs=$TRAIN_BATCH_TOKENS"
  echo "  ttt=${TTT_ENABLED:-0} sliding=${SLIDING_WINDOW_ENABLED} qk=${QK_GAIN_INIT:-default}"
  echo "  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=============================================="
} > "$LOG"

# Persist run metadata at job start.
append_stage_row "started" "started" "0" "0" ""
STAGE_STARTED_WRITTEN=1

# FAST_SMOKE=1: kill the python process as soon as the final step's val_bpb is
# logged. We capture train_loss and val_bpb before killing, skipping the expensive
# post-training eval chain (pre-quant eval, GPTQ, quant eval). Cuts wall from
# ~25m to ~7m per run. Use FAST_SMOKE=0 for full-eval runs that need quant_bpb.
FAST_SMOKE="${FAST_SMOKE:-1}"

# Run with hard kill timeout; -k gives 10s grace after SIGTERM
# Build launcher: torchrun for multi-GPU, plain python3 for single-GPU.
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  TRAIN_LAUNCHER="torchrun --nproc_per_node=${NPROC_PER_NODE}"
  echo "[run_experiment] torchrun launch: NPROC_PER_NODE=${NPROC_PER_NODE}" >> "$LOG"
else
  TRAIN_LAUNCHER="python3 -u"
fi

if [[ "$FAST_SMOKE" == "1" ]]; then
  # Launch training in background, tail-watch log for final val_bpb marker.
  timeout --kill-after=10 "${TIMEOUT_SECS}" ${TRAIN_LAUNCHER} "$SCRIPT" >> "$LOG" 2>&1 &
  TRAIN_PID=$!
  # Watcher: the training script logs "<iter>/<iter> val_loss: X val_bpb: Y" at
  # the final step (guaranteed by last_step branch). Once we see that line,
  # kill the python tree and record. Poll every 2s.
  TARGET="${ITERATIONS}/${ITERATIONS} val_loss:"
  WATCH_DEADLINE=$(( $(date +%s) + TIMEOUT_SECS ))
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    WALL_NOW=$(( $(date +%s) - START_TS ))
    maybe_write_stage_markers "$WALL_NOW"
    if grep -q -F "$TARGET" "$LOG" 2>/dev/null; then
      echo "[run_experiment] FAST_SMOKE: final val_bpb observed, terminating training" >> "$LOG"
      # Give it 3s to flush EMA + peak_mem log lines we also want
      sleep 3
      pkill -TERM -P "$TRAIN_PID" 2>/dev/null
      kill -TERM "$TRAIN_PID" 2>/dev/null
      sleep 2
      pkill -KILL -P "$TRAIN_PID" 2>/dev/null
      kill -KILL "$TRAIN_PID" 2>/dev/null
      wait "$TRAIN_PID" 2>/dev/null
      EXIT_CODE=0
      break
    fi
    if (( $(date +%s) >= WATCH_DEADLINE )); then
      echo "[run_experiment] FAST_SMOKE: watcher deadline hit, letting timeout handle it" >> "$LOG"
      break
    fi
    sleep 2
  done
  wait "$TRAIN_PID" 2>/dev/null
  WALL_NOW=$(( $(date +%s) - START_TS ))
  maybe_write_stage_markers "$WALL_NOW"
  EXIT_CODE="${EXIT_CODE:-$?}"
else
  timeout --kill-after=10 "${TIMEOUT_SECS}" ${TRAIN_LAUNCHER} "$SCRIPT" >> "$LOG" 2>&1 &
  TRAIN_PID=$!
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    WALL_NOW=$(( $(date +%s) - START_TS ))
    maybe_write_stage_markers "$WALL_NOW"
    sleep 2
  done
  wait "$TRAIN_PID" 2>/dev/null
  EXIT_CODE=$?
fi
END_TS=$(date +%s)
WALL=$(( END_TS - START_TS ))

# Final stage probe in case marker appeared right before process exit.
maybe_write_stage_markers "$WALL"

STATUS="ok"
case "$EXIT_CODE" in
  0)   STATUS="ok" ;;
  124) STATUS="timeout" ;;
  137) STATUS="killed" ;;
  143) STATUS="ok" ;;  # FAST_SMOKE SIGTERM — expected path when we got our metric
  *)   STATUS="error" ;;
esac

# Archive checkpoint per-label so future runs don't clobber each other.
# Needed for N2 model-soup (average N seed checkpoints). Enabled by default.
# Disable by setting ARCHIVE_CHECKPOINT=0.
if [[ "${ARCHIVE_CHECKPOINT:-1}" == "1" && -f "$REPO_ROOT/final_model.pt" ]]; then
  ARCHIVE_DIR="$REPO_ROOT/records/ckpts"
  mkdir -p "$ARCHIVE_DIR"
  cp -f "$REPO_ROOT/final_model.pt" "$ARCHIVE_DIR/${LABEL}.pt" 2>/dev/null \
    && echo "[run_experiment] archived checkpoint: records/ckpts/${LABEL}.pt ($(stat -c%s "$ARCHIVE_DIR/${LABEL}.pt" 2>/dev/null || echo ?) bytes)" >> "$LOG"
fi

# Ensure train_complete row exists for scripts without explicit stage markers.
if [[ "$STAGE_TRAIN_COMPLETE_WRITTEN" == "0" && -f "$REPO_ROOT/final_model.pt" ]]; then
  append_stage_row "train_complete_inferred" "train_complete" "0" "$WALL" "$REPO_ROOT/final_model.pt"
  STAGE_TRAIN_COMPLETE_WRITTEN=1
fi

append_row "$STATUS" "$EXIT_CODE" "$WALL"
exit 0
