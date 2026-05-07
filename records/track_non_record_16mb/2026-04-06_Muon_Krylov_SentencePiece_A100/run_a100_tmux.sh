#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  REMOTE_HOST=<host> ./run_a100_tmux.sh

Optional environment overrides:
  REMOTE_USER          SSH user (default: root)
  REMOTE_PORT          SSH port (default: 22)
  REMOTE_KEY           SSH private key path (default: use your normal ssh config)
  REMOTE_BASE_DIR      Remote records base dir
  REMOTE_DIR           Exact remote record dir (overrides REMOTE_BASE_DIR)
  SESSION              tmux session name
  RUN_ID               training RUN_ID (default: SESSION)
  DATA_PATH            remote dataset path
  TOKENIZER_PATH       remote tokenizer path
  NPROC_PER_NODE       torchrun worker count (default: 1)
  TARGET_MB            export target in MiB (default: 15.2587890625)
  SEED                 random seed (default: 1337)
  MAX_WALLCLOCK_SECONDS training cap in seconds (default: 0)

Example:
  REMOTE_HOST=216.81.245.7 REMOTE_PORT=49989 REMOTE_KEY=~/.ssh/id_ed25519 ./run_a100_tmux.sh
EOF
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECORD_DIR_NAME="$(basename "${SCRIPT_DIR}")"

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_KEY="${REMOTE_KEY:-}"

REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/workspace/parametergolf/records/track_non_record_16mb}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_BASE_DIR}/${RECORD_DIR_NAME}}"

SESSION="${SESSION:-$(printf '%s' "${RECORD_DIR_NAME}" | tr -cs '[:alnum:]_-' '_')}"
RUN_ID="${RUN_ID:-${SESSION}}"

DATA_PATH="${DATA_PATH:-/workspace/parameter-golf/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

TARGET_MB="${TARGET_MB:-15.2587890625}"
SEED="${SEED:-1337}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

MUON_KRYLOV_ENABLED="${MUON_KRYLOV_ENABLED:-1}"
MUON_KRYLOV_ALPHA="${MUON_KRYLOV_ALPHA:-0.05}"
MUON_KRYLOV_ETA_THRESHOLD="${MUON_KRYLOV_ETA_THRESHOLD:-0.03}"
MUON_KRYLOV_WARMUP_STEPS="${MUON_KRYLOV_WARMUP_STEPS:-1000}"
MUON_KRYLOV_DECISION_EVERY="${MUON_KRYLOV_DECISION_EVERY:-100}"
MUON_KRYLOV_EVERY="${MUON_KRYLOV_EVERY:-2}"
MUON_KRYLOV_HUTCHINSON_SAMPLES="${MUON_KRYLOV_HUTCHINSON_SAMPLES:-2}"
MUON_KRYLOV_RANK_MAX="${MUON_KRYLOV_RANK_MAX:-4}"
MUON_KRYLOV_RANK_SCALE="${MUON_KRYLOV_RANK_SCALE:-1.0}"

VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-2000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "error: set REMOTE_HOST before running this script" >&2
  exit 1
fi

SSH_ARGS=("${REMOTE_USER}@${REMOTE_HOST}" -p "${REMOTE_PORT}")
if [[ -n "${REMOTE_KEY}" ]]; then
  SSH_ARGS+=(-i "${REMOTE_KEY}")
fi

ssh "${SSH_ARGS[@]}" 'bash -s' <<EOF
set -euo pipefail
cd "${REMOTE_DIR}"
mkdir -p logs
LOG="logs/${SESSION}_\$(date +%Y%m%d_%H%M%S).log"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "
cd ${REMOTE_DIR} && env \\
RUN_ID=${RUN_ID} \\
DATA_PATH=${DATA_PATH} \\
TOKENIZER_PATH=${TOKENIZER_PATH} \\
TARGET_MB=${TARGET_MB} \\
MUON_KRYLOV_ENABLED=${MUON_KRYLOV_ENABLED} \\
MUON_KRYLOV_ALPHA=${MUON_KRYLOV_ALPHA} \\
MUON_KRYLOV_ETA_THRESHOLD=${MUON_KRYLOV_ETA_THRESHOLD} \\
MUON_KRYLOV_WARMUP_STEPS=${MUON_KRYLOV_WARMUP_STEPS} \\
MUON_KRYLOV_DECISION_EVERY=${MUON_KRYLOV_DECISION_EVERY} \\
MUON_KRYLOV_EVERY=${MUON_KRYLOV_EVERY} \\
MUON_KRYLOV_HUTCHINSON_SAMPLES=${MUON_KRYLOV_HUTCHINSON_SAMPLES} \\
MUON_KRYLOV_RANK_MAX=${MUON_KRYLOV_RANK_MAX} \\
MUON_KRYLOV_RANK_SCALE=${MUON_KRYLOV_RANK_SCALE} \\
VAL_LOSS_EVERY=${VAL_LOSS_EVERY} \\
TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY} \\
WARMUP_STEPS=${WARMUP_STEPS} \\
SEED=${SEED} \\
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} \\
torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} train_gpt.py 2>&1 | tee \${LOG}"
echo "session:${SESSION}"
echo "log:\${LOG}"
EOF
