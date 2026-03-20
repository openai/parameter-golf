#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_remote_experiment.sh <control|group64|group64_outliers8>
  scripts/run_remote_experiment.sh summary <log1> [log2 ...]

Environment overrides:
  REPO_DIR=/workspace/parameter-golf
  DATA_PATH=$REPO_DIR/data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024
  NPROC_PER_NODE=1
  MAX_WALLCLOCK_SECONDS=600
  VAL_LOSS_EVERY=200
  TRAIN_LOG_EVERY=100
EOF
}

summary() {
  for log in "$@"; do
    echo "===== ${log} ====="
    grep -E 'Serialized model int8\+zlib:|Total submission size int8\+zlib:|final_int8_zlib_roundtrip ' "$log" || true
    echo
  done
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift || true

if [[ "$MODE" == "summary" ]]; then
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  summary "$@"
  exit 0
fi

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"

mkdir -p "$REPO_DIR/logs/remote_runs"
cd "$REPO_DIR"

case "$MODE" in
  control)
    RUN_ID="control_per_row"
    export INT8_GROUP_SIZE=999999
    export INT8_OUTLIER_COLS=0
    export INT8_CLIP_PERCENTILE=99.99984
    ;;
  group64)
    RUN_ID="group64"
    export INT8_GROUP_SIZE=64
    export INT8_OUTLIER_COLS=0
    export INT8_CLIP_PERCENTILE=99.99984
    ;;
  group64_outliers8)
    RUN_ID="group64_outliers8"
    export INT8_GROUP_SIZE=64
    export INT8_OUTLIER_COLS=8
    export INT8_OUTLIER_MIN_ROWS=256
    export INT8_OUTLIER_NAME_PATTERNS='tok_emb,lm_head,c_q,c_k,c_v,proj,fc'
    export INT8_CLIP_PERCENTILE=99.99984
    ;;
  *)
    usage
    exit 1
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/remote_runs/${STAMP}_${RUN_ID}.log"

echo "mode=$MODE run_id=$RUN_ID log=$LOG"
echo "repo=$REPO_DIR data=$DATA_PATH nproc=$NPROC_PER_NODE"

export RUN_ID DATA_PATH TOKENIZER_PATH VOCAB_SIZE MAX_WALLCLOCK_SECONDS VAL_LOSS_EVERY TRAIN_LOG_EVERY

set -x
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py 2>&1 | tee "$LOG"
set +x

echo
echo "Summary from $LOG"
grep -E 'Serialized model int8\+zlib:|Total submission size int8\+zlib:|final_int8_zlib_roundtrip ' "$LOG" || true
