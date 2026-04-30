#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

if [[ ! -f "${repo_root}/.venv-wsl/bin/activate" ]]; then
  echo "Missing WSL venv at ${repo_root}/.venv-wsl" >&2
  echo "Set up the WSL environment first." >&2
  exit 1
fi

source "${repo_root}/.venv-wsl/bin/activate"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Shared 3080 sweep baseline. Every config below inherits these unless overridden.
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"

export ITERATIONS="${ITERATIONS:-200000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
export WARMUP_STEPS="${WARMUP_STEPS:-1}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export ENABLE_FINAL_EVAL="${ENABLE_FINAL_EVAL:-1}"
export MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-262144}"

export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
export SDP_BACKEND="${SDP_BACKEND:-flash}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-768}"
export VAL_CONTEXT_LEN="${VAL_CONTEXT_LEN:-768}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-61440}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-12288}"
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-4}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"

export NUM_LAYERS="${NUM_LAYERS:-8}"
export MODEL_DIM="${MODEL_DIM:-640}"
export NUM_HEADS="${NUM_HEADS:-10}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-5}"
export MLP_MULT="${MLP_MULT:-2.75}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.2}"

export PARTIAL_ROPE_DIM="${PARTIAL_ROPE_DIM:-16}"
export LN_SCALE_ENABLED="${LN_SCALE_ENABLED:-1}"

export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-4096}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export XSA_ENABLED="${XSA_ENABLED:-1}"
export XSA_TOP_LAYERS="${XSA_TOP_LAYERS:-4}"
export XSA_SCALE="${XSA_SCALE:-1.0}"

export D_MEMORY="${D_MEMORY:-4}"
export TRUNK_D_MEMORY="${TRUNK_D_MEMORY:-0}"
export CORE_D_MEMORY="${CORE_D_MEMORY:-10}"
export CORE_MEMORY_ONLY="${CORE_MEMORY_ONLY:-1}"
export MEMORY_CHUNK_SIZE="${MEMORY_CHUNK_SIZE:-192}"
export MEMORY_SURPRISE_DECAY="${MEMORY_SURPRISE_DECAY:-0.95}"
export MEMORY_SURPRISE_THRESHOLD="${MEMORY_SURPRISE_THRESHOLD:-1.05}"
export MEMORY_SLOW_MAX_NORM="${MEMORY_SLOW_MAX_NORM:-2.0}"
export MEMORY_FAST_MAX_NORM="${MEMORY_FAST_MAX_NORM:-1.0}"
export MEMORY_WRITE_TOPK="${MEMORY_WRITE_TOPK:-8}"
export MEMORY_AUX_WEIGHT="${MEMORY_AUX_WEIGHT:-0.05}"
export TRAIN_MEMORY_CARRY_ENABLED="${TRAIN_MEMORY_CARRY_ENABLED:-0}"
export TRAIN_MEMORY_RESET_INTERVAL="${TRAIN_MEMORY_RESET_INTERVAL:-0}"
export TRAIN_MEMORY_FAST_DECAY="${TRAIN_MEMORY_FAST_DECAY:-0.25}"
export EVAL_MEMORY_FAST_DECAY="${EVAL_MEMORY_FAST_DECAY:-0.25}"
export TRAIN_MEMORY_SURPRISE_CARRY="${TRAIN_MEMORY_SURPRISE_CARRY:-0.95}"
export EVAL_MEMORY_SURPRISE_CARRY="${EVAL_MEMORY_SURPRISE_CARRY:-0.95}"
export TRAIN_MEMORY_MATCH_EVAL="${TRAIN_MEMORY_MATCH_EVAL:-0}"

export LOOP_CORE_ENABLED="${LOOP_CORE_ENABLED:-1}"
export LOOP_CORE_LAYERS="${LOOP_CORE_LAYERS:-3}"
export LOOP_REPEATS="${LOOP_REPEATS:-3}"
export LOOP_ATTN_EVERY="${LOOP_ATTN_EVERY:-2}"
export LOOP_ADAPTER_DIM="${LOOP_ADAPTER_DIM:-64}"
export LOOP_REPEAT_EMBED="${LOOP_REPEAT_EMBED:-1}"

export VALUE_RESIDUAL="${VALUE_RESIDUAL:-1}"
export GATED_ATTENTION="${GATED_ATTENTION:-1}"
export CATALYTIC_RESIDUAL="${CATALYTIC_RESIDUAL:-0}"

export QAT_ENABLED="${QAT_ENABLED:-1}"
export QAT_START_FRAC="${QAT_START_FRAC:-0.9}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export EMA_ENABLED="${EMA_ENABLED:-0}"

export INT5_MLP_EXPORT="${INT5_MLP_EXPORT:-1}"
export INT6_ATTN_EXPORT="${INT6_ATTN_EXPORT:-1}"
export INT6_OTHER_EXPORT="${INT6_OTHER_EXPORT:-0}"
export DISABLE_QUANTIZATION="${DISABLE_QUANTIZATION:-1}"
export USE_ZSTD="${USE_ZSTD:-1}"
export ZSTD_LEVEL="${ZSTD_LEVEL:-22}"

export NGRAM_CACHE_ORDER="${NGRAM_CACHE_ORDER:-4}"
export NGRAM_CACHE_MIN_ORDER="${NGRAM_CACHE_MIN_ORDER:-4}"
export NGRAM_CACHE_LAMBDA="${NGRAM_CACHE_LAMBDA:-0.20}"
export NGRAM_BACKOFF_DECAY="${NGRAM_BACKOFF_DECAY:-1.00}"
export CONTINUOUS_CACHE_LAMBDA="${CONTINUOUS_CACHE_LAMBDA:-0.20}"
export CONTINUOUS_CACHE_WINDOW="${CONTINUOUS_CACHE_WINDOW:-4096}"
export CONTINUOUS_CACHE_TOPK="${CONTINUOUS_CACHE_TOPK:-64}"
export CONTINUOUS_CACHE_TEMP="${CONTINUOUS_CACHE_TEMP:-0.10}"
export KNN_CACHE_LAMBDA="${KNN_CACHE_LAMBDA:-0.25}"
export KNN_TOPK="${KNN_TOPK:-64}"
export KNN_TEMP="${KNN_TEMP:-0.10}"
export KNN_DATASTORE_TOKENS="${KNN_DATASTORE_TOKENS:-32768}"
export OGD_BIAS_LR="${OGD_BIAS_LR:-0.02}"
export OGD_BIAS_WEIGHT_DECAY="${OGD_BIAS_WEIGHT_DECAY:-0.001}"
export OGD_BIAS_MAX_ABS="${OGD_BIAS_MAX_ABS:-3.0}"
export TTT_LR="${TTT_LR:-0.001}"
export TTT_WEIGHT_DECAY="${TTT_WEIGHT_DECAY:-0.0}"
export TTT_SCOPE="${TTT_SCOPE:-control}"
export POST_TRAINING_EVAL_MODE="${POST_TRAINING_EVAL_MODE:-4gram_ogd_bias}"

BASE_RUN_ID="${RUN_ID:-lnm_3080_sweep}"
BASE_TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}"
BASE_NUM_LAYERS="${NUM_LAYERS}"
BASE_MODEL_DIM="${MODEL_DIM}"
BASE_NUM_HEADS="${NUM_HEADS}"
BASE_NUM_KV_HEADS="${NUM_KV_HEADS}"
BASE_XSA_TOP_LAYERS="${XSA_TOP_LAYERS}"
BASE_MEMORY_CHUNK_SIZE="${MEMORY_CHUNK_SIZE}"
BASE_MEMORY_SURPRISE_THRESHOLD="${MEMORY_SURPRISE_THRESHOLD}"
BASE_D_MEMORY="${D_MEMORY}"
BASE_TRUNK_D_MEMORY="${TRUNK_D_MEMORY}"
BASE_CORE_D_MEMORY="${CORE_D_MEMORY}"
BASE_CORE_MEMORY_ONLY="${CORE_MEMORY_ONLY}"
BASE_MEMORY_WRITE_TOPK="${MEMORY_WRITE_TOPK}"
BASE_MEMORY_AUX_WEIGHT="${MEMORY_AUX_WEIGHT}"
BASE_TRAIN_MEMORY_MATCH_EVAL="${TRAIN_MEMORY_MATCH_EVAL}"
BASE_LOOP_CORE_ENABLED="${LOOP_CORE_ENABLED}"
BASE_LOOP_REPEATS="${LOOP_REPEATS}"
BASE_LOOP_ATTN_EVERY="${LOOP_ATTN_EVERY}"
BASE_NGRAM_CACHE_LAMBDA="${NGRAM_CACHE_LAMBDA}"
BASE_OGD_BIAS_LR="${OGD_BIAS_LR}"
BASE_VALUE_RESIDUAL="${VALUE_RESIDUAL}"
BASE_GATED_ATTENTION="${GATED_ATTENTION}"
BASE_POST_TRAINING_EVAL_MODE="${POST_TRAINING_EVAL_MODE}"
mkdir -p logs
SUMMARY_FILE="logs/${BASE_RUN_ID}_summary.txt"
: > "${SUMMARY_FILE}"

# Two-run 30-minute full-dataset compare: stripped baseline vs MAL memory-only.
DEFAULT_SWEEP_MODES=(
  no_novel
  memory_only
)

if [[ -n "${SWEEP_MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES=(${SWEEP_MODES})
else
  MODES=("${DEFAULT_SWEEP_MODES[@]}")
fi

reset_base_env() {
  export RUN_ID="${BASE_RUN_ID}"
  export TRAIN_BATCH_TOKENS="${BASE_TRAIN_BATCH_TOKENS}"
  export NUM_LAYERS="${BASE_NUM_LAYERS}"
  export MODEL_DIM="${BASE_MODEL_DIM}"
  export NUM_HEADS="${BASE_NUM_HEADS}"
  export NUM_KV_HEADS="${BASE_NUM_KV_HEADS}"
  export XSA_TOP_LAYERS="${BASE_XSA_TOP_LAYERS}"
  export MEMORY_CHUNK_SIZE="${BASE_MEMORY_CHUNK_SIZE}"
  export MEMORY_SURPRISE_THRESHOLD="${BASE_MEMORY_SURPRISE_THRESHOLD}"
  export D_MEMORY="${BASE_D_MEMORY}"
  export TRUNK_D_MEMORY="${BASE_TRUNK_D_MEMORY}"
  export CORE_D_MEMORY="${BASE_CORE_D_MEMORY}"
  export CORE_MEMORY_ONLY="${BASE_CORE_MEMORY_ONLY}"
  export MEMORY_WRITE_TOPK="${BASE_MEMORY_WRITE_TOPK}"
  export MEMORY_AUX_WEIGHT="${BASE_MEMORY_AUX_WEIGHT}"
  export TRAIN_MEMORY_MATCH_EVAL="${BASE_TRAIN_MEMORY_MATCH_EVAL}"
  export LOOP_CORE_ENABLED="${BASE_LOOP_CORE_ENABLED}"
  export LOOP_REPEATS="${BASE_LOOP_REPEATS}"
  export LOOP_ATTN_EVERY="${BASE_LOOP_ATTN_EVERY}"
  export NGRAM_CACHE_LAMBDA="${BASE_NGRAM_CACHE_LAMBDA}"
  export OGD_BIAS_LR="${BASE_OGD_BIAS_LR}"
  export VALUE_RESIDUAL="${BASE_VALUE_RESIDUAL}"
  export GATED_ATTENTION="${BASE_GATED_ATTENTION}"
  export POST_TRAINING_EVAL_MODE="${BASE_POST_TRAINING_EVAL_MODE}"
}

apply_mode() {
  local mode="$1"
  reset_base_env
  export RUN_ID="${BASE_RUN_ID}_${mode}"
  case "${mode}" in
    no_novel)
      export LOOP_CORE_ENABLED=0
      export D_MEMORY=0
      export TRUNK_D_MEMORY=0
      export CORE_D_MEMORY=0
      export CORE_MEMORY_ONLY=0
      export VALUE_RESIDUAL=0
      export GATED_ATTENTION=0
      export POST_TRAINING_EVAL_MODE=none
      ;;
    recurrence_only)
      export LOOP_CORE_ENABLED=1
      export D_MEMORY=0
      export TRUNK_D_MEMORY=0
      export CORE_D_MEMORY=0
      export CORE_MEMORY_ONLY=0
      export VALUE_RESIDUAL=0
      export GATED_ATTENTION=0
      export POST_TRAINING_EVAL_MODE=none
      ;;
    memory_only)
      export LOOP_CORE_ENABLED=0
      export D_MEMORY=8
      export TRUNK_D_MEMORY=8
      export CORE_D_MEMORY=16
      export CORE_MEMORY_ONLY="${BASE_CORE_MEMORY_ONLY}"
      export MEMORY_WRITE_TOPK=16
      export MEMORY_AUX_WEIGHT=0.10
      export TRAIN_MEMORY_MATCH_EVAL=1
      export VALUE_RESIDUAL=0
      export GATED_ATTENTION=0
      export POST_TRAINING_EVAL_MODE=none
      ;;
    value_residual_only)
      export LOOP_CORE_ENABLED=0
      export D_MEMORY=0
      export TRUNK_D_MEMORY=0
      export CORE_D_MEMORY=0
      export CORE_MEMORY_ONLY=0
      export VALUE_RESIDUAL=1
      export GATED_ATTENTION=0
      export POST_TRAINING_EVAL_MODE=none
      ;;
    gated_attention_only)
      export LOOP_CORE_ENABLED=0
      export D_MEMORY=0
      export TRUNK_D_MEMORY=0
      export CORE_D_MEMORY=0
      export CORE_MEMORY_ONLY=0
      export VALUE_RESIDUAL=0
      export GATED_ATTENTION=1
      export POST_TRAINING_EVAL_MODE=none
      ;;
    post_training_only)
      export LOOP_CORE_ENABLED=0
      export D_MEMORY=0
      export TRUNK_D_MEMORY=0
      export CORE_D_MEMORY=0
      export CORE_MEMORY_ONLY=0
      export VALUE_RESIDUAL=0
      export GATED_ATTENTION=0
      export POST_TRAINING_EVAL_MODE="${BASE_POST_TRAINING_EVAL_MODE}"
      ;;
    recurrence_memory)
      export LOOP_CORE_ENABLED=1
      export D_MEMORY="${BASE_D_MEMORY}"
      export TRUNK_D_MEMORY="${BASE_TRUNK_D_MEMORY}"
      export CORE_D_MEMORY="${BASE_CORE_D_MEMORY}"
      export CORE_MEMORY_ONLY="${BASE_CORE_MEMORY_ONLY}"
      export VALUE_RESIDUAL=0
      export GATED_ATTENTION=0
      export POST_TRAINING_EVAL_MODE=none
      ;;
    recurrence_memory_value_gated)
      export LOOP_CORE_ENABLED=1
      export D_MEMORY="${BASE_D_MEMORY}"
      export TRUNK_D_MEMORY="${BASE_TRUNK_D_MEMORY}"
      export CORE_D_MEMORY="${BASE_CORE_D_MEMORY}"
      export CORE_MEMORY_ONLY="${BASE_CORE_MEMORY_ONLY}"
      export VALUE_RESIDUAL=1
      export GATED_ATTENTION=1
      export POST_TRAINING_EVAL_MODE=none
      ;;
    best_config) ;;
    *)
      echo "Unknown sweep mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

extract_summary_line() {
  local mode="$1"
  local logfile="$2"
  python - "${mode}" "${logfile}" <<'PY'
import re
import sys
from pathlib import Path

mode = sys.argv[1]
log_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8", errors="replace")

patterns = {
    "raw": r"final_raw_model_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)",
    "post": r"final_raw_model_post_training_[A-Za-z0-9_]+_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)",
    "slide": r"final_raw_sliding_window_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)",
    "step": r"step:(\d+)/200000 val_loss:([0-9.]+) val_bpb:([0-9.]+)",
}

def last_match(pattern):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

raw = last_match(patterns["raw"])
post = last_match(patterns["post"])
slide = last_match(patterns["slide"])
step = last_match(patterns["step"])

status = "ok" if raw and slide else "fail"
raw_bpb = raw[1] if raw else "-"
post_bpb = post[1] if post else "-"
slide_bpb = slide[1] if slide else "-"
steps = step[0] if step else "-"
print(f"{mode}\t{status}\t{steps}\t{raw_bpb}\t{post_bpb}\t{slide_bpb}")
PY
}

{
  echo "Running 3080 sweep from ${repo_root}"
  echo "Base run id: ${BASE_RUN_ID}"
  echo "30-minute configs: ${#MODES[@]}"
  echo
  echo -e "mode\tstatus\tsteps\traw_bpb\tpost_bpb\tsliding_bpb"
} | tee -a "${SUMMARY_FILE}"

for mode in "${MODES[@]}"; do
  apply_mode "${mode}"
  logfile="logs/${RUN_ID}.txt"
  rm -f "${logfile}"
  echo
  echo "=== ${mode} ===" | tee -a "${SUMMARY_FILE}"
  echo "RUN_ID=${RUN_ID} LOOP_CORE_ENABLED=${LOOP_CORE_ENABLED} D_MEMORY=${D_MEMORY} TRUNK_D_MEMORY=${TRUNK_D_MEMORY} CORE_D_MEMORY=${CORE_D_MEMORY} CORE_MEMORY_ONLY=${CORE_MEMORY_ONLY} XSA_TOP_LAYERS=${XSA_TOP_LAYERS} LOOP_REPEATS=${LOOP_REPEATS} LOOP_ATTN_EVERY=${LOOP_ATTN_EVERY} POST_TRAINING_EVAL_MODE=${POST_TRAINING_EVAL_MODE} VALUE_RESIDUAL=${VALUE_RESIDUAL} GATED_ATTENTION=${GATED_ATTENTION}"
  if python ./train_gpt.py; then
    extract_summary_line "${mode}" "${logfile}" | tee -a "${SUMMARY_FILE}"
  else
    echo -e "${mode}\tfail\t-\t-\t-\t-" | tee -a "${SUMMARY_FILE}"
  fi
done

echo | tee -a "${SUMMARY_FILE}"
echo "Sorted by sliding_bpb:" | tee -a "${SUMMARY_FILE}"
python - "${SUMMARY_FILE}" <<'PY' | tee -a "${SUMMARY_FILE}"
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = []
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    parts = line.split("\t")
    if len(parts) != 6 or parts[0] in {"mode", "Sorted by sliding_bpb:"}:
        continue
    mode, status, steps, raw_bpb, post_bpb, slide_bpb = parts
    if status != "ok":
        continue
    try:
        key = float(slide_bpb)
    except ValueError:
        key = math.inf
    rows.append((key, mode, steps, raw_bpb, post_bpb, slide_bpb))

for _, mode, steps, raw_bpb, post_bpb, slide_bpb in sorted(rows):
    print(f"{mode}\tsteps={steps}\traw={raw_bpb}\tpost={post_bpb}\tsliding={slide_bpb}")
PY

echo
echo "Sweep summary written to ${SUMMARY_FILE}"
