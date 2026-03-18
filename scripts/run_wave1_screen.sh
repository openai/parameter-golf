#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launch_id_default="$(date +%Y%m%dT%H%M%S)"
launch_id="${LAUNCH_ID:-$launch_id_default}"
run_root="${RUN_ROOT:-$repo_root/runs/$launch_id}"
world_size="${WORLD_SIZE_OVERRIDE:-1}"
dry_run=0
git_sha="$(git -C "$repo_root" rev-parse HEAD)"
git_branch="$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)"

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      dry_run=1
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$run_root"

export DATA_PATH="${DATA_PATH:-$repo_root/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$repo_root/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export ITERATIONS="${ITERATIONS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
export SEED="${SEED:-1337}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.05}"
export MATRIX_LR="${MATRIX_LR:-0.04}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.85}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-500}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1e-8}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30}"
export ROPE_BASE="${ROPE_BASE:-10000}"

base_env_pairs=(
  "GIT_SHA=$git_sha"
  "GIT_BRANCH=$git_branch"
  "LAUNCH_ID=$launch_id"
  "DATA_PATH=$DATA_PATH"
  "TOKENIZER_PATH=$TOKENIZER_PATH"
  "VOCAB_SIZE=$VOCAB_SIZE"
  "NUM_LAYERS=$NUM_LAYERS"
  "MODEL_DIM=$MODEL_DIM"
  "NUM_HEADS=$NUM_HEADS"
  "NUM_KV_HEADS=$NUM_KV_HEADS"
  "MLP_MULT=$MLP_MULT"
  "TIE_EMBEDDINGS=$TIE_EMBEDDINGS"
  "TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN"
  "TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS"
  "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
  "ITERATIONS=$ITERATIONS"
  "WARMUP_STEPS=$WARMUP_STEPS"
  "WARMDOWN_ITERS=$WARMDOWN_ITERS"
  "TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY"
  "VAL_LOSS_EVERY=$VAL_LOSS_EVERY"
  "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
  "SEED=$SEED"
  "TIED_EMBED_LR=$TIED_EMBED_LR"
  "MATRIX_LR=$MATRIX_LR"
  "SCALAR_LR=$SCALAR_LR"
  "MUON_MOMENTUM=$MUON_MOMENTUM"
  "MUON_BACKEND_STEPS=$MUON_BACKEND_STEPS"
  "MUON_MOMENTUM_WARMUP_START=$MUON_MOMENTUM_WARMUP_START"
  "MUON_MOMENTUM_WARMUP_STEPS=$MUON_MOMENTUM_WARMUP_STEPS"
  "BETA1=$BETA1"
  "BETA2=$BETA2"
  "ADAM_EPS=$ADAM_EPS"
  "GRAD_CLIP_NORM=$GRAD_CLIP_NORM"
  "QK_GAIN_INIT=$QK_GAIN_INIT"
  "LOGIT_SOFTCAP=$LOGIT_SOFTCAP"
  "ROPE_BASE=$ROPE_BASE"
  "WORLD_SIZE=$world_size"
)

configs=(
  "wave1_00_base"
  "wave1_01_tiedlr_004 TIED_EMBED_LR=0.04"
  "wave1_02_tiedlr_006 TIED_EMBED_LR=0.06"
  "wave1_03_matrixlr_003 MATRIX_LR=0.03"
  "wave1_04_matrixlr_005 MATRIX_LR=0.05"
  "wave1_05_scalarlr_003 SCALAR_LR=0.03"
  "wave1_06_scalarlr_005 SCALAR_LR=0.05"
  "wave1_07_muonmom_093 MUON_MOMENTUM=0.93"
  "wave1_08_muonmom_097 MUON_MOMENTUM=0.97"
  "wave1_09_qkgain_125 QK_GAIN_INIT=1.25"
  "wave1_10_qkgain_175 QK_GAIN_INIT=1.75"
  "wave1_11_softcap_20 LOGIT_SOFTCAP=20"
  "wave1_12_softcap_40 LOGIT_SOFTCAP=40"
)

for config in "${configs[@]}"; do
  IFS=' ' read -r run_id maybe_key maybe_value <<<"$config"
  run_dir="$run_root/$run_id"
  mkdir -p "$run_dir"

  override_pairs=()
  if [[ -n "${maybe_key:-}" ]]; then
    override_pairs+=("$maybe_key")
    if [[ -n "${maybe_value:-}" ]]; then
      override_pairs+=("$maybe_value")
    fi
    rest="${config#"$run_id "}"
  else
    rest=""
  fi

  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$run_dir"
    printf 'env RUN_ID=%q ' "$run_id"
    printf '%q ' "${base_env_pairs[@]}"
    if [[ -n "$rest" ]]; then
      # shellcheck disable=SC2206
      extra_parts=($rest)
      printf '%q ' "${extra_parts[@]}"
    fi
    printf 'torchrun --standalone --nproc_per_node=%q %q\n' "$world_size" "$repo_root/train_gpt.py"
  } >"$run_dir/command.sh"
  chmod +x "$run_dir/command.sh"

  {
    printf 'RUN_ID=%s\n' "$run_id"
    printf '%s\n' "${base_env_pairs[@]}"
    if [[ -n "$rest" ]]; then
      # shellcheck disable=SC2206
      extra_parts=($rest)
      printf '%s\n' "${extra_parts[@]}"
    fi
  } | sort >"$run_dir/env.txt"

  if [[ "$dry_run" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$run_id"
    cat "$run_dir/command.sh"
    continue
  fi

  (
    cd "$run_dir"
    ./command.sh 2>&1 | tee train.log
  )
done
