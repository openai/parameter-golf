#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_PHASE="${RUN_PHASE:-smoke}"
DATA_PATH="${DATA_PATH:-/workspace/parameter-golf/data/datasets/fineweb10B_byte260}"
VOCAB_SIZE="${VOCAB_SIZE:-260}"
PATCH_SIZE="${PATCH_SIZE:-8}"
NUM_SLOTS="${NUM_SLOTS:-4}"
SLOT_BYTES="${SLOT_BYTES:-2}"
BYTE_EMBED_DIM="${BYTE_EMBED_DIM:-64}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-4096}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE-}"
VAL_MAX_SEQS="${VAL_MAX_SEQS-}"
FINAL_VAL_MAX_SEQS="${FINAL_VAL_MAX_SEQS-}"
LR="${LR:-0.0003}"
MATRIX_LR="${MATRIX_LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.01}"
PATCH_SUMMARY_WEIGHT="${PATCH_SUMMARY_WEIGHT:-0.1}"
MASKED_CONTEXT_PROB="${MASKED_CONTEXT_PROB:-0.15}"
EMA_DECAY="${EMA_DECAY:-0.99}"
SEED="${SEED:-42}"
RUN_FILTER="${RUN_FILTER:-}"
RUN_CHEAP="${RUN_CHEAP:-0}"
WINNER_BACKBONE="${WINNER_BACKBONE:-}"
WINNER_OBJECTIVE="${WINNER_OBJECTIVE:-}"
WINNER_HORIZONS="${WINNER_HORIZONS:-}"
WINNER_SCALES="${WINNER_SCALES:-}"
WINNER_MODEL_DIM="${WINNER_MODEL_DIM:-512}"
BACKBONE_GPU_COUNT="${BACKBONE_GPU_COUNT:-1}"
SCALE_BACKBONE_SECONDS_BY_GPU="${SCALE_BACKBONE_SECONDS_BY_GPU:-1}"
PROBE_PARALLEL_JOBS="${PROBE_PARALLEL_JOBS:-${BACKBONE_GPU_COUNT}}"
RUN_FULL_PROBE="${RUN_FULL_PROBE-}"

RESULT_ROOT="results/${RUN_PHASE}"
LOG_DIR="${RESULT_ROOT}/logs"
ARTIFACT_DIR="${RESULT_ROOT}/artifacts"
PROBE_CONFIG="${RESULT_ROOT}/probe_config.env"
VARIANTS_TSV="${RESULT_ROOT}/variants.tsv"
SUMMARY_JSON="${RESULT_ROOT}/summary.json"
CURVES_TSV="${RESULT_ROOT}/curves.tsv"
SCALING_FIT_JSON="${RESULT_ROOT}/scaling_fit.json"
REACH_MD="${RESULT_ROOT}/reach_baseline.md"

mkdir -p "${RESULT_ROOT}"
rm -f "${SUMMARY_JSON}" "${CURVES_TSV}" "${SCALING_FIT_JSON}" "${REACH_MD}" "${PROBE_CONFIG}" "${VARIANTS_TSV}"
rm -rf "${LOG_DIR}" "${ARTIFACT_DIR}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

default_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    printf -v "${name}" '%s' "${value}"
  fi
}

case "${RUN_PHASE}" in
  smoke)
    default_env TRAIN_SHARDS 1
    default_env TRAIN_BATCH_TOKENS 65536
    default_env VAL_BATCH_SIZE 65536
    default_env VAL_MAX_SEQS 16
    default_env FINAL_VAL_MAX_SEQS 16
    default_env BACKBONE_SECONDS 300
    default_env STRONG_PROBE_ITERATIONS 150
    default_env STRONG_PROBE_SECONDS 180
    default_env STRONG_PROBE_VAL_EVERY 40
    default_env STRONG_PROBE_LOG_EVERY 20
    ;;
  backbone_screen|objective_screen)
    default_env BACKBONE_SECONDS 1200
    default_env BACKBONE_VAL_EVERY 200
    default_env BACKBONE_LOG_EVERY 50
    default_env STOP_AFTER_LAST_CHECKPOINT 1
    default_env STRONG_PROBE_ITERATIONS 350
    default_env STRONG_PROBE_SECONDS 420
    default_env STRONG_PROBE_VAL_EVERY 70
    default_env STRONG_PROBE_LOG_EVERY 35
    ;;
  encoder_screen)
    default_env BACKBONE_SECONDS 0
    default_env BACKBONE_ITERATIONS 1200
    default_env BACKBONE_VAL_EVERY 400
    default_env BACKBONE_LOG_EVERY 100
    default_env STOP_AFTER_LAST_CHECKPOINT 0
    default_env STRONG_PROBE_ITERATIONS 180
    default_env STRONG_PROBE_SECONDS 240
    default_env STRONG_PROBE_VAL_EVERY 45
    default_env STRONG_PROBE_LOG_EVERY 30
    default_env RUN_FULL_PROBE 0
    ;;
  ablate|scale|data_scale)
    default_env BACKBONE_SECONDS 2700
    default_env BACKBONE_VAL_EVERY 500
    default_env BACKBONE_LOG_EVERY 100
    default_env STRONG_PROBE_ITERATIONS 700
    default_env STRONG_PROBE_SECONDS 900
    default_env STRONG_PROBE_VAL_EVERY 100
    default_env STRONG_PROBE_LOG_EVERY 50
    ;;
  *)
    echo "unsupported RUN_PHASE=${RUN_PHASE}" >&2
    exit 1
    ;;
esac

scale_backbone_seconds_if_needed() {
  if (( BACKBONE_SECONDS <= 0 )); then
    return
  fi
  if [[ "${SCALE_BACKBONE_SECONDS_BY_GPU}" != "1" ]]; then
    return
  fi
  if (( BACKBONE_GPU_COUNT <= 1 )); then
    return
  fi
  local secs="${BACKBONE_SECONDS}"
  local scaled=$(( (secs + BACKBONE_GPU_COUNT - 1) / BACKBONE_GPU_COUNT ))
  if (( scaled < 60 )); then
    scaled=60
  fi
  BACKBONE_SECONDS="${scaled}"
}

default_env BACKBONE_SECONDS 1200
default_env TRAIN_SHARDS 10
default_env TRAIN_BATCH_TOKENS 131072
default_env VAL_BATCH_SIZE 131072
default_env VAL_MAX_SEQS 256
default_env FINAL_VAL_MAX_SEQS 0
default_env BACKBONE_ITERATIONS 1000000
default_env STOP_AFTER_LAST_CHECKPOINT 0
default_env BACKBONE_VAL_EVERY 200
default_env BACKBONE_LOG_EVERY 50
default_env CHEAP_PROBE_ITERATIONS 200
default_env CHEAP_PROBE_SECONDS 300
default_env CHEAP_PROBE_VAL_EVERY 50
default_env CHEAP_PROBE_LOG_EVERY 25
default_env STRONG_PROBE_ITERATIONS 500
default_env STRONG_PROBE_SECONDS 600
default_env STRONG_PROBE_VAL_EVERY 100
default_env STRONG_PROBE_LOG_EVERY 50
default_env RUN_FULL_PROBE 1

scale_backbone_seconds_if_needed

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

write_header() {
  printf 'run_id\tbackbone_kind\tpatch_encoder_kind\tobjective_kind\tsize_label\tmodel_dim\tnum_layers\tnum_heads\tnum_kv_heads\tff_mult\ttrain_shards\ttrain_batch_tokens\tbackbone_seconds\tpredict_horizons\tmultiscale_groups\tseed\tnotes\n' > "${VARIANTS_TSV}"
}

emit_variant() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${VARIANTS_TSV}"
}

backbone_dims() {
  local backbone="$1"
  local size="$2"
  case "${size}" in
    smoke) echo "256 4 4 2 3" ;;
    anchor) echo "512 8 8 4 3" ;;
    s384) echo "384 8 6 3 3" ;;
    s512) echo "512 8 8 4 3" ;;
    s768) echo "768 8 12 6 3" ;;
    s1024) echo "1024 8 16 8 3" ;;
    *)
      echo "unsupported size=${size}" >&2
      exit 1
      ;;
  esac
}

resolve_best_field() {
  local summary_path="$1"
  local field="$2"
  python3 - "${summary_path}" "${field}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ranking = summary.get("ranking", [])
if not ranking:
    raise SystemExit("missing ranking")
best = ranking[0]
variant = best.get("variant", {})
backbone = best.get("backbone") or {}
config = backbone.get("config", {})
field = sys.argv[2]
for source in (variant, backbone, config):
    if field in source and source[field] not in (None, ""):
        value = source[field]
        if isinstance(value, (list, tuple)):
            print(",".join(str(item) for item in value))
        else:
            print(value)
        raise SystemExit(0)
raise SystemExit(f"missing field {field}")
PY
}

resolve_default_winner_backbone() {
  if [[ -n "${WINNER_BACKBONE}" ]]; then
    printf '%s\n' "${WINNER_BACKBONE}"
  else
    resolve_best_field "results/backbone_screen/summary.json" "backbone_kind"
  fi
}

resolve_default_winner_objective() {
  if [[ -n "${WINNER_OBJECTIVE}" ]]; then
    printf '%s\n' "${WINNER_OBJECTIVE}"
  else
    resolve_best_field "results/objective_screen/summary.json" "objective_kind"
  fi
}

resolve_default_winner_horizons() {
  if [[ -n "${WINNER_HORIZONS}" ]]; then
    printf '%s\n' "${WINNER_HORIZONS}"
  elif [[ -f "results/ablate/summary.json" ]]; then
    resolve_best_field "results/ablate/summary.json" "predict_horizons"
  else
    printf '1\n'
  fi
}

resolve_default_winner_scales() {
  if [[ -n "${WINNER_SCALES}" ]]; then
    printf '%s\n' "${WINNER_SCALES}"
  elif [[ -f "results/ablate/summary.json" ]]; then
    resolve_best_field "results/ablate/summary.json" "multiscale_groups"
  else
    printf '8\n'
  fi
}

append_variants_for_phase() {
  case "${RUN_PHASE}" in
    smoke)
      read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims transformer_rope_gqa_base smoke)"
      emit_variant "smoke_slot_l2" "transformer_rope_gqa_base" "mlp_baseline" "slot_l2" "smoke" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "smoke base + slot_l2"
      emit_variant "smoke_slot_cosine" "transformer_rope_gqa_base" "mlp_baseline" "slot_cosine" "smoke" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "smoke base + slot_cosine"
      emit_variant "smoke_slot_ema_teacher" "transformer_rope_gqa_base" "mlp_baseline" "slot_ema_teacher" "smoke" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "smoke base + slot_ema_teacher"
      ;;
    backbone_screen)
      read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims transformer_rope_gqa_base anchor)"
      for backbone in transformer_rope_gqa_base transformer_rope_gqa_convstem transformer_rope_gqa_localglobal; do
        emit_variant "backbone_${backbone}" "${backbone}" "mlp_baseline" "slot_l2" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "20-minute backbone screen"
      done
      ;;
    objective_screen)
      local backbone
      backbone="$(resolve_default_winner_backbone)"
      read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims "${backbone}" anchor)"
      for objective in slot_l2 slot_cosine slot_vicreg slot_ema_teacher masked_slot_jepa; do
        emit_variant "objective_${backbone}_${objective}" "${backbone}" "mlp_baseline" "${objective}" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "20-minute objective screen"
      done
      ;;
    encoder_screen)
      read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims transformer_rope_gqa_localglobal anchor)"
      for patch_encoder_kind in mlp_baseline patch_transformer latent_queries conv_patch; do
        emit_variant "encoder_transformer_rope_gqa_localglobal_${patch_encoder_kind}" "transformer_rope_gqa_localglobal" "${patch_encoder_kind}" "slot_ema_teacher" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "15-minute encoder screen"
      done
      ;;
    ablate)
      local backbone objective
      backbone="$(resolve_default_winner_backbone)"
      objective="$(resolve_default_winner_objective)"
      read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims "${backbone}" anchor)"
      emit_variant "ablate_${backbone}_${objective}_h1_s8" "${backbone}" "mlp_baseline" "${objective}" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8" "${SEED}" "slot target single horizon single scale"
      emit_variant "ablate_${backbone}_${objective}_h1416_s8" "${backbone}" "mlp_baseline" "${objective}" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1,4,16" "8" "${SEED}" "slot target multihorizon"
      emit_variant "ablate_${backbone}_${objective}_h1_s832" "${backbone}" "mlp_baseline" "${objective}" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1" "8,32" "${SEED}" "slot target multiscale"
      emit_variant "ablate_${backbone}_${objective}_h1416_s832" "${backbone}" "mlp_baseline" "${objective}" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "1,4,16" "8,32" "${SEED}" "slot target multihorizon+multiscale"
      ;;
    scale)
      local backbone objective horizons scales
      backbone="$(resolve_default_winner_backbone)"
      objective="$(resolve_default_winner_objective)"
      horizons="$(resolve_default_winner_horizons)"
      scales="$(resolve_default_winner_scales)"
      for size in s384 s512 s768 s1024; do
        read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims "${backbone}" "${size}")"
        emit_variant "scale_${backbone}_${objective}_${size}" "${backbone}" "mlp_baseline" "${objective}" "${size}" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${TRAIN_SHARDS}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "${horizons}" "${scales}" "${SEED}" "45-minute scaling run"
      done
      ;;
    data_scale)
      local backbone objective horizons scales
      backbone="$(resolve_default_winner_backbone)"
      objective="$(resolve_default_winner_objective)"
      horizons="$(resolve_default_winner_horizons)"
      scales="$(resolve_default_winner_scales)"
      read -r model_dim num_layers num_heads num_kv_heads ff_mult <<<"$(backbone_dims "${backbone}" anchor)"
      for shards in 1 3 10; do
        emit_variant "data_${backbone}_${objective}_shards${shards}" "${backbone}" "mlp_baseline" "${objective}" "anchor" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" "${shards}" "${TRAIN_BATCH_TOKENS}" "${BACKBONE_SECONDS}" "${horizons}" "${scales}" "${SEED}" "45-minute data scaling run"
      done
      ;;
  esac
}

write_header
append_variants_for_phase

cat > "${PROBE_CONFIG}" <<EOF
RUN_PHASE=${RUN_PHASE}
DATA_PATH=${DATA_PATH}
VOCAB_SIZE=${VOCAB_SIZE}
PATCH_SIZE=${PATCH_SIZE}
NUM_SLOTS=${NUM_SLOTS}
SLOT_BYTES=${SLOT_BYTES}
BYTE_EMBED_DIM=${BYTE_EMBED_DIM}
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE}
VAL_MAX_SEQS=${VAL_MAX_SEQS}
FINAL_VAL_MAX_SEQS=${FINAL_VAL_MAX_SEQS}
LR=${LR}
MATRIX_LR=${MATRIX_LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM}
MIN_LR_RATIO=${MIN_LR_RATIO}
SIGREG_WEIGHT=${SIGREG_WEIGHT}
PATCH_SUMMARY_WEIGHT=${PATCH_SUMMARY_WEIGHT}
MASKED_CONTEXT_PROB=${MASKED_CONTEXT_PROB}
EMA_DECAY=${EMA_DECAY}
TRAIN_SHARDS=${TRAIN_SHARDS}
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS}
BACKBONE_SECONDS=${BACKBONE_SECONDS}
SCALE_BACKBONE_SECONDS_BY_GPU=${SCALE_BACKBONE_SECONDS_BY_GPU}
STOP_AFTER_LAST_CHECKPOINT=${STOP_AFTER_LAST_CHECKPOINT}
SEED=${SEED}
EOF

checkpoint_bytes_for_phase() {
  case "${RUN_PHASE}" in
    smoke) printf '125000000\n' ;;
    backbone_screen|objective_screen) printf '125000000,250000000,500000000,1000000000\n' ;;
    encoder_screen) printf '\n' ;;
    *) printf '125000000,250000000,500000000,1000000000,2000000000,4000000000\n' ;;
  esac
}

run_backbone_variant() {
  local run_id="$1"
  local backbone_kind="$2"
  local patch_encoder_kind="$3"
  local objective_kind="$4"
  local model_dim="$5"
  local num_layers="$6"
  local num_heads="$7"
  local num_kv_heads="$8"
  local ff_mult="$9"
  local train_shards="${10}"
  local train_batch_tokens="${11}"
  local backbone_seconds="${12}"
  local predict_horizons="${13}"
  local multiscale_groups="${14}"
  local seed="${15}"
  local checkpoint_bytes="${16}"

  local cmd=(python3 -X faulthandler train_gpt.py)
  if (( BACKBONE_GPU_COUNT > 1 )); then
    cmd=(torchrun --standalone --nnodes=1 --nproc_per_node="${BACKBONE_GPU_COUNT}" train_gpt.py)
  fi

  env \
    RUN_MODE=backbone \
    RUN_ID="${run_id}" \
    RUN_PHASE="${RUN_PHASE}" \
    OUTPUT_ROOT="${RESULT_ROOT}" \
    DATA_PATH="${DATA_PATH}" \
    VOCAB_SIZE="${VOCAB_SIZE}" \
    PAD_ID=0 BOS_ID=1 EOS_ID=2 UNK_ID=3 \
    BACKBONE_KIND="${backbone_kind}" \
    PATCH_ENCODER_KIND="${patch_encoder_kind}" \
    OBJECTIVE_KIND="${objective_kind}" \
    PATCH_SIZE="${PATCH_SIZE}" \
    NUM_SLOTS="${NUM_SLOTS}" \
    SLOT_BYTES="${SLOT_BYTES}" \
    BYTE_EMBED_DIM="${BYTE_EMBED_DIM}" \
    MODEL_DIM="${model_dim}" \
    NUM_LAYERS="${num_layers}" \
    NUM_HEADS="${num_heads}" \
    NUM_KV_HEADS="${num_kv_heads}" \
    FF_MULT="${ff_mult}" \
    TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}" \
    TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
    TRAIN_SHARDS="${train_shards}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    VAL_MAX_SEQS="${VAL_MAX_SEQS}" \
    FINAL_VAL_MAX_SEQS="${FINAL_VAL_MAX_SEQS}" \
    ITERATIONS="${BACKBONE_ITERATIONS}" \
    MAX_WALLCLOCK_SECONDS="${backbone_seconds}" \
    VAL_LOSS_EVERY="${BACKBONE_VAL_EVERY}" \
    TRAIN_LOG_EVERY="${BACKBONE_LOG_EVERY}" \
    LR="${LR}" \
    MATRIX_LR="${MATRIX_LR}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    GRAD_CLIP_NORM="${GRAD_CLIP_NORM}" \
    MIN_LR_RATIO="${MIN_LR_RATIO}" \
    SEED="${seed}" \
    JEPA_WEIGHT=1.0 \
    SIGREG_WEIGHT="${SIGREG_WEIGHT}" \
    PATCH_SUMMARY_WEIGHT="${PATCH_SUMMARY_WEIGHT}" \
    MASKED_CONTEXT_PROB="${MASKED_CONTEXT_PROB}" \
    EMA_DECAY="${EMA_DECAY}" \
    PREDICT_HORIZONS="${predict_horizons}" \
    MULTISCALE_GROUPS="${multiscale_groups}" \
    CHECKPOINT_BYTES="${checkpoint_bytes}" \
    STOP_AFTER_LAST_CHECKPOINT="${STOP_AFTER_LAST_CHECKPOINT}" \
    "${cmd[@]}"
}

checkpoint_lines() {
  local run_id="$1"
  python3 - "${RESULT_ROOT}" "${run_id}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads((Path(sys.argv[1]) / "artifacts" / sys.argv[2] / "backbone_run.json").read_text(encoding="utf-8"))
for row in payload["checkpoint_records"]:
    print(f"{row['label']}\t{row['path']}")
PY
}

best_probe_checkpoint_label() {
  local run_id="$1"
  local probe_kind="$2"
  local probe_val_mode="$3"
  python3 - "${RESULT_ROOT}" "${run_id}" "${probe_kind}" "${probe_val_mode}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "artifacts" / sys.argv[2] / "probe_results"
probe_kind = sys.argv[3]
probe_val_mode = sys.argv[4]
best = None
for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("probe_kind") != probe_kind or payload.get("probe_val_mode") != probe_val_mode:
        continue
    score = float(payload["best_val_bpb"])
    if best is None or score < best[0]:
        best = (score, payload["checkpoint_label"])
if best is None:
    raise SystemExit("no matching probe results found")
print(best[1])
PY
}

run_probe_variant() {
  local run_id="$1"
  local checkpoint_path="$2"
  local probe_kind="$3"
  local probe_val_mode="$4"
  local probe_iterations="$5"
  local probe_seconds="$6"
  local probe_val_every="$7"
  local probe_log_every="$8"
  local probe_train_shards="$9"
  local train_batch_tokens="${10}"
  local seed="${11}"

  env \
    RUN_MODE=probe \
    RUN_ID="${run_id}" \
    RUN_PHASE="${RUN_PHASE}" \
    OUTPUT_ROOT="${RESULT_ROOT}" \
    DATA_PATH="${DATA_PATH}" \
    VOCAB_SIZE="${VOCAB_SIZE}" \
    PATCH_SIZE="${PATCH_SIZE}" \
    NUM_SLOTS="${NUM_SLOTS}" \
    SLOT_BYTES="${SLOT_BYTES}" \
    TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    VAL_MAX_SEQS="${VAL_MAX_SEQS}" \
    FINAL_VAL_MAX_SEQS="${FINAL_VAL_MAX_SEQS}" \
    PROBE_KIND="${probe_kind}" \
    PROBE_CHECKPOINT="${checkpoint_path}" \
    PROBE_DETACH_BACKBONE=1 \
    PROBE_VAL_MODE="${probe_val_mode}" \
    PROBE_TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
    PROBE_TRAIN_SHARDS="${probe_train_shards}" \
    PROBE_ITERATIONS="${probe_iterations}" \
    PROBE_MAX_WALLCLOCK_SECONDS="${probe_seconds}" \
    PROBE_VAL_LOSS_EVERY="${probe_val_every}" \
    PROBE_TRAIN_LOG_EVERY="${probe_log_every}" \
    PROBE_LR=0.0005 \
    PROBE_WEIGHT_DECAY=0.01 \
    PROBE_GRAD_CLIP_NORM=1.0 \
    DECODER_HIDDEN=512 \
    DECODER_LAYERS=4 \
    DECODER_HEADS=8 \
    DECODER_NUM_KV_HEADS=4 \
    DECODER_FF_MULT=2 \
    SEED="${seed}" \
    python3 -X faulthandler train_gpt.py
}

declare -a ACTIVE_PROBE_PIDS=()
declare -a ACTIVE_PROBE_GPUS=()
PROBE_FAILED=0

probe_gpu_ids() {
  local total="${PROBE_PARALLEL_JOBS}"
  if (( total < 1 )); then
    total=1
  fi
  local gpu
  for (( gpu = 0; gpu < total; gpu++ )); do
    printf '%s\n' "${gpu}"
  done
}

prune_probe_jobs() {
  local idx=0
  while (( idx < ${#ACTIVE_PROBE_PIDS[@]} )); do
    local pid="${ACTIVE_PROBE_PIDS[idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
      ((idx += 1))
      continue
    fi
    if ! wait "${pid}"; then
      PROBE_FAILED=1
    fi
    ACTIVE_PROBE_PIDS=("${ACTIVE_PROBE_PIDS[@]:0:idx}" "${ACTIVE_PROBE_PIDS[@]:idx+1}")
    ACTIVE_PROBE_GPUS=("${ACTIVE_PROBE_GPUS[@]:0:idx}" "${ACTIVE_PROBE_GPUS[@]:idx+1}")
  done
}

wait_for_probe_slot() {
  local limit="${PROBE_PARALLEL_JOBS}"
  if (( limit < 1 )); then
    limit=1
  fi
  while (( ${#ACTIVE_PROBE_PIDS[@]} >= limit )); do
    sleep 1
    prune_probe_jobs
  done
}

wait_for_all_probe_jobs() {
  while (( ${#ACTIVE_PROBE_PIDS[@]} > 0 )); do
    sleep 1
    prune_probe_jobs
  done
  if (( PROBE_FAILED != 0 )); then
    echo "one or more probe jobs failed" >&2
    exit 1
  fi
}

next_probe_gpu() {
  local gpu used active
  while IFS= read -r gpu; do
    used=0
    for active in "${ACTIVE_PROBE_GPUS[@]}"; do
      if [[ "${active}" == "${gpu}" ]]; then
        used=1
        break
      fi
    done
    if (( used == 0 )); then
      printf '%s\n' "${gpu}"
      return
    fi
  done < <(probe_gpu_ids)
  printf '0\n'
}

launch_probe_job() {
  local gpu="$1"
  shift
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    run_probe_variant "$@"
  ) &
  ACTIVE_PROBE_PIDS+=("$!")
  ACTIVE_PROBE_GPUS+=("${gpu}")
}

run_variant_pipeline() {
  local run_id="$1"
  local backbone_kind="$2"
  local patch_encoder_kind="$3"
  local objective_kind="$4"
  local size_label="$5"
  local model_dim="$6"
  local num_layers="$7"
  local num_heads="$8"
  local num_kv_heads="$9"
  local ff_mult="${10}"
  local train_shards="${11}"
  local train_batch_tokens="${12}"
  local backbone_seconds="${13}"
  local predict_horizons="${14}"
  local multiscale_groups="${15}"
  local seed="${16}"

  if [[ -n "${RUN_FILTER}" ]]; then
    case ",${RUN_FILTER}," in
      *,"${run_id}",*) ;;
      *) return ;;
    esac
  fi

  local checkpoint_bytes
  checkpoint_bytes="$(checkpoint_bytes_for_phase)"

  run_backbone_variant \
    "${run_id}" "${backbone_kind}" "${patch_encoder_kind}" "${objective_kind}" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" \
    "${train_shards}" "${train_batch_tokens}" "${backbone_seconds}" "${predict_horizons}" "${multiscale_groups}" "${seed}" "${checkpoint_bytes}"

  if (( RUN_CHEAP == 1 )); then
    while IFS=$'\t' read -r _ checkpoint_path; do
      wait_for_probe_slot
      launch_probe_job "$(next_probe_gpu)" "${run_id}" "${checkpoint_path}" cheap proxy "${CHEAP_PROBE_ITERATIONS}" "${CHEAP_PROBE_SECONDS}" "${CHEAP_PROBE_VAL_EVERY}" "${CHEAP_PROBE_LOG_EVERY}" "${train_shards}" "${train_batch_tokens}" "${seed}"
    done < <(checkpoint_lines "${run_id}")
    wait_for_all_probe_jobs
  fi

  while IFS=$'\t' read -r _ checkpoint_path; do
    wait_for_probe_slot
    launch_probe_job "$(next_probe_gpu)" "${run_id}" "${checkpoint_path}" strong proxy "${STRONG_PROBE_ITERATIONS}" "${STRONG_PROBE_SECONDS}" "${STRONG_PROBE_VAL_EVERY}" "${STRONG_PROBE_LOG_EVERY}" "${train_shards}" "${train_batch_tokens}" "${seed}"
  done < <(checkpoint_lines "${run_id}")
  wait_for_all_probe_jobs

  if (( RUN_FULL_PROBE == 1 )); then
    local best_proxy_label final_checkpoint_path best_proxy_checkpoint_path
    best_proxy_label="$(best_probe_checkpoint_label "${run_id}" strong proxy)"
    final_checkpoint_path="$(checkpoint_lines "${run_id}" | awk -F'\t' '$1=="final" {print $2; exit}')"
    best_proxy_checkpoint_path="$(checkpoint_lines "${run_id}" | awk -F'\t' -v label="${best_proxy_label}" '$1==label {print $2; exit}')"
    wait_for_probe_slot
    launch_probe_job "$(next_probe_gpu)" "${run_id}" "${best_proxy_checkpoint_path}" strong full "${STRONG_PROBE_ITERATIONS}" "${STRONG_PROBE_SECONDS}" "${STRONG_PROBE_VAL_EVERY}" "${STRONG_PROBE_LOG_EVERY}" "${train_shards}" "${train_batch_tokens}" "${seed}"
    if [[ "${best_proxy_label}" != "final" ]]; then
      wait_for_probe_slot
      launch_probe_job "$(next_probe_gpu)" "${run_id}" "${final_checkpoint_path}" strong full "${STRONG_PROBE_ITERATIONS}" "${STRONG_PROBE_SECONDS}" "${STRONG_PROBE_VAL_EVERY}" "${STRONG_PROBE_LOG_EVERY}" "${train_shards}" "${train_batch_tokens}" "${seed}"
    fi
    wait_for_all_probe_jobs
  fi
}

tail -n +2 "${VARIANTS_TSV}" | while IFS=$'\t' read -r run_id backbone_kind patch_encoder_kind objective_kind size_label model_dim num_layers num_heads num_kv_heads ff_mult train_shards train_batch_tokens backbone_seconds predict_horizons multiscale_groups seed notes; do
  run_variant_pipeline \
    "${run_id}" "${backbone_kind}" "${patch_encoder_kind}" "${objective_kind}" "${size_label}" "${model_dim}" "${num_layers}" "${num_heads}" "${num_kv_heads}" "${ff_mult}" \
    "${train_shards}" "${train_batch_tokens}" "${backbone_seconds}" "${predict_horizons}" "${multiscale_groups}" "${seed}"
done

python3 summarize_sweep.py --phase-root "${RESULT_ROOT}" --summary-out "${SUMMARY_JSON}" --curves-out "${CURVES_TSV}" --scaling-fit-out "${SCALING_FIT_JSON}" --reach-out "${REACH_MD}"
