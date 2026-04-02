#!/bin/bash
set -euo pipefail
# ================================================================
# BW17_DGXSpark_Cadence_Longform
#
# One-command interaction suite around the stable 9F crawler stack:
#   1) RAPID stage (small-token local screening)
#   2) LONGFORM stage (replay top rapid candidates at 600s)
#   3) POST_WINDOW quant bake-off on best LONGFORM checkpoint
#
# Usage:
#   SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW17_DGXSpark_Cadence_Longform/run_ablation_sequence.sh
#
# Optional:
#   RUN_LONGFORM=0 ...  # rapid-only
#   RUN_QUANT=0 ...     # skip quant stage
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${TRAIN_PY:-${REPO_ROOT}/crawler/2026-04-01_BW12_Interaction_2k/train_gpt_ablate.py}"
RUN_TAG="BW17DGX"
TS="$(date +%Y%m%d_%H%M%S)"

RUN_LONGFORM="${RUN_LONGFORM:-1}"
LONGFORM_TOPK="${LONGFORM_TOPK:-2}"
RUN_QUANT="${RUN_QUANT:-1}"
RUN_LOOP_AWARE_GPTQ="${RUN_LOOP_AWARE_GPTQ:-0}"

RAPID_MAX_WALLCLOCK_SECONDS="${RAPID_MAX_WALLCLOCK_SECONDS:-240}"
RAPID_ITERATIONS="${RAPID_ITERATIONS:-12000}"
RAPID_TRAIN_BATCH_TOKENS="${RAPID_TRAIN_BATCH_TOKENS:-393216}"
RAPID_VAL_BATCH_SIZE="${RAPID_VAL_BATCH_SIZE:-262144}"

LONG_MAX_WALLCLOCK_SECONDS="${LONG_MAX_WALLCLOCK_SECONDS:-600}"
LONG_ITERATIONS="${LONG_ITERATIONS:-20000}"
LONG_TRAIN_BATCH_TOKENS="${LONG_TRAIN_BATCH_TOKENS:-786432}"
LONG_VAL_BATCH_SIZE="${LONG_VAL_BATCH_SIZE:-524288}"

CRAWLER_QUANT_INT8_DEFAULT="${CRAWLER_QUANT_INT8_DEFAULT:-0}" # 0 = more size-safe for 16MB track

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

BASE_ENV=(
    SEED="${SEED}"
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=9
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8_DEFAULT}"
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    GPTQ_CAL_SAMPLES=128
    GPTQ_CAL_SEQ_LEN=2048
    NPROC_PER_NODE="${NPROC}"
)

RAPID_ENV=(
    ITERATIONS="${RAPID_ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${RAPID_MAX_WALLCLOCK_SECONDS}"
    TRAIN_BATCH_TOKENS="${RAPID_TRAIN_BATCH_TOKENS}"
    VAL_BATCH_SIZE="${RAPID_VAL_BATCH_SIZE}"
)

LONG_ENV=(
    ITERATIONS="${LONG_ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${LONG_MAX_WALLCLOCK_SECONDS}"
    TRAIN_BATCH_TOKENS="${LONG_TRAIN_BATCH_TOKENS}"
    VAL_BATCH_SIZE="${LONG_VAL_BATCH_SIZE}"
)

declare -A CONTROL_INT6=()
declare -A RUN_INT6=()
declare -A RUN_CKPT=()
declare -A RUN_LOG=()

BEST_LONG_ARM=""
BEST_LONG_SOURCE=""
BEST_LONG_INT6=""
BEST_LONG_CKPT=""

{
    echo -e "phase\tlane\tarm\tsource_arm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tgptq_layers\tgptq_cal_sec\tdelta_vs_control\tlog\tckpt"
} > "${SUMMARY}"

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

is_numeric() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

calc_delta() {
    local control="$1"
    local value="$2"
    if ! is_numeric "${control}" || ! is_numeric "${value}"; then
        echo "?"
        return
    fi
    python3 - "${control}" "${value}" <<'PY'
import sys
c = float(sys.argv[1])
v = float(sys.argv[2])
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

is_better() {
    local candidate="$1"
    local incumbent="$2"
    if ! is_numeric "${candidate}"; then
        return 1
    fi
    if ! is_numeric "${incumbent}"; then
        return 0
    fi
    python3 - "${candidate}" "${incumbent}" <<'PY'
import sys
cand = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if cand < best else 1)
PY
}

arm_desc() {
    case "$1" in
        BW17DGX-00) echo "control: 9F, 1 crawler layer, loops=3, inst=32, rope=(9,1,1), smear=0" ;;
        BW17DGX-01) echo "cadence down: loops=2" ;;
        BW17DGX-02) echo "cadence up: loops=4" ;;
        BW17DGX-03) echo "cadence hard: loops=5" ;;
        BW17DGX-04) echo "depth/cadence split: num_crawler_layers=2, loops=2" ;;
        BW17DGX-05) echo "rope battery shift: loop_rope_scales=(16,4,1)" ;;
        BW17DGX-06) echo "instruction width up: inst_dim=64" ;;
        BW17DGX-07) echo "smear gate on: crawler_loop_smear=1" ;;
        *) echo "unknown-arm" ;;
    esac
}

arm_overrides() {
    case "$1" in
        BW17DGX-00)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=3
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-01)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=2
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-02)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=4
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-03)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=5
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-04)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=2
CRAWLER_LOOPS=2
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-05)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=3
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=16,4,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-06)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=3
INST_DIM=64
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
EOF
            ;;
        BW17DGX-07)
            cat <<'EOF'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=3
INST_DIM=32
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=1
EOF
            ;;
        *)
            echo "ERROR: unknown source_arm=$1" >&2
            exit 1
            ;;
    esac
}

run_arm() {
    local phase="$1"; shift
    local lane="$1"; shift
    local arm="$1"; shift
    local source_arm="$1"; shift
    local desc="$1"; shift
    local mode="$1"; shift
    local ctrl_key="$1"; shift
    local is_control="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"
    local run_env=("${BASE_ENV[@]}")
    local overrides=()

    case "${mode}" in
        RAPID) run_env+=("${RAPID_ENV[@]}") ;;
        LONG|QUANT) run_env+=("${LONG_ENV[@]}") ;;
        *)
            echo "ERROR: unknown mode=${mode}" >&2
            exit 1
            ;;
    esac

    mapfile -t overrides < <(arm_overrides "${source_arm}" | sed '/^$/d')
    run_env+=("${overrides[@]}")
    run_env+=("${extra_env[@]}")

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${phase} ${lane} ${arm}: ${desc}"
    echo "  source_arm: ${source_arm}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    env "${run_env[@]}" \
      "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local params raw int6 step_ms bytes gptq_layers gptq_cal_sec delta ckpt
    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${log}")
    gptq_layers=$(extract_metric 'gptq_quantize: \K[0-9]+' "${log}")
    gptq_cal_sec=$(extract_metric 'gptq:(?:loop-aware )?calibrated [0-9]+ layers in \K[0-9.]+' "${log}")

    if [[ -z "${params}" ]]; then params="?"; fi
    if [[ -z "${raw}" ]]; then raw="?"; fi
    if [[ -z "${int6}" ]]; then int6="?"; fi
    if [[ -z "${step_ms}" ]]; then step_ms="?"; fi
    if [[ -z "${bytes}" ]]; then bytes="?"; fi
    if [[ -z "${gptq_layers}" ]]; then gptq_layers="0"; fi
    if [[ -z "${gptq_cal_sec}" ]]; then gptq_cal_sec="-"; fi

    if [[ "${is_control}" == "1" ]]; then
        CONTROL_INT6["${ctrl_key}"]="${int6}"
    fi
    delta=$(calc_delta "${CONTROL_INT6[${ctrl_key}]:-}" "${int6}")

    ckpt="-"
    if [[ "${lane}" == "WINDOW" && -f "${REPO_ROOT}/final_model.pt" ]]; then
        ckpt="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"
        cp -f "${REPO_ROOT}/final_model.pt" "${ckpt}"
    fi

    RUN_INT6["${arm}"]="${int6}"
    RUN_CKPT["${arm}"]="${ckpt}"
    RUN_LOG["${arm}"]="${log}"

    if [[ "${phase}" == "LONGFORM" && "${lane}" == "WINDOW" ]]; then
        if is_better "${int6}" "${BEST_LONG_INT6}"; then
            BEST_LONG_ARM="${arm}"
            BEST_LONG_SOURCE="${source_arm}"
            BEST_LONG_INT6="${int6}"
            BEST_LONG_CKPT="${ckpt}"
        fi
    fi

    echo -e "${phase}\t${lane}\t${arm}\t${source_arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${log}\t${ckpt}" >> "${SUMMARY}"
    echo "  ${arm}: raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} delta_vs_ctrl=${delta}"
}

pick_top_rapid_arms() {
    python3 - "${SUMMARY}" "${LONGFORM_TOPK}" <<'PY'
import csv, sys

summary = sys.argv[1]
topk = int(sys.argv[2])
rows = []
with open(summary, newline="") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        if row.get("phase") != "RAPID" or row.get("lane") != "WINDOW":
            continue
        arm = row.get("source_arm", "")
        if arm == "BW17DGX-00":
            continue
        try:
            score = float(row.get("int6_sw_bpb", "nan"))
        except ValueError:
            continue
        if score != score:
            continue
        rows.append((score, arm))

rows.sort(key=lambda x: x[0])
seen = set()
picked = []
for _, arm in rows:
    if arm in seen:
        continue
    seen.add(arm)
    picked.append(arm)
    if len(picked) >= topk:
        break

for arm in picked:
    print(arm)
PY
}

# ----------------------------------------------------------------
# 1) RAPID stage: small-token local interaction screening.
# ----------------------------------------------------------------
RAPID_SOURCE_ARMS=(
    BW17DGX-00
    BW17DGX-01
    BW17DGX-02
    BW17DGX-03
    BW17DGX-04
    BW17DGX-05
    BW17DGX-06
    BW17DGX-07
)

for i in "${!RAPID_SOURCE_ARMS[@]}"; do
    src="${RAPID_SOURCE_ARMS[$i]}"
    arm="${src}"
    ctrl=0
    if [[ "${src}" == "BW17DGX-00" ]]; then
        ctrl=1
    fi
    run_arm "RAPID" "WINDOW" "${arm}" "${src}" "$(arm_desc "${src}")" "RAPID" "rapid" "${ctrl}"
done

TOP_RAPID_ARMS=()
if mapfile -t TOP_RAPID_ARMS < <(pick_top_rapid_arms); then
    :
fi

echo ""
echo "Top RAPID candidates (excluding control): ${TOP_RAPID_ARMS[*]:-(none)}"

# ----------------------------------------------------------------
# 2) LONGFORM stage: replay top RAPID candidates at full horizon.
# ----------------------------------------------------------------
if [[ "${RUN_LONGFORM}" == "1" ]]; then
    LONG_SOURCE_ARMS=(BW17DGX-00)
    for src in "${TOP_RAPID_ARMS[@]}"; do
        [[ "${src}" == "BW17DGX-00" ]] && continue
        LONG_SOURCE_ARMS+=("${src}")
    done

    # De-duplicate while preserving order.
    declare -A _seen=()
    LONG_SOURCE_UNIQ=()
    for src in "${LONG_SOURCE_ARMS[@]}"; do
        if [[ -z "${_seen[${src}]:-}" ]]; then
            _seen["${src}"]=1
            LONG_SOURCE_UNIQ+=("${src}")
        fi
    done
    unset _seen

    idx=0
    for src in "${LONG_SOURCE_UNIQ[@]}"; do
        arm="$(printf "BW17L-%02d" "${idx}")"
        ctrl=0
        if [[ "${idx}" -eq 0 ]]; then
            ctrl=1
        fi
        run_arm "LONGFORM" "WINDOW" "${arm}" "${src}" "long replay from ${src}: $(arm_desc "${src}")" "LONG" "long" "${ctrl}"
        idx=$((idx + 1))
    done
fi

# ----------------------------------------------------------------
# 3) Quant stage on best LONGFORM checkpoint.
# ----------------------------------------------------------------
if [[ "${RUN_LONGFORM}" == "1" && "${RUN_QUANT}" == "1" ]]; then
    if [[ -z "${BEST_LONG_CKPT}" || ! -f "${BEST_LONG_CKPT}" ]]; then
        echo "ERROR: best LONGFORM checkpoint missing: ${BEST_LONG_CKPT:-(unset)}" >&2
        exit 1
    fi

    run_arm "LONGFORM" "POST_WINDOW" "BW17Q-00" "${BEST_LONG_SOURCE}" \
        "naive int6 on frozen best LONGFORM checkpoint" "QUANT" "quant" 1 \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${BEST_LONG_CKPT}" \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0

    run_arm "LONGFORM" "POST_WINDOW" "BW17Q-01" "${BEST_LONG_SOURCE}" \
        "standard GPTQ (128x2048) on frozen best LONGFORM checkpoint" "QUANT" "quant" 0 \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${BEST_LONG_CKPT}" \
        SKIP_GPTQ=0 \
        LOOP_AWARE_GPTQ=0 \
        GPTQ_CAL_SAMPLES=128 \
        GPTQ_CAL_SEQ_LEN=2048

    run_arm "LONGFORM" "POST_WINDOW" "BW17Q-01L" "${BEST_LONG_SOURCE}" \
        "GPTQ-lite (64x1024) on frozen best LONGFORM checkpoint" "QUANT" "quant" 0 \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${BEST_LONG_CKPT}" \
        SKIP_GPTQ=0 \
        LOOP_AWARE_GPTQ=0 \
        GPTQ_CAL_SAMPLES=64 \
        GPTQ_CAL_SEQ_LEN=1024

    if [[ "${RUN_LOOP_AWARE_GPTQ}" == "1" ]]; then
        run_arm "LONGFORM" "POST_WINDOW" "BW17Q-02" "${BEST_LONG_SOURCE}" \
            "loop-aware GPTQ on frozen best LONGFORM checkpoint" "QUANT" "quant" 0 \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${BEST_LONG_CKPT}" \
            SKIP_GPTQ=0 \
            LOOP_AWARE_GPTQ=1 \
            GPTQ_CAL_SAMPLES=128 \
            GPTQ_CAL_SEQ_LEN=2048
    fi
fi

cat <<TXT

================================================================
BW17 DGX-Spark cadence interaction sequence complete.
summary: ${SUMMARY}

Stage policy:
  - RAPID: small-token local screening around 9F stable stack
  - LONGFORM: replay control + top-${LONGFORM_TOPK} rapid candidates
  - POST_WINDOW: quant-only bake-off on best LONGFORM checkpoint

Best LONGFORM:
  arm=${BEST_LONG_ARM:-(not run)}
  source_arm=${BEST_LONG_SOURCE:-(not run)}
  int6_sw_bpb=${BEST_LONG_INT6:-(not run)}
  ckpt=${BEST_LONG_CKPT:-(not run)}
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
