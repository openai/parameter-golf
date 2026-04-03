#!/bin/bash
set -euo pipefail
# ================================================================
#  BW15_AllCrawler_2k — Unified Ablation Sequence (4x-first)
#
#  One entry point for crawler ablations with ordering:
#    1) Big swings first (BW14-style architecture phase shifts)
#    2) Small interaction + quant sweeps last (BW13/BW12 families)
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW15_AllCrawler_2k/run_ablation_sequence.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${REPO_ROOT}/crawler/2026-04-01_BW12_Interaction_2k/train_gpt_ablate.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

BASE_COMMON=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
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
    NUM_FLAT_LAYERS=5
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
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
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    GPTQ_CAL_SAMPLES=128
    GPTQ_CAL_SEQ_LEN=2048
    NPROC_PER_NODE="${NPROC}"
)

# Tap-off baseline (used by BW14 + BW13 families)
BASE_TAPOFF=(
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
)

# Nightcrawler 5F + TAP shared baseline (used by BW12 family)
BASE_TAPSHARED=(
    CRAWLER_TAP_DIM=32
    CRAWLER_TAP_LOOP_SPECIFIC=0
)

declare -A CONTROL_INT6=()
declare -A CONTROL_CKPT=()
declare -A ARM_PARAMS=()
declare -A ARM_RAW=()
declare -A ARM_INT6=()
declare -A ARM_STEP_MS=()
declare -A ARM_BYTES=()
declare -A ARM_GPTQ_LAYERS=()
declare -A ARM_GPTQ_CAL=()
declare -A ARM_LOG=()

{
    echo -e "phase\tlane\tarm\tdesc\tctrl_group\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tgptq_layers\tgptq_cal_sec\tdelta_vs_control\tlog"
} > "${SUMMARY}"

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

calc_delta() {
    local control="$1"
    local value="$2"
    if [[ -z "${control}" || -z "${value}" || "${control}" == "?" || "${value}" == "?" ]]; then
        echo "?"
        return
    fi
    python3 - <<PY
c = float("${control}")
v = float("${value}")
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

run_arm() {
    local phase="$1"; shift
    local lane="$1"; shift
    local arm="$1"; shift
    local desc="$1"; shift
    local base_kind="$1"; shift
    local ctrl_group="$1"; shift
    local is_control="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"
    local run_env=("${BASE_COMMON[@]}")

    case "${base_kind}" in
        tapoff)
            run_env+=("${BASE_TAPOFF[@]}")
            ;;
        tapshared)
            run_env+=("${BASE_TAPSHARED[@]}")
            ;;
        *)
            echo "ERROR: unknown base_kind=${base_kind}" >&2
            exit 1
            ;;
    esac

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${phase} ${lane} ${arm}: ${desc}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    env "${run_env[@]}" "${extra_env[@]}" \
      "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local params raw int6 step_ms bytes gptq_layers gptq_cal_sec delta
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

    ARM_PARAMS["${arm}"]="${params}"
    ARM_RAW["${arm}"]="${raw}"
    ARM_INT6["${arm}"]="${int6}"
    ARM_STEP_MS["${arm}"]="${step_ms}"
    ARM_BYTES["${arm}"]="${bytes}"
    ARM_GPTQ_LAYERS["${arm}"]="${gptq_layers}"
    ARM_GPTQ_CAL["${arm}"]="${gptq_cal_sec}"
    ARM_LOG["${arm}"]="${log}"

    if [[ "${is_control}" == "1" ]]; then
        CONTROL_INT6["${ctrl_group}"]="${int6}"
        CONTROL_CKPT["${ctrl_group}"]="${RESULTS_DIR}/${ctrl_group}_control_s${SEED}_${TS}.final_model.pt"
        if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
            cp -f "${REPO_ROOT}/final_model.pt" "${CONTROL_CKPT[${ctrl_group}]}"
            echo "  control checkpoint (${ctrl_group}): ${CONTROL_CKPT[${ctrl_group}]}"
        else
            echo "  WARNING: final_model.pt missing after control arm ${arm}"
        fi
    fi

    delta=$(calc_delta "${CONTROL_INT6[${ctrl_group}]:-}" "${int6}")
    echo -e "${phase}\t${lane}\t${arm}\t${desc}\t${ctrl_group}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${log}" >> "${SUMMARY}"

    echo "  ${arm}: raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} gptq_layers=${gptq_layers} gptq_cal_sec=${gptq_cal_sec} delta_vs_ctrl=${delta}"
}

run_post_window_arm() {
    local phase="$1"; shift
    local arm="$1"; shift
    local desc="$1"; shift
    local base_kind="$1"; shift
    local ctrl_group="$1"; shift

    if [[ -z "${CONTROL_CKPT[${ctrl_group}]:-}" || ! -f "${CONTROL_CKPT[${ctrl_group}]}" ]]; then
        echo "ERROR: missing control checkpoint for ${ctrl_group}: ${CONTROL_CKPT[${ctrl_group}]:-(unset)}" >&2
        exit 1
    fi

    run_arm "${phase}" "POST_WINDOW" "${arm}" "${desc}" "${base_kind}" "${ctrl_group}" 0 \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${CONTROL_CKPT[${ctrl_group}]}" \
        "$@"
}

add_alias_arm() {
    local phase="$1"; shift
    local lane="$1"; shift
    local arm="$1"; shift
    local desc="$1"; shift
    local ctrl_group="$1"; shift
    local source_arm="$1"; shift

    local params raw int6 step_ms bytes gptq_layers gptq_cal_sec log delta
    params="${ARM_PARAMS[${source_arm}]:-?}"
    raw="${ARM_RAW[${source_arm}]:-?}"
    int6="${ARM_INT6[${source_arm}]:-?}"
    step_ms="${ARM_STEP_MS[${source_arm}]:-?}"
    bytes="${ARM_BYTES[${source_arm}]:-?}"
    gptq_layers="${ARM_GPTQ_LAYERS[${source_arm}]:-0}"
    gptq_cal_sec="${ARM_GPTQ_CAL[${source_arm}]:--}"
    log="${ARM_LOG[${source_arm}]:-(reused)}"

    delta=$(calc_delta "${CONTROL_INT6[${ctrl_group}]:-}" "${int6}")
    echo -e "${phase}\t${lane}\t${arm}\t${desc}\t${ctrl_group}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${log}" >> "${SUMMARY}"
    echo "  ${arm}: reused ${source_arm} metrics (delta_vs_ctrl=${delta})"
}

# ----------------------------------------------------------------
# 1) BIG SWINGS FIRST (tap-off family)
# ----------------------------------------------------------------
run_arm "BIG_SWING" "WINDOW" "BW14BS-00" "control (tap-off Nightcrawler, naive int6)" "tapoff" "tapoff" 1
run_arm "BIG_SWING" "WINDOW" "BW14BS-01" "depth phase shift: NUM_FLAT_LAYERS=6" "tapoff" "tapoff" 0 \
    NUM_FLAT_LAYERS=6
run_arm "BIG_SWING" "WINDOW" "BW14BS-02" "crawler choke flat-128" "tapoff" "tapoff" 0 \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=flat
run_arm "BIG_SWING" "WINDOW" "BW14BS-03" "crawler choke flat-512" "tapoff" "tapoff" 0 \
    CRAWLER_MLP_CHOKE_DIM=512 \
    CRAWLER_MLP_CHOKE_SHAPE=flat
run_arm "BIG_SWING" "WINDOW" "BW14BS-04" "crawler choke residual-128" "tapoff" "tapoff" 0 \
    CRAWLER_MLP_CHOKE_DIM=128 \
    CRAWLER_MLP_CHOKE_SHAPE=residual

# ----------------------------------------------------------------
# 2) SMALL WINDOW + POST_WINDOW (tap-off family; BW13 set)
#    Reuse BW14BS-00 as BW13INT-00 control to avoid duplicate retrain.
# ----------------------------------------------------------------
add_alias_arm "SMALL" "WINDOW" "BW13INT-00" "control (tap-off Nightcrawler, naive int6) [reused BW14BS-00]" "tapoff" "BW14BS-00"

run_arm "SMALL" "WINDOW" "BW13INT-01" "tap-off + anchor dim=32" "tapoff" "tapoff" 0 \
    ANCHOR_DIM=32
run_arm "SMALL" "WINDOW" "BW13INT-02" "tap-off + anchor dim=64" "tapoff" "tapoff" 0 \
    ANCHOR_DIM=64

run_post_window_arm "SMALL" "BW13INT-Q0" "naive int6 on frozen tap-off control" "tapoff" "tapoff" \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0
run_post_window_arm "SMALL" "BW13INT-Q1" "standard GPTQ (128x2048) on frozen tap-off control" "tapoff" "tapoff" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0 \
    GPTQ_CAL_SAMPLES=128 \
    GPTQ_CAL_SEQ_LEN=2048
run_post_window_arm "SMALL" "BW13INT-Q1L" "standard GPTQ-lite (64x1024) on frozen tap-off control" "tapoff" "tapoff" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0 \
    GPTQ_CAL_SAMPLES=64 \
    GPTQ_CAL_SEQ_LEN=1024

# ----------------------------------------------------------------
# 3) SMALL WINDOW + POST_WINDOW (tap-shared family; BW12 set)
# ----------------------------------------------------------------
run_arm "SMALL" "WINDOW" "BW12INT-00" "control (Nightcrawler 5F + TAP shared, naive int6)" "tapshared" "tapshared" 1
run_arm "SMALL" "WINDOW" "BW12INT-01" "tap off (isolate 5F depth without tap)" "tapshared" "tapshared" 0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1
run_arm "SMALL" "WINDOW" "BW12INT-02" "anchor dim=32 on Nightcrawler stack" "tapshared" "tapshared" 0 \
    ANCHOR_DIM=32

run_post_window_arm "SMALL" "BW12INT-Q0" "naive int6 on frozen tap-shared control" "tapshared" "tapshared" \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0
run_post_window_arm "SMALL" "BW12INT-Q1" "standard GPTQ on frozen tap-shared control" "tapshared" "tapshared" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0
run_post_window_arm "SMALL" "BW12INT-Q2" "loop-aware GPTQ on frozen tap-shared control" "tapshared" "tapshared" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=1

cat <<TXT

================================================================
BW15 unified crawler sequence complete.
summary: ${SUMMARY}

Ordering policy:
  1) BIG_SWING phase first (high-amplitude architecture levers)
  2) SMALL phase second (interaction + quant polish)

Promotion guidance:
  - BIG_SWING promote: delta_vs_control <= -0.0060
  - SMALL promote: delta_vs_control <= -0.0008

Classification:
  - WINDOW lanes: retrain-required changes
  - POST_WINDOW lanes: sequential quant-only checks after one control window
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
