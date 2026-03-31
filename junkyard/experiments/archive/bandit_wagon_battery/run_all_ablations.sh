#!/bin/bash
set -euo pipefail
# ================================================================
#  MEGA ABLATION — bandit_wagon series, single GPU, all arms
#
#  Runs all 4 experiment series back-to-back using the unified
#  bandit_wagon_battery train_gpt.py (which supports all features).
#
#  Total: 20 arms @ ~13 min/arm ≈ 4-5 hours on 1×H100
#
#  Series:
#    CTRL  — 1 shared control arm
#    BWC   — choke sweep (4 arms)
#    BWS   — smear (1 arm)
#    BWT   — encoder tap (6 arms)
#    BWB   — battery / rope scale (7 arms)
#
#  Usage:
#    bash experiments/bandit_wagon_battery/run_all_ablations.sh
#    ABLATION_STEPS=200 bash experiments/bandit_wagon_battery/run_all_ablations.sh  # quick smoke test
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-500}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] patching torch inductor AttrsDescriptor bug (if present)..."
python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('torch._inductor.runtime.hints')
if spec and spec.origin:
    p = pathlib.Path(spec.origin)
    txt = p.read_text()
    old = 'attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}'
    if old in txt:
        import attr
        new = 'import attr as _attr; attr_desc_fields = {f.name for f in _attr.fields(AttrsDescriptor)}'
        p.write_text(txt.replace(old, new))
        print('  patched OK')
    else:
        print('  no patch needed')
" 2>/dev/null || echo "  WARNING: could not patch hints.py"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

RESULTS=()

run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2
    # remaining args are env var pairs: KEY=VALUE ...

    echo ""
    echo "================================================================"
    echo "  ${arm_id} — ${label}"
    echo "  [${ABLATION_STEPS} steps | seed=${SEED} | nproc=${NPROC}]"
    echo "================================================================"

    local logfile="${LOGDIR}/mega_${arm_id}_s${SEED}_$(date +%H%M%S).log"

    env \
        MAX_WALLCLOCK_SECONDS=0 \
        ITERATIONS="${ABLATION_STEPS}" \
        WARMDOWN_ITERS=0 \
        SEED="${SEED}" \
        NPROC_PER_NODE="${NPROC}" \
        MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_LEAKY_SLOPE=0.5 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=0 \
        MODEL_DIM=512 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=4 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS=3 \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=1 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        NITRUST_ENABLE=0 \
        NITRUST_STRICT=0 \
        CRAWLER_MLP_CHOKE_DIM=0 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        CRAWLER_LOOP_ROPE_SCALES=1,1,1 \
        "$@" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${logfile}"

    local bpb raw_bpb step_avg quant_gap
    bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null | tail -1 || echo "?")
    raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null | tail -1 || echo "?")
    step_avg=$(grep -oP 'step:[0-9]+/[0-9]+.*?step_avg:\K[0-9.]+' "${logfile}" 2>/dev/null | tail -1 || echo "?")
    if [[ "${raw_bpb}" != "?" && "${bpb}" != "?" ]]; then
        quant_gap=$(python3 -c "print(f'{float(\"${bpb}\")-float(\"${raw_bpb}\"):.4f}')" 2>/dev/null || echo "?")
    else
        quant_gap="?"
    fi

    RESULTS+=("${arm_id}|${label}|${step_avg}ms|${raw_bpb}|${bpb}|${quant_gap}")
    echo "  -> step_avg:${step_avg}ms  raw_bpb:${raw_bpb}  int6_sw_bpb:${bpb}  quant_gap:${quant_gap}"
}

# ----------------------------------------------------------------
# CTRL — shared control (all features disabled)
# ----------------------------------------------------------------
run_arm CTRL-00 "control (all disabled)"

# ----------------------------------------------------------------
# BWC — choke sweep (CRAWLER_MLP_CHOKE_DIM)
# ----------------------------------------------------------------
run_arm BWC-01 "choke=32 (extreme)"        CRAWLER_MLP_CHOKE_DIM=32
run_arm BWC-02 "choke=128 (moderate)"      CRAWLER_MLP_CHOKE_DIM=128
run_arm BWC-03 "choke=256 (conservative)"  CRAWLER_MLP_CHOKE_DIM=256
run_arm BWC-04 "choke=512 (minimal)"       CRAWLER_MLP_CHOKE_DIM=512

# ----------------------------------------------------------------
# BWS — loop smeargate
# ----------------------------------------------------------------
run_arm BWS-01 "loop smear=1"              CRAWLER_LOOP_SMEAR=1

# ----------------------------------------------------------------
# BWT — encoder tap sweep
# ----------------------------------------------------------------
run_arm BWT-01 "tap dim=32 shared all"        CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=0 CRAWLER_TAP_LAYERS=all
run_arm BWT-02 "tap dim=32 per-loop all"      CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=1 CRAWLER_TAP_LAYERS=all
run_arm BWT-03 "tap dim=16 per-loop all"      CRAWLER_TAP_DIM=16 CRAWLER_TAP_LOOP_SPECIFIC=1 CRAWLER_TAP_LAYERS=all
run_arm BWT-04 "tap dim=64 per-loop all"      CRAWLER_TAP_DIM=64 CRAWLER_TAP_LOOP_SPECIFIC=1 CRAWLER_TAP_LAYERS=all
run_arm BWT-05 "tap dim=32 per-loop deep"     CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=1 CRAWLER_TAP_LAYERS=deep
run_arm BWT-06 "tap dim=32 per-loop shallow"  CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=1 CRAWLER_TAP_LAYERS=shallow

# ----------------------------------------------------------------
# BWB — battery / per-loop RoPE scale sweep
# ----------------------------------------------------------------
run_arm BWB-01 "battery 1,2,4 (gentle asc)"     CRAWLER_LOOP_ROPE_SCALES=1,2,4
run_arm BWB-02 "battery 1,3,9 (moderate asc)"   CRAWLER_LOOP_ROPE_SCALES=1,3,9
run_arm BWB-03 "battery 1,5,25 (aggressive)"    CRAWLER_LOOP_ROPE_SCALES=1,5,25
run_arm BWB-04 "battery 9,3,1 (descending)"     CRAWLER_LOOP_ROPE_SCALES=9,3,1
run_arm BWB-05 "battery 1,9,1 (middle wide)"    CRAWLER_LOOP_ROPE_SCALES=1,9,1
run_arm BWB-06 "battery 1,1,9 (final wide)"     CRAWLER_LOOP_ROPE_SCALES=1,1,9
run_arm BWB-07 "battery 9,1,1 (first wide)"     CRAWLER_LOOP_ROPE_SCALES=9,1,1

# ================================================================
#  SUMMARY
# ================================================================
CTRL_BPB=$(echo "${RESULTS[0]}" | cut -d'|' -f5)

echo ""
echo "================================================================"
echo "  MEGA ABLATION SUMMARY"
echo "  seed=${SEED}  steps=${ABLATION_STEPS}  Reference: BW2-00=1.52365"
echo "================================================================"
printf "%-10s %-35s %-10s %-12s %-12s %-10s %s\n" \
    "ARM" "LABEL" "STEP_AVG" "RAW_BPB" "INT6_SW_BPB" "QUANT_GAP" "DELTA"
printf "%-10s %-35s %-10s %-12s %-12s %-10s %s\n" \
    "---" "-----" "--------" "-------" "-----------" "---------" "-----"

for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg raw bpb quant_gap <<< "${r}"
    delta="—"
    if [[ "${bpb}" != "?" && "${CTRL_BPB}" != "?" ]]; then
        delta=$(python3 -c "
v=float('${bpb}')-float('${CTRL_BPB}')
sign='+' if v>=0 else ''
print(f'{sign}{v:.5f}')
" 2>/dev/null || echo "?")
    fi
    printf "%-10s %-35s %-10s %-12s %-12s %-10s %s\n" \
        "${arm}" "${label}" "${step_avg}" "${raw}" "${bpb}" "${quant_gap}" "${delta}"
done

echo ""
echo "  Control: CTRL-00 int6_sw_bpb = ${CTRL_BPB}"
echo "  Reference: BW2-00 (prior session) = 1.52365"
echo "  Threshold: beat control by ≥0.005 to qualify for promotion"
echo ""

# Find winner
echo "  Winners (beat control by ≥0.005):"
found_winner=0
for r in "${RESULTS[@]}"; do
    IFS='|' read -r arm label step_avg raw bpb quant_gap <<< "${r}"
    if [[ "${arm}" == "CTRL-00" ]]; then continue; fi
    if [[ "${bpb}" != "?" && "${CTRL_BPB}" != "?" ]]; then
        is_winner=$(python3 -c "
ctrl=float('${CTRL_BPB}')
bpb=float('${bpb}')
print('yes' if (ctrl - bpb) >= 0.005 else 'no')
" 2>/dev/null || echo "no")
        if [[ "${is_winner}" == "yes" ]]; then
            echo "    *** ${arm} ${label} → ${bpb} (delta=$(python3 -c "print(f'{float(\"${bpb}\")-float(\"${CTRL_BPB}\"):.5f}')" 2>/dev/null))"
            found_winner=1
        fi
    fi
done
if [[ ${found_winner} -eq 0 ]]; then
    echo "    (none cleared the 0.005 threshold)"
fi

echo "================================================================"
echo "  DONE. All logs in ${LOGDIR}/mega_*"
echo "================================================================"
