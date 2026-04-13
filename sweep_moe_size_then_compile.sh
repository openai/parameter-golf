#!/usr/bin/env bash
set -euo pipefail

# Two-stage MoE sweep:
# 1) Size gate (no torch.compile) to find near-16MB configs.
# 2) Compile check (max-autotune) ONLY for size-qualified configs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/moe_size_then_compile_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

TARGET_TOTAL_BYTES="${TARGET_TOTAL_BYTES:-15990000}"
MAX_TOTAL_BYTES="${MAX_TOTAL_BYTES:-16000000}"
TOPK_PER_EXPERT="${TOPK_PER_EXPERT:-3}"

TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_8192_bpe.model}"
DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp8192}"

BASE_ENV=(
  "TOKENIZER_PATH=${TOKENIZER_PATH}"
  "DATA_PATH=${DATA_PATH}"
  "VOCAB_SIZE=8192"
  "NUM_HEADS=8"
  "NUM_KV_HEADS=8"
  "TRAIN_SEQ_LEN=1024"
  "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-8192}"
  "GRAD_ACCUM_STEPS=1"
  "FAST_SMOKE=1"
  "FAST_SMOKE_BATCHES=32"
  "ITERATIONS=0"
  "MAX_WALLCLOCK_SECONDS=20"
  "VAL_BATCH_SIZE=4096"
  "VAL_MAX_TOKENS=8192"
  "CURRICULUM_ENABLED=0"
  "SEQ_LEN_START=0"
  "BATCH_TOKENS_START=0"
  "DDP_FIND_UNUSED_PARAMETERS=1"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

# label experts layers dim mlp_mult moe_frac shared_blocks
CANDIDATES=(
  "e2_hi1 2 16 600 5 0.95 0"
  "e2_hi2 2 18 560 5 0.95 0"
  "e3_hi1 3 16 560 5 0.95 0"
  "e3_hi2 3 18 520 5 0.95 0"
  "e4_hi1 4 16 540 5 0.95 0"
  "e4_hi2 4 18 500 5 0.95 0"
  "e6_hi1 6 14 560 4 0.95 0"
  "e6_hi2 6 16 520 4 0.95 0"
  "e8_hi1 8 14 520 4 0.95 0"
  "e8_hi2 8 16 480 4 0.95 0"
)

SIZE_TSV="${OUT_DIR}/size_pass.tsv"
COMPILE_TSV="${OUT_DIR}/compile_pass.tsv"
SHORTLIST_TSV="${OUT_DIR}/shortlist.tsv"

echo -e "label\texperts\tlayers\tdim\tmlp_mult\tmoe_frac\tshared_blocks\tartifact_mb\ttotal_mb\ttotal_bytes\tbudget\trun_status" > "${SIZE_TSV}"
echo -e "label\texperts\tlayers\tdim\tmlp_mult\tmoe_frac\tshared_blocks\tcompile_s\tcompile_ok\trun_status" > "${COMPILE_TSV}"

run_size_pass() {
  local label="$1" experts="$2" layers="$3" dim="$4" mlp="$5" frac="$6" shared="$7"
  local logf="${OUT_DIR}/${label}.size.log"
  local rc=0

  env "${BASE_ENV[@]}" \
    COMPILE_MODE=none \
    COMPILER_WARMUP_STEPS=0 \
    PRECOMPILE_ONLY=0 \
    SYNTHETIC_WARMUP=0 \
    MOE_ENABLED=1 \
    MOE_NUM_EXPERTS="${experts}" \
    MOE_LAYER_FRAC="${frac}" \
    SHARED_BLOCKS="${shared}" \
    NUM_LAYERS="${layers}" \
    MODEL_DIM="${dim}" \
    MLP_MULT="${mlp}" \
    python3 build_submission.py >/dev/null && python3 train_gpt.py >"${logf}" 2>&1 || rc=$?

  local artifact_mb total_bytes total_mb budget
  artifact_mb="$(grep -Eo 'artifact:[0-9.]+MB' "${logf}" | tail -n1 | sed -E 's/artifact:([0-9.]+)MB/\1/' || true)"
  total_bytes="$(grep -Eo 'budget:[0-9]+/16000000' "${logf}" | tail -n1 | sed -E 's/budget:([0-9]+)\/16000000/\1/' || true)"
  budget="$(grep -Eo '(FITS|OVER)$' "${logf}" | tail -n1 || true)"

  if [[ -z "${artifact_mb}" || -z "${total_bytes}" || -z "${budget}" ]]; then
    echo -e "${label}\t${experts}\t${layers}\t${dim}\t${mlp}\t${frac}\t${shared}\t-\t-\t-\t-\trun_fail" >> "${SIZE_TSV}"
    return 0
  fi

  total_mb="$(awk -v b="${total_bytes}" 'BEGIN{printf "%.2f", b/1000000.0}')"
  local run_status="ok"
  [[ "${rc}" -ne 0 ]] && run_status="run_fail"

  echo -e "${label}\t${experts}\t${layers}\t${dim}\t${mlp}\t${frac}\t${shared}\t${artifact_mb}\t${total_mb}\t${total_bytes}\t${budget}\t${run_status}" >> "${SIZE_TSV}"
}

run_compile_pass() {
  local label="$1" experts="$2" layers="$3" dim="$4" mlp="$5" frac="$6" shared="$7"
  local logf="${OUT_DIR}/${label}.compile.log"
  local t0 t1 rc=0
  t0="$(date +%s)"
  env "${BASE_ENV[@]}" \
    COMPILE_MODE=max-autotune \
    COMPILER_WARMUP_STEPS=1 \
    PRECOMPILE_ONLY=1 \
    SYNTHETIC_WARMUP=1 \
    MOE_ENABLED=1 \
    MOE_NUM_EXPERTS="${experts}" \
    MOE_LAYER_FRAC="${frac}" \
    SHARED_BLOCKS="${shared}" \
    NUM_LAYERS="${layers}" \
    MODEL_DIM="${dim}" \
    MLP_MULT="${mlp}" \
    python3 build_submission.py >/dev/null && python3 train_gpt.py >"${logf}" 2>&1 || rc=$?
  t1="$(date +%s)"
  local dt=$((t1 - t0))

  local compile_ok="no"
  if grep -q 'precompile_only:done' "${logf}"; then
    if [[ "${dt}" -lt 60 ]]; then
      compile_ok="yes"
    fi
  fi
  local run_status="ok"
  [[ "${rc}" -ne 0 ]] && run_status="run_fail"
  echo -e "${label}\t${experts}\t${layers}\t${dim}\t${mlp}\t${frac}\t${shared}\t${dt}\t${compile_ok}\t${run_status}" >> "${COMPILE_TSV}"
}

echo "[1/3] Size pass..."
for c in "${CANDIDATES[@]}"; do
  run_size_pass ${c}
done

echo "[2/3] Build shortlist (size-qualified only)..."
python3 - "${SIZE_TSV}" "${SHORTLIST_TSV}" "${TARGET_TOTAL_BYTES}" "${MAX_TOTAL_BYTES}" "${TOPK_PER_EXPERT}" <<'PY'
import csv, sys
size_tsv, out_tsv, target_s, max_s, topk_s = sys.argv[1:]
target, max_bytes, topk = int(target_s), int(max_s), int(topk_s)
rows = []
with open(size_tsv, newline="") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        if row["budget"] != "FITS":
            continue
        try:
            total_bytes = int(row["total_bytes"])
            experts = int(row["experts"])
        except Exception:
            continue
        if total_bytes > max_bytes:
            continue
        row["distance"] = abs(total_bytes - target)
        row["total_bytes_i"] = total_bytes
        row["experts_i"] = experts
        rows.append(row)

picked = []
# Keep top-k per expert count present in the sweep.
expert_values = sorted({r["experts_i"] for r in rows})
for experts in expert_values:
    cands = [r for r in rows if r["experts_i"] == experts]
    cands.sort(key=lambda x: (x["distance"], -x["total_bytes_i"]))
    picked.extend(cands[:topk])

picked.sort(key=lambda x: (x["distance"], -x["total_bytes_i"]))
fields = ["label","experts","layers","dim","mlp_mult","moe_frac","shared_blocks","artifact_mb","total_mb","total_bytes","budget","run_status","distance"]
with open(out_tsv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    for row in picked:
        w.writerow({k: row.get(k, "") for k in fields})
print(f"shortlisted={len(picked)}")
PY

echo "[3/3] Compile pass for shortlist..."
tail -n +2 "${SHORTLIST_TSV}" | while IFS=$'\t' read -r label experts layers dim mlp frac shared _; do
  run_compile_pass "${label}" "${experts}" "${layers}" "${dim}" "${mlp}" "${frac}" "${shared}"
done

echo
echo "Size pass:    ${SIZE_TSV}"
echo "Shortlist:    ${SHORTLIST_TSV}"
echo "Compile pass: ${COMPILE_TSV}"
echo
echo "Top compile-pass rows:"
python3 - "${SHORTLIST_TSV}" "${COMPILE_TSV}" <<'PY'
import csv, sys
short_tsv, comp_tsv = sys.argv[1:]
short = {}
with open(short_tsv, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        short[r["label"]] = r
rows = []
with open(comp_tsv, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        s = short.get(r["label"])
        if not s:
            continue
        rows.append({
            "label": r["label"],
            "experts": r["experts"],
            "layers": r["layers"],
            "dim": r["dim"],
            "total_mb": s["total_mb"],
            "distance": s["distance"],
            "compile_s": r["compile_s"],
            "compile_ok": r["compile_ok"],
            "run_status": r["run_status"],
        })
rows.sort(key=lambda x: (x["compile_ok"] != "yes", int(x["distance"]), int(x["compile_s"])))
print("label\texperts\tlayers\tdim\ttotal_mb\tdistance\tcompile_s\tcompile_ok\trun_status")
for r in rows[:10]:
    print("{label}\t{experts}\t{layers}\t{dim}\t{total_mb}\t{distance}\t{compile_s}\t{compile_ok}\t{run_status}".format(**r))
PY
