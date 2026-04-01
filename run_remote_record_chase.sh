#!/usr/bin/env bash
set -euo pipefail

# Record-chase pipeline for Parameter Golf.
# Goal: maximize chance of leaderboard #1 through staged search.
#
# Stage A (screen):
# - many variants, 1 seed each, shorter wallclock
# - pick top-K by val_bpb
#
# Stage B (finals):
# - top-K variants, 3 seeds each, full wallclock
# - compute per-variant mean/std and best run
# - optionally scaffold + push submission PR
#
# Required: CUDA box
# Optional env:
#   REPO_DIR=/workspace/parameter-golf
#   PYTHON_BIN=python3
#   NPROC=8
#   TRAIN_SHARDS=10
#   STAGE_A_SECONDS=180
#   STAGE_B_SECONDS=600
#   TOPK=3
#   SWEEP_PARALLEL=1
#   NPROC_PER_JOB=2
#   GITHUB_ID=your_handle
#   AUTHOR_NAME="Your Name"
#   AUTO_SUBMIT=1

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC="${NPROC:-8}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
STAGE_A_SECONDS="${STAGE_A_SECONDS:-180}"
STAGE_B_SECONDS="${STAGE_B_SECONDS:-600}"
TOPK="${TOPK:-3}"
SWEEP_PARALLEL="${SWEEP_PARALLEL:-1}"
NPROC_PER_JOB="${NPROC_PER_JOB:-}"
AUTO_SUBMIT="${AUTO_SUBMIT:-0}"

cd "${REPO_DIR}"

${PYTHON_BIN} -m pip install -U pip
${PYTHON_BIN} -m pip install -r requirements.txt

${PYTHON_BIN} data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
  echo "No CUDA GPUs detected."
  exit 1
fi

RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="runs/record_chase_${RUNSTAMP}"
mkdir -p "${OUTROOT}/stage_a" "${OUTROOT}/stage_b"

# Base scripts from strongest public entries.
SCRIPT_TOP1="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
SCRIPT_TOP2="records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py"

# Variant bank: name|script|extra_env
VARIANTS=(
  "top1_base|${SCRIPT_TOP1}|"
  "top1_bigram12k|${SCRIPT_TOP1}|BIGRAM_VOCAB_SIZE=12288"
  "top1_bigram14k|${SCRIPT_TOP1}|BIGRAM_VOCAB_SIZE=14336"
  "top1_swa35|${SCRIPT_TOP1}|SWA_START_FRAC=0.35"
  "top1_swa45|${SCRIPT_TOP1}|SWA_START_FRAC=0.45"
  "top1_lr195|${SCRIPT_TOP1}|MATRIX_LR=0.0195 SCALAR_LR=0.0195"
  "top1_lr205|${SCRIPT_TOP1}|MATRIX_LR=0.0205 SCALAR_LR=0.0205"
  "top1_tied031|${SCRIPT_TOP1}|TIED_EMBED_LR=0.031"
  "top2_base|${SCRIPT_TOP2}|"
  "top2_wd04|${SCRIPT_TOP2}|WEIGHT_DECAY=0.04"
)

STAGE_A_SEED=42
FINAL_SEEDS=(42 1337 2024)

run_one() {
  local outlog="$1"
  local script="$2"
  local nproc_local="$3"
  local max_seconds="$4"
  local seed="$5"
  local visible_gpus="$6"
  local extra_env="$7"
  local run_id="$8"

  if [[ -n "${visible_gpus}" ]]; then
    CUDA_VISIBLE_DEVICES="${visible_gpus}" \
    SEED="${seed}" \
    RUN_ID="${run_id}" \
    DATA_PATH="./data/datasets/fineweb10B_sp1024" \
    TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
    MAX_WALLCLOCK_SECONDS="${max_seconds}" \
    bash -lc "${extra_env} torchrun --standalone --nproc_per_node=${nproc_local} ${script}" \
    2>&1 | tee "${outlog}"
  else
    SEED="${seed}" \
    RUN_ID="${run_id}" \
    DATA_PATH="./data/datasets/fineweb10B_sp1024" \
    TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
    MAX_WALLCLOCK_SECONDS="${max_seconds}" \
    bash -lc "${extra_env} torchrun --standalone --nproc_per_node=${nproc_local} ${script}" \
    2>&1 | tee "${outlog}"
  fi
}

parse_logs_py='import json, pathlib, re, statistics, sys\n'
parse_logs_py+="root=pathlib.Path(sys.argv[1])\n"
parse_logs_py+="rows=[]\n"
parse_logs_py+="for p in sorted(root.glob('**/*.log')):\n"
parse_logs_py+="  t=p.read_text(encoding='utf-8',errors='ignore')\n"
parse_logs_py+="  m=re.findall(r'final_int8_zlib_roundtrip_exact\\s+val_loss:([0-9.]+)\\s+val_bpb:([0-9.]+)',t)\n"
parse_logs_py+="  s=re.findall(r'Total submission size int8\\+zlib:\\s*([0-9]+)\\s*bytes',t)\n"
parse_logs_py+="  if m:\n"
parse_logs_py+="    vl,vb=m[-1]\n"
parse_logs_py+="    rows.append({'log':str(p),'val_loss':float(vl),'val_bpb':float(vb),'total_bytes':int(s[-1]) if s else None})\n"
parse_logs_py+="rows.sort(key=lambda x:x['val_bpb'])\n"
parse_logs_py+="print(json.dumps(rows,indent=2))\n"

# Stage A
if [[ "${SWEEP_PARALLEL}" == "1" && "${GPU_COUNT}" -ge 2 ]]; then
  J="${NPROC_PER_JOB}"
  if [[ -z "${J}" ]]; then
    J=1
  fi
else
  J="${NPROC}"
fi

echo "[stage_a] screening ${#VARIANTS[@]} variants"

a_idx=0
pids=()
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name script extra <<< "${v}"
  d="${OUTROOT}/stage_a/${name}"
  mkdir -p "${d}"
  log="${d}/seed${STAGE_A_SEED}.log"

  if [[ "${SWEEP_PARALLEL}" == "1" && "${GPU_COUNT}" -ge 2 ]]; then
    g="$((a_idx % GPU_COUNT))"
    run_one "${log}" "${script}" "${J}" "${STAGE_A_SECONDS}" "${STAGE_A_SEED}" "${g}" "${extra}" "stageA_${name}_${RUNSTAMP}" &
    pids+=("$!")
  else
    run_one "${log}" "${script}" "${J}" "${STAGE_A_SECONDS}" "${STAGE_A_SEED}" "" "${extra}" "stageA_${name}_${RUNSTAMP}"
  fi
  a_idx=$((a_idx+1))
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  for p in "${pids[@]}"; do wait "${p}"; done
fi

${PYTHON_BIN} -c "${parse_logs_py}" "${OUTROOT}/stage_a" > "${OUTROOT}/stage_a/ranked.json"

TOP_VARIANTS="$(${PYTHON_BIN} - << 'PY'
import json, pathlib
root = pathlib.Path("${OUTROOT}")
ranked = json.loads((root / "stage_a" / "ranked.json").read_text(encoding="utf-8"))
seen=[]
for r in ranked:
    cand = pathlib.Path(r['log']).parent.name
    if cand not in seen:
        seen.append(cand)
    if len(seen) >= int("${TOPK}"):
        break
print("\n".join(seen))
PY
)"

echo "[stage_b] finalists:"
echo "${TOP_VARIANTS}"

# Stage B
pids=()
b_idx=0
while IFS= read -r finalist; do
  [[ -z "${finalist}" ]] && continue
  # recover variant tuple
  chosen=""
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r name script extra <<< "${v}"
    if [[ "${name}" == "${finalist}" ]]; then
      chosen="${v}"
      break
    fi
  done
  if [[ -z "${chosen}" ]]; then
    continue
  fi

  IFS='|' read -r name script extra <<< "${chosen}"
  mkdir -p "${OUTROOT}/stage_b/${name}"

  for seed in "${FINAL_SEEDS[@]}"; do
    log="${OUTROOT}/stage_b/${name}/seed${seed}.log"
    if [[ "${SWEEP_PARALLEL}" == "1" && "${GPU_COUNT}" -ge 2 ]]; then
      g="$((b_idx % GPU_COUNT))"
      run_one "${log}" "${script}" "1" "${STAGE_B_SECONDS}" "${seed}" "${g}" "${extra}" "stageB_${name}_seed${seed}_${RUNSTAMP}" &
      pids+=("$!")
      b_idx=$((b_idx+1))
    else
      run_one "${log}" "${script}" "${NPROC}" "${STAGE_B_SECONDS}" "${seed}" "" "${extra}" "stageB_${name}_seed${seed}_${RUNSTAMP}"
    fi
  done
done <<< "${TOP_VARIANTS}"

if [[ "${#pids[@]}" -gt 0 ]]; then
  for p in "${pids[@]}"; do wait "${p}"; done
fi

${PYTHON_BIN} - << 'PY'
import json
import pathlib
import re
import statistics

out = pathlib.Path("${OUTROOT}")
rows = []
for p in sorted((out / "stage_b").glob("*/*.log")):
    t = p.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)", t)
    s = re.findall(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes", t)
    if m:
        vl, vb = m[-1]
        rows.append(
            {
                "candidate": p.parent.name,
                "log": str(p),
                "val_loss": float(vl),
                "val_bpb": float(vb),
                "total_bytes": int(s[-1]) if s else None,
            }
        )

rows.sort(key=lambda x: x["val_bpb"])
by = {}
for r in rows:
    by.setdefault(r["candidate"], []).append(r)

cands = {}
for c, rr in by.items():
    vals = [x["val_bpb"] for x in rr]
    cands[c] = {
        "n": len(vals),
        "mean_val_bpb": statistics.mean(vals),
        "std_val_bpb": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "best_val_bpb": min(vals),
        "best_log": min(rr, key=lambda x: x["val_bpb"])["log"],
    }

summary = {
    "runs": rows,
    "best_run": rows[0] if rows else None,
    "candidate_summary": cands,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Summary written to {out / 'summary.json'}")
PY

if [[ "${AUTO_SUBMIT}" == "1" ]]; then
  export RUN_DIR="${OUTROOT}"
  bash submit_remote_fast.sh
fi

echo "Record chase complete: ${OUTROOT}"
