#!/usr/bin/env bash
set -euo pipefail

# Ultra-aggressive remote runner for Parameter Golf on CUDA hosts (RunPod/H100 recommended).
# Usage:
#   bash run_remote_fast.sh
# Optional env:
#   REPO_DIR=/workspace/parameter-golf
#   NPROC=8
#   TRAIN_SHARDS=10
#   SWEEP_PARALLEL=1
#   NPROC_PER_CANDIDATE=2

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
NPROC="${NPROC:-8}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
SWEEP_PARALLEL="${SWEEP_PARALLEL:-1}"
NPROC_PER_CANDIDATE="${NPROC_PER_CANDIDATE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_DIR}"

${PYTHON_BIN} -m pip install -U pip
${PYTHON_BIN} -m pip install -r requirements.txt

${PYTHON_BIN} data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"

SEEDS=(42 1337 2024)

# Candidate 1: current #1 leaderboard script.
CAND1_NAME="sota_int5_10l"
CAND1_SCRIPT="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
CAND1_ENV=""

# Candidate 2: current #2 leaderboard script.
CAND2_NAME="sota_int6_9l"
CAND2_SCRIPT="records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py"
CAND2_ENV=""

# Candidate 3: tuned variant of top-1 to hunt for extra gain.
CAND3_NAME="sota_int5_10l_tuned"
CAND3_SCRIPT="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
CAND3_ENV="MATRIX_LR=0.0195 SCALAR_LR=0.0195 TIED_EMBED_LR=0.031 SWA_START_FRAC=0.35 BIGRAM_VOCAB_SIZE=12288"

RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="runs/remote_fast_${RUNSTAMP}"
mkdir -p "${OUTDIR}"

cat > "${OUTDIR}/candidate_manifest.json" << EOF
{
    "${CAND1_NAME}": {"script": "${CAND1_SCRIPT}", "env": "${CAND1_ENV}"},
    "${CAND2_NAME}": {"script": "${CAND2_SCRIPT}", "env": "${CAND2_ENV}"},
    "${CAND3_NAME}": {"script": "${CAND3_SCRIPT}", "env": "${CAND3_ENV}"}
}
EOF

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
    echo "No CUDA GPUs detected by nvidia-smi."
    exit 1
fi

echo "Detected GPUs: ${GPU_COUNT}"
echo "Output directory: ${OUTDIR}"

run_one_candidate() {
    local cname="$1"
    local cscript="$2"
    local cenv="$3"
    local nproc_local="$4"
    local visible_gpus="$5"

    local cand_dir="${OUTDIR}/${cname}"
    mkdir -p "${cand_dir}"

    for seed in "${SEEDS[@]}"; do
        local logfile="${cand_dir}/seed${seed}.log"
        local runid="remote_${cname}_seed${seed}_${RUNSTAMP}"
        echo "[run] ${cname} seed=${seed} nproc=${nproc_local} gpus=${visible_gpus}"

        if [[ -n "${visible_gpus}" ]]; then
            CUDA_VISIBLE_DEVICES="${visible_gpus}" \
            SEED="${seed}" \
            RUN_ID="${runid}" \
            DATA_PATH="./data/datasets/fineweb10B_sp1024" \
            TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
            bash -lc "${cenv} torchrun --standalone --nproc_per_node=${nproc_local} ${cscript}" \
            2>&1 | tee "${logfile}"
        else
            SEED="${seed}" \
            RUN_ID="${runid}" \
            DATA_PATH="./data/datasets/fineweb10B_sp1024" \
            TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
            bash -lc "${cenv} torchrun --standalone --nproc_per_node=${nproc_local} ${cscript}" \
            2>&1 | tee "${logfile}"
        fi
    done
}

if [[ "${SWEEP_PARALLEL}" == "1" && "${GPU_COUNT}" -ge 3 ]]; then
    GPC="${NPROC_PER_CANDIDATE}"
    if [[ -z "${GPC}" ]]; then
        GPC=$((GPU_COUNT / 3))
        if [[ "${GPC}" -lt 1 ]]; then
            GPC=1
        fi
    fi

    if [[ $((GPC * 3)) -gt "${GPU_COUNT}" ]]; then
        echo "NPROC_PER_CANDIDATE=${GPC} is too large for ${GPU_COUNT} GPUs and 3 candidates."
        exit 1
    fi

    make_gpu_list() {
        local start="$1"
        local count="$2"
        local out=""
        local i
        for ((i = 0; i < count; i++)); do
            local idx=$((start + i))
            if [[ -z "${out}" ]]; then
                out="${idx}"
            else
                out="${out},${idx}"
            fi
        done
        echo "${out}"
    }

    GPUSET1="$(make_gpu_list 0 "${GPC}")"
    GPUSET2="$(make_gpu_list "${GPC}" "${GPC}")"
    GPUSET3="$(make_gpu_list "$((2 * GPC))" "${GPC}")"

    run_one_candidate "${CAND1_NAME}" "${CAND1_SCRIPT}" "${CAND1_ENV}" "${GPC}" "${GPUSET1}" &
    pid1=$!
    run_one_candidate "${CAND2_NAME}" "${CAND2_SCRIPT}" "${CAND2_ENV}" "${GPC}" "${GPUSET2}" &
    pid2=$!
    run_one_candidate "${CAND3_NAME}" "${CAND3_SCRIPT}" "${CAND3_ENV}" "${GPC}" "${GPUSET3}" &
    pid3=$!

    wait "${pid1}"
    wait "${pid2}"
    wait "${pid3}"
else
    echo "Parallel sweep disabled or insufficient GPUs; running sequentially with NPROC=${NPROC}."
    run_one_candidate "${CAND1_NAME}" "${CAND1_SCRIPT}" "${CAND1_ENV}" "${NPROC}" ""
    run_one_candidate "${CAND2_NAME}" "${CAND2_SCRIPT}" "${CAND2_ENV}" "${NPROC}" ""
    run_one_candidate "${CAND3_NAME}" "${CAND3_SCRIPT}" "${CAND3_ENV}" "${NPROC}" ""
fi

${PYTHON_BIN} - << 'PY'
import json
import pathlib
import re
import statistics

root = pathlib.Path("runs")
latest = max((p for p in root.glob("remote_fast_*")), key=lambda p: p.stat().st_mtime)
rows = []
for p in sorted(latest.glob("*/*.log")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = re.findall(r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)", txt)
        s = re.findall(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes", txt)
        if m:
                vl, vb = m[-1]
                rows.append(
                        {
                                "candidate": p.parent.name,
                                "log": str(p),
                                "val_loss": float(vl),
                                "val_bpb": float(vb),
                                "total_bytes": int(s[-1]) if s else None,
                                "valid_16mb": (int(s[-1]) <= 16_000_000) if s else None,
                        }
                )

rows.sort(key=lambda x: x["val_bpb"])
by_candidate = {}
for r in rows:
        by_candidate.setdefault(r["candidate"], []).append(r)

candidate_summary = {}
for c, cruns in by_candidate.items():
        vals = [x["val_bpb"] for x in cruns]
        candidate_summary[c] = {
                "n": len(vals),
                "mean_val_bpb": statistics.mean(vals),
                "std_val_bpb": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "best_val_bpb": min(vals),
                "best_log": min(cruns, key=lambda x: x["val_bpb"])["log"],
        }

summary = {
    "runs": rows,
    "mean_val_bpb": statistics.mean([r["val_bpb"] for r in rows]) if rows else None,
    "std_val_bpb": statistics.stdev([r["val_bpb"] for r in rows]) if len(rows) > 1 else 0.0,
        "best_run": rows[0] if rows else None,
        "candidate_summary": candidate_summary,
}
out = latest / "summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Summary written to {out}")
PY
