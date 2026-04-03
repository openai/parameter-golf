#!/usr/bin/env bash
set -euo pipefail

# BANDIT_WAGON ad-hoc winddown A/B matrix
# Runs post-train winddown only from a finished checkpoint and ranks final BPB.
#
# Usage:
#   MODEL_PATH=/abs/path/to/final_model.pt \
#   SEED=1337 NPROC_PER_NODE=8 \
#   bash experiments/Bandit_Wagon/winddown_ab.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEEDS="${SEEDS:-${SEED:-1337}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/final_model.pt}"
AUTO_ARCH_FROM_CKPT="${AUTO_ARCH_FROM_CKPT:-1}"

WINDDOWN_WALLCLOCK_SECONDS="${WINDDOWN_WALLCLOCK_SECONDS:-220}"
WINDDOWN_ITERATIONS="${WINDDOWN_ITERATIONS:-1600}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"

# Keep architecture + core knobs aligned with Bandit_Wagon/run.sh by default.
MODEL_DIM="${MODEL_DIM:-512}"
USE_CRAWLER="${USE_CRAWLER:-1}"
NUM_LAYERS="${NUM_LAYERS:-11}"
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-4}"
NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-1}"
CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT:-6.0}"
INST_DIM="${INST_DIM:-32}"
CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8:-1}"
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
BIGRAM_DIM="${BIGRAM_DIM:-128}"
MATRIX_LR="${MATRIX_LR:-0.03}"
SCALAR_LR="${SCALAR_LR:-0.025}"
HEAD_LR="${HEAD_LR:-0.008}"
TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-400}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
EVAL_STRIDE="${EVAL_STRIDE:-64}"
DRY_RUN="${DRY_RUN:-0}"
ARM_FILTER="${ARM_FILTER:-}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-results/bandit_wagon_winddown_ab}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/${RUN_TS}}"
LOG_DIR="${RESULT_DIR}/logs"
ARTIFACT_DIR="${RESULT_DIR}/artifacts"
SUMMARY_TSV="${RESULT_DIR}/summary.tsv"
RANKED_TSV="${RESULT_DIR}/ranking.tsv"

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

MODEL_PATH_ABS="$(MODEL_PATH="${MODEL_PATH}" python3 -c 'from pathlib import Path; import os; print(Path(os.environ["MODEL_PATH"]).expanduser().resolve())')"
if [[ ! -f "${MODEL_PATH_ABS}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH_ABS}" >&2
  exit 1
fi

if [[ "${AUTO_ARCH_FROM_CKPT}" == "1" ]]; then
  mapfile -t ckpt_arch < <(MODEL_PATH="${MODEL_PATH_ABS}" python3 - <<'PY'
from __future__ import annotations

import os
from collections.abc import Mapping

import torch

path = os.environ["MODEL_PATH"]
obj = torch.load(path, map_location="cpu")
if isinstance(obj, Mapping) and "model" in obj and isinstance(obj["model"], Mapping):
    sd = dict(obj["model"])
elif isinstance(obj, Mapping):
    sd = dict(obj)
else:
    raise TypeError(f"Unsupported checkpoint type: {type(obj).__name__}")

keys = list(sd.keys())
use_crawler = any(k.startswith("flat_blocks.") or k.startswith("crawler_blocks.") for k in keys)

def count_prefix(prefix: str) -> int:
    idx = set()
    for k in keys:
        if not k.startswith(prefix):
            continue
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            idx.add(int(parts[1]))
    return len(idx)

tok = sd.get("tok_emb.weight")
if tok is None or tok.ndim != 2:
    raise KeyError("tok_emb.weight not found in checkpoint")
vocab_size = int(tok.shape[0])
model_dim = int(tok.shape[1])

bg = sd.get("bigram.embed.weight")
if bg is not None and getattr(bg, "ndim", 0) == 2:
    bigram_vocab_size = int(bg.shape[0])
    bigram_dim = int(bg.shape[1])
else:
    bigram_vocab_size = 0
    bigram_dim = 128

if use_crawler:
    num_flat_layers = count_prefix("flat_blocks.")
    num_crawler_layers = count_prefix("crawler_blocks.")
    loops = 3
    loop_up = count_prefix("loop_inst_up.")
    if loop_up > 0:
        loops = loop_up
    elif "loop_pos" in sd and getattr(sd["loop_pos"], "ndim", 0) >= 2:
        loops = int(sd["loop_pos"].shape[0])
    print(f"USE_CRAWLER=1")
    print(f"NUM_FLAT_LAYERS={num_flat_layers}")
    print(f"NUM_CRAWLER_LAYERS={num_crawler_layers}")
    print(f"CRAWLER_LOOPS={loops}")
else:
    num_layers = count_prefix("blocks.")
    print(f"USE_CRAWLER=0")
    print(f"NUM_LAYERS={num_layers}")

print(f"MODEL_DIM={model_dim}")
print(f"VOCAB_SIZE={vocab_size}")
print(f"BIGRAM_VOCAB_SIZE={bigram_vocab_size}")
print(f"BIGRAM_DIM={bigram_dim}")
PY
  )
  for kv in "${ckpt_arch[@]}"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    case "${key}" in
      USE_CRAWLER) USE_CRAWLER="${val}" ;;
      NUM_FLAT_LAYERS) NUM_FLAT_LAYERS="${val}" ;;
      NUM_CRAWLER_LAYERS) NUM_CRAWLER_LAYERS="${val}" ;;
      CRAWLER_LOOPS) CRAWLER_LOOPS="${val}" ;;
      NUM_LAYERS) NUM_LAYERS="${val}" ;;
      MODEL_DIM) MODEL_DIM="${val}" ;;
      VOCAB_SIZE) VOCAB_SIZE="${val}" ;;
      BIGRAM_VOCAB_SIZE) BIGRAM_VOCAB_SIZE="${val}" ;;
      BIGRAM_DIM) BIGRAM_DIM="${val}" ;;
    esac
  done
fi

echo -e "arm\tseed\tcap_step\tcap_val_bpb\tdiag_post_ema_bpb\tfinal_roundtrip_bpb\tfinal_sliding_bpb\tpeak_alloc_mib\tmeta\tlog" > "${SUMMARY_TSV}"

declare -a ARM_NAMES=(
  "A_control_live"
  "B_ema_only"
  "C_ema_swa25"
  "D_ema_distill24"
  "E_ema_distill36"
  "F_ema_ttt_e1_lr005"
  "G_ema_ttt_e2_lr010"
  "H_ema_ttt_distill24"
)

declare -a ARM_ENVS=(
  "WARMDOWN_ITERS=1200 SKIP_EMA=1 SWA_ENABLED=0 DISTILL_ENABLED=0 DISTILL_STEPS=0 TTT_BURST_ENABLED=0"
  "WARMDOWN_ITERS=1200 SKIP_EMA=0 SWA_ENABLED=0 DISTILL_ENABLED=0 DISTILL_STEPS=0 TTT_BURST_ENABLED=0"
  "WARMDOWN_ITERS=1200 SKIP_EMA=0 SWA_ENABLED=1 SWA_EVERY=25 DISTILL_ENABLED=0 DISTILL_STEPS=0 TTT_BURST_ENABLED=0"
  "WARMDOWN_ITERS=1200 SKIP_EMA=0 SWA_ENABLED=0 DISTILL_ENABLED=1 DISTILL_STEPS=24 DISTILL_LR_FACTOR=0.02 DISTILL_TEMPERATURE=1.5 DISTILL_ALPHA=0.60 TTT_BURST_ENABLED=0"
  "WARMDOWN_ITERS=1600 SKIP_EMA=0 SWA_ENABLED=0 DISTILL_ENABLED=1 DISTILL_STEPS=36 DISTILL_LR_FACTOR=0.03 DISTILL_TEMPERATURE=1.7 DISTILL_ALPHA=0.65 TTT_BURST_ENABLED=0"
  "WARMDOWN_ITERS=1200 SKIP_EMA=0 SWA_ENABLED=0 DISTILL_ENABLED=0 DISTILL_STEPS=0 TTT_BURST_ENABLED=1 TTT_BURST_EPOCHS=1 TTT_BURST_LR_FACTOR=0.05 TTT_BURST_STEPS=64 TTT_BURST_TRIGGER=0.35"
  "WARMDOWN_ITERS=1600 SKIP_EMA=0 SWA_ENABLED=0 DISTILL_ENABLED=0 DISTILL_STEPS=0 TTT_BURST_ENABLED=1 TTT_BURST_EPOCHS=2 TTT_BURST_LR_FACTOR=0.10 TTT_BURST_STEPS=96 TTT_BURST_TRIGGER=0.40"
  "WARMDOWN_ITERS=1600 SKIP_EMA=0 SWA_ENABLED=1 SWA_EVERY=25 DISTILL_ENABLED=1 DISTILL_STEPS=24 DISTILL_LR_FACTOR=0.02 DISTILL_TEMPERATURE=1.5 DISTILL_ALPHA=0.60 TTT_BURST_ENABLED=1 TTT_BURST_EPOCHS=1 TTT_BURST_LR_FACTOR=0.05 TTT_BURST_STEPS=64 TTT_BURST_TRIGGER=0.35"
)

echo "============================================"
echo "  BANDIT_WAGON Ad-hoc Winddown A/B Matrix"
echo "  MODEL_PATH: ${MODEL_PATH_ABS}"
echo "  Seeds: ${SEEDS}"
echo "  NPROC: ${NPROC_PER_NODE}"
echo "  Iterations: ${WINDDOWN_ITERATIONS}"
echo "  Wallclock cap: ${WINDDOWN_WALLCLOCK_SECONDS}s"
if [[ "${USE_CRAWLER}" == "1" ]]; then
  echo "  Model arch: crawler (d${MODEL_DIM}, flat=${NUM_FLAT_LAYERS}, crawler_layers=${NUM_CRAWLER_LAYERS}, loops=${CRAWLER_LOOPS})"
else
  echo "  Model arch: gpt (d${MODEL_DIM}, layers=${NUM_LAYERS})"
fi
echo "  Bigram: vocab=${BIGRAM_VOCAB_SIZE} dim=${BIGRAM_DIM}"
echo "  Dry-run: ${DRY_RUN}"
if [[ -n "${ARM_FILTER}" ]]; then
  echo "  Arm filter: ${ARM_FILTER}"
fi
echo "  Results: ${RESULT_DIR}"
echo "============================================"

rows_written=0
for seed in ${SEEDS//,/ }; do
  for i in "${!ARM_NAMES[@]}"; do
    arm="${ARM_NAMES[$i]}"
    if [[ -n "${ARM_FILTER}" ]] && [[ ! "${arm}" =~ ${ARM_FILTER} ]]; then
      continue
    fi
    arm_env="${ARM_ENVS[$i]}"
    read -r -a arm_kvs <<< "${arm_env}"

    safe_arm="${arm//[^a-zA-Z0-9_\-]/_}"
    log_path="${LOG_DIR}/${safe_arm}_s${seed}.log"
    out_dir="${ARTIFACT_DIR}/${safe_arm}_s${seed}"
    mkdir -p "${out_dir}"

    echo
    echo "==> seed=${seed} arm=${arm}"
    echo "    ${arm_env}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "    [dry-run] skipping torchrun"
      continue
    fi

    env "${arm_kvs[@]}" \
      SEED="${seed}" \
      RUN_ID="bw_winddown_${safe_arm}_s${seed}_${RUN_TS}" \
      INIT_MODEL_PATH="${MODEL_PATH_ABS}" \
      OUTPUT_DIR="${out_dir}" \
      MAX_WALLCLOCK_SECONDS="${WINDDOWN_WALLCLOCK_SECONDS}" \
      ITERATIONS="${WINDDOWN_ITERATIONS}" \
      WARMUP_STEPS="${WARMUP_STEPS}" \
      TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" \
      EVAL_STRIDE="${EVAL_STRIDE}" \
      MATRIX_LR="${MATRIX_LR}" \
      SCALAR_LR="${SCALAR_LR}" \
      HEAD_LR="${HEAD_LR}" \
      TIED_EMBED_LR="${TIED_EMBED_LR}" \
      MODEL_DIM="${MODEL_DIM}" \
      VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
      USE_CRAWLER="${USE_CRAWLER}" \
      NUM_LAYERS="${NUM_LAYERS}" \
      NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
      NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS}" \
      CRAWLER_LOOPS="${CRAWLER_LOOPS}" \
      CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT}" \
      INST_DIM="${INST_DIM}" \
      CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8}" \
      XSA_LAST_N=11 \
      BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE}" \
      BIGRAM_DIM="${BIGRAM_DIM}" \
      ROPE_DIMS=16 \
      MTP_NUM_HEADS=0 \
      LATE_QAT_THRESHOLD=0 \
      TORCHDYNAMO_OPTIMIZE_DDP=0 \
      COMPILE_FULLGRAPH=0 \
      SKIP_GPTQ=1 \
      LOOP_AWARE_GPTQ=0 \
      torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/train_gpt_winddown_adhoc.py" \
      2>&1 | tee "${log_path}"

    meta="$(echo "${arm_env}" | tr ' ' ';')"
    python3 "${SCRIPT_DIR}/parse_winddown_log.py" \
      --log "${log_path}" \
      --arm "${arm}" \
      --seed "${seed}" \
      --meta "${meta}" >> "${SUMMARY_TSV}"
    rows_written=$((rows_written + 1))
  done
done

if [[ "${rows_written}" -eq 0 ]]; then
  echo
  echo "No runs executed. Summary TSV initialized at: ${SUMMARY_TSV}"
  exit 0
fi

python3 - "${SUMMARY_TSV}" "${RANKED_TSV}" <<'PY'
from __future__ import annotations
import csv
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
ranked_path = Path(sys.argv[2])

rows = []
with summary_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        def fnum(v: str) -> float:
            try:
                return float(v)
            except Exception:
                return math.inf
        slide = fnum(row.get("final_sliding_bpb", ""))
        rnd = fnum(row.get("final_roundtrip_bpb", ""))
        primary = slide if math.isfinite(slide) else rnd
        row["rank_primary_bpb"] = f"{primary:.8f}" if math.isfinite(primary) else "-"
        rows.append((primary, rnd, row))

rows.sort(key=lambda x: (x[0], x[1]))

headers = [
    "rank",
    "arm",
    "seed",
    "rank_primary_bpb",
    "final_sliding_bpb",
    "final_roundtrip_bpb",
    "diag_post_ema_bpb",
    "cap_val_bpb",
    "cap_step",
    "peak_alloc_mib",
    "log",
]

with ranked_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
    writer.writeheader()
    for i, (_primary, _rnd, row) in enumerate(rows, start=1):
        out = {k: row.get(k, "-") for k in headers}
        out["rank"] = str(i)
        writer.writerow(out)
PY

echo
echo "Summary TSV: ${SUMMARY_TSV}"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${SUMMARY_TSV}"
else
  cat "${SUMMARY_TSV}"
fi

echo
echo "Ranking TSV: ${RANKED_TSV}"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${RANKED_TSV}"
else
  cat "${RANKED_TSV}"
fi
