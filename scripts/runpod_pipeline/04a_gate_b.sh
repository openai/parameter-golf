#!/usr/bin/env bash
# Stage 4a: Gate B — coherent 3-seed corrector mean
# Step 1: re-eval seed 0 with best corrector config (eval-only, ~$3.60)
# Step 2: full train+eval seeds 1 and 2 with best corrector config
# Step 3: 3-seed corrector mean vs published baseline mean 1.07280628
# All three seeds use the same corrector config; mean is directly comparable to PR #1610.
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"
RUNS_DIR="${REPO_DIR}/runs"
SUMMARY_FILE="${RUNS_DIR}/ablation_summary.json"
GATE_A_SUMMARY="${RUNS_DIR}/gate_a_summary.json"
CKPT_SEED0="/workspace/checkpoints/seed0"
PYTHON="/opt/pg-venv/bin/python"

# Per-seed artifact limits (16MB - headroom)
SEED1_ARTIFACT_LIMIT=15996808   # 16,000,000 - 3,192
SEED2_ARTIFACT_LIMIT=15989628   # 16,000,000 - 10,372

mkdir -p "${RUNS_DIR}"
exec > >(tee -a "${RUNS_DIR}/04a_gate_b.log") 2>&1

echo "=== Stage 4a: Gate B (3-seed corrector mean) === $(date)"
echo "Step 1: seed-0 corrector re-eval (eval-only on Gate A checkpoint)"
echo "Step 2: seeds 1+2 full train+eval with same corrector config"
echo "Step 3: 3-seed corrector mean vs published 1.07280628"

cd "${REPO_DIR}"

[ -f "${GATE_A_SUMMARY}" ] || {
    echo "ERROR: ${GATE_A_SUMMARY} not found — run 02_gate_a.sh first" >&2; exit 1
}
[ -f "${CKPT_SEED0}/final_model.int6.ptz" ] || {
    echo "ERROR: ${CKPT_SEED0}/final_model.int6.ptz not found — run 02_gate_a.sh first" >&2; exit 1
}

# Resolve best corrector config
if [ -n "${BEST_ALPHA:-}" ] && [ -n "${BEST_ORDERS:-}" ]; then
    echo "Config from environment: BEST_ALPHA=${BEST_ALPHA} BEST_ORDERS=${BEST_ORDERS}"
elif [ -f "${SUMMARY_FILE}" ]; then
    BEST_ALPHA=$(  "${PYTHON}" -c "import json; d=json.load(open('${SUMMARY_FILE}')); print(d['best_config_details']['CORRECTOR_ALPHA'])")
    BEST_ORDERS=$( "${PYTHON}" -c "import json; d=json.load(open('${SUMMARY_FILE}')); print(d['best_config_details']['CORRECTOR_ORDERS'])")
    echo "Config from ablation_summary.json: BEST_ALPHA=${BEST_ALPHA} BEST_ORDERS=${BEST_ORDERS}"
else
    echo "ERROR: set BEST_ALPHA/BEST_ORDERS env vars or run 03_ablations.sh first" >&2
    exit 1
fi

parse_result() {
    # parse_result <log_file> <artifact_limit|0> <seed_label> <result_json>
    "${PYTHON}" - "$1" "$2" "$3" "$4" <<'PY'
import re, sys, json, pathlib

log_text       = pathlib.Path(sys.argv[1]).read_text()
limit          = int(sys.argv[2])
seed_label     = sys.argv[3]
result_json    = pathlib.Path(sys.argv[4])

m = re.search(
    r"quantized_ttt_phased val_loss:[0-9.]+ val_bpb:([0-9.]+) eval_time:([0-9]+)ms",
    log_text
)
if not m:
    print(f"ERROR: {seed_label}: quantized_ttt_phased not found in log {sys.argv[1]}", flush=True)
    sys.exit(1)

bpb     = float(m.group(1))
eval_ms = int(m.group(2))

ms = re.search(r"Total submission size quantized\+\S+: ([0-9]+) bytes", log_text)
artifact_bytes = int(ms.group(1)) if ms else 0

print(f"  {seed_label}: bpb={bpb:.8f}  eval={eval_ms/1000:.1f}s  artifact={artifact_bytes}B")

errors = []
if eval_ms > 600_000:
    errors.append(f"eval_time {eval_ms/1000:.0f}s > 600s limit")
if limit > 0 and artifact_bytes > limit:
    errors.append(f"artifact {artifact_bytes}B > {limit}B limit (headroom violation)")
if errors:
    for e in errors:
        print(f"GATE_B {seed_label}: FAIL — {e}", flush=True)
    sys.exit(1)

result_json.write_text(json.dumps({
    "label": seed_label, "bpb": bpb, "eval_ms": eval_ms, "artifact_bytes": artifact_bytes
}, indent=2))
print(f"  {seed_label}: OK — saved to {result_json}")
PY
}

# ── Step 1: seed-0 corrector re-eval (eval-only, ~$3.60) ───────────────────────
echo ""
echo "--- Step 1: seed-0 corrector eval (eval-only on Gate A checkpoint) ---"
SEED0_CORR_DIR="${RUNS_DIR}/seed0_corrector"
SEED0_CORR_LOG="${RUNS_DIR}/seed0_corrector_log.txt"
SEED0_CORR_JSON="${RUNS_DIR}/gate_b_seed0_corrector.json"

mkdir -p "${SEED0_CORR_DIR}"

if [ -f "${SEED0_CORR_JSON}" ]; then
    echo "seed-0 corrector result already exists. Skipping."
else
    EVAL_ONLY_QUANTIZED_PATH="${CKPT_SEED0}/final_model.int6.ptz" \
    ARTIFACT_DIR="${SEED0_CORR_DIR}" \
    CORRECTOR_ALPHA="${BEST_ALPHA}" \
    CORRECTOR_ORDERS="${BEST_ORDERS}" \
    PHASED_TTT_ENABLED=1 \
    PHASED_TTT_PREFIX_DOCS=2000 \
    PYTHONUNBUFFERED=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NCCL_DEBUG=WARN \
        /opt/pg-venv/bin/torchrun \
        --standalone --nproc_per_node=8 \
        "${TRAIN_SCRIPT}" \
        2>&1 | tee "${SEED0_CORR_LOG}"

    parse_result "${SEED0_CORR_LOG}" "0" "seed0-corrector" "${SEED0_CORR_JSON}"
fi

# ── Step 2: seeds 1 and 2 full train+eval ──────────────────────────────────────
run_full_seed() {
    local seed="$1"
    local artifact_limit="$2"
    local seed_dir="${RUNS_DIR}/seed${seed}"
    local log_file="${RUNS_DIR}/seed${seed}_log.txt"
    local ckpt_dir="/workspace/checkpoints/seed${seed}"
    local result_json="${RUNS_DIR}/gate_b_seed${seed}.json"

    mkdir -p "${seed_dir}" "${ckpt_dir}"

    echo ""
    echo "--- Seed ${seed}: full train+eval (artifact limit: ${artifact_limit}B) ---"

    if [ -f "${ckpt_dir}/final_model.int6.ptz" ] && [ -f "${result_json}" ]; then
        echo "Seed ${seed}: checkpoint + result already exist. Skipping."
        return 0
    fi

    FREE_GB=$(df -BG /workspace --output=avail 2>/dev/null | tail -1 | tr -d 'G ')
    [ "${FREE_GB:-0}" -lt 20 ] && { echo "ERROR: < 20G free in /workspace" >&2; exit 1; }

    SEED="${seed}" \
    ARTIFACT_DIR="${seed_dir}" \
    CORRECTOR_ALPHA="${BEST_ALPHA}" \
    CORRECTOR_ORDERS="${BEST_ORDERS}" \
    PHASED_TTT_ENABLED=1 \
    PHASED_TTT_PREFIX_DOCS=2000 \
    PYTHONUNBUFFERED=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NCCL_DEBUG=WARN \
        /opt/pg-venv/bin/torchrun \
        --standalone --nproc_per_node=8 \
        "${TRAIN_SCRIPT}" \
        2>&1 | tee "${log_file}"

    # Persist checkpoint before parse (mirrors Gate A fix)
    cp "${seed_dir}/final_model.pt"       "${ckpt_dir}/final_model.pt"
    cp "${seed_dir}/final_model.int6.ptz" "${ckpt_dir}/final_model.int6.ptz"
    echo "  Checkpoint → ${ckpt_dir}: $(du -sh ${ckpt_dir} | cut -f1)"

    parse_result "${log_file}" "${artifact_limit}" "seed${seed}" "${result_json}"
}

run_full_seed 1 "${SEED1_ARTIFACT_LIMIT}"
run_full_seed 2 "${SEED2_ARTIFACT_LIMIT}"

# ── Step 3: 3-seed corrector mean check ───────────────────────────────────────
echo ""
echo "=== Gate B: 3-seed corrector mean check ==="
"${PYTHON}" - "${RUNS_DIR}" "${BEST_ALPHA}" "${BEST_ORDERS}" <<'PY'
import json, pathlib, sys

runs_dir    = pathlib.Path(sys.argv[1])
best_alpha  = float(sys.argv[2])
best_orders = sys.argv[3]

seed0c = json.loads((runs_dir / "gate_b_seed0_corrector.json").read_text())
seed1  = json.loads((runs_dir / "gate_b_seed1.json").read_text())
seed2  = json.loads((runs_dir / "gate_b_seed2.json").read_text())

bpbs = [seed0c["bpb"], seed1["bpb"], seed2["bpb"]]
mean_bpb       = sum(bpbs) / 3.0
published_mean = 1.07280628   # PR #1610 published 3-seed baseline mean
upper_limit    = published_mean + 0.002  # one-sided: improvements (lower BPB) always pass

print(f"  seed 0 (corrector): {seed0c['bpb']:.8f}")
print(f"  seed 1 (corrector): {seed1['bpb']:.8f}")
print(f"  seed 2 (corrector): {seed2['bpb']:.8f}")
print(f"  corrector mean:     {mean_bpb:.8f}")
print(f"  published baseline: {published_mean:.8f}  (upper_limit: {upper_limit:.8f})")
print(f"  delta vs baseline:  {mean_bpb - published_mean:+.6f} BPB")

# Also read baseline for comparison
ga = json.loads((runs_dir / "gate_a_summary.json").read_text())
print(f"\n  Gate A baseline BPB (seed 0, no corrector): {ga['bpb']:.8f}")
print(f"  Corrector gain on seed 0: {ga['bpb'] - seed0c['bpb']:+.6f} BPB")

if mean_bpb > upper_limit:
    print(f"\nGATE_B: FAIL — corrector mean {mean_bpb:.8f} > upper_limit {upper_limit:.8f}")
    sys.exit(1)

summary = {
    "seed0_corrector_bpb": seed0c["bpb"],
    "seed1_bpb":  seed1["bpb"],
    "seed2_bpb":  seed2["bpb"],
    "corrector_mean_bpb":    mean_bpb,
    "published_baseline_mean": published_mean,
    "delta_from_published":  mean_bpb - published_mean,
    "corrector_alpha": best_alpha,
    "corrector_orders": best_orders,
    "status": "PASS"
}
out = runs_dir / "gate_b_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\nGATE_B: PASS — corrector 3-seed mean {mean_bpb:.8f}")
print(f"Summary → {out}")
PY

echo ""
echo "04a_gate_b: PASS"
echo "Next: UPLOAD_TARGET=hf:<repo>:<path> bash scripts/runpod_pipeline/05_preserve_artifacts.sh"
