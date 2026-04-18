#!/usr/bin/env bash
# Stage 2: Gate A — full train+eval, seed 0, CORRECTOR_ALPHA=0.0
# Kill if BPB > 1.07516564 (published 1.07216564 + 0.003), eval > 600s, or artifact > limit
# Persists checkpoint to /workspace/checkpoints/seed0/ after PASS
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"
RUNS_DIR="${REPO_DIR}/runs"
SEED0_DIR="${RUNS_DIR}/seed0"
LOG_FILE="${RUNS_DIR}/seed0_log.txt"
CKPT_DIR="/workspace/checkpoints/seed0"
PYTHON="/opt/pg-venv/bin/python"
REQUIRED_COMMIT="a33191f572430566b88c4d61badb0369e1e6f9a3"

# Gate A BPB ceiling: published 1.07216564 + 0.003 tolerance
GATE_A_BPB_CEILING="1.07516564"
# Per-seed artifact limit: 16,000,000 - 2,480 headroom (post-fix corrector code)
SEED0_ARTIFACT_LIMIT=15997520

mkdir -p "${RUNS_DIR}" "${SEED0_DIR}" "${CKPT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Stage 2: Gate A (seed 0) === $(date)"
echo "Published BPB: 1.07216564  |  Kill if > ${GATE_A_BPB_CEILING} (published + 0.003)"
echo "Artifact limit: ${SEED0_ARTIFACT_LIMIT} bytes"

cd "${REPO_DIR}"

# SHA guard
CURRENT_COMMIT=$(git rev-parse HEAD)
if ! git merge-base --is-ancestor "${REQUIRED_COMMIT}" HEAD 2>/dev/null; then
    echo "ERROR: required commit ${REQUIRED_COMMIT} not in history (HEAD=${CURRENT_COMMIT})" >&2
    exit 1
fi
echo "SHA check: ${CURRENT_COMMIT}: OK"

# Idempotency: skip if checkpoint + PASS marker already exist
if [ -f "${CKPT_DIR}/final_model.int6.ptz" ] && grep -q "GATE_A: PASS" "${LOG_FILE}" 2>/dev/null; then
    echo "Checkpoint and PASS marker already present. Skipping training."
    echo "02_gate_a: PASS (cached)"
    exit 0
fi

# Free space guard
FREE_GB=$(df -BG /workspace --output=avail 2>/dev/null | tail -1 | tr -d 'G ')
[ "${FREE_GB:-0}" -lt 20 ] && { echo "ERROR: < 20G free in /workspace" >&2; exit 1; }

echo ""
echo "Starting train+eval (torchrun 8xH100)..."
SEED=0 \
ARTIFACT_DIR="${SEED0_DIR}" \
CORRECTOR_ALPHA=0.0 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 \
PYTHONUNBUFFERED=1 \
MKL_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
NCCL_DEBUG=WARN \
    /opt/pg-venv/bin/torchrun \
    --standalone --nproc_per_node=8 \
    "${TRAIN_SCRIPT}"

# Persist checkpoint BEFORE log parsing — a parse crash must not lose the $7 Gate A run
echo ""
echo "=== Persisting seed-0 checkpoint (before log parse) ==="
cp "${SEED0_DIR}/final_model.pt"       "${CKPT_DIR}/final_model.pt"
cp "${SEED0_DIR}/final_model.int6.ptz" "${CKPT_DIR}/final_model.int6.ptz"
echo "Checkpoints saved: $(du -sh ${CKPT_DIR} | cut -f1)"

echo ""
echo "=== Gate A: Parsing results ==="

"${PYTHON}" - "${LOG_FILE}" "${SEED0_ARTIFACT_LIMIT}" "${GATE_A_BPB_CEILING}" <<'PY'
import re, sys, json, pathlib

log_path    = pathlib.Path(sys.argv[1])
limit       = int(sys.argv[2])
bpb_ceiling = float(sys.argv[3])
log_text    = log_path.read_text()

# BPB + eval_time
m = re.search(
    r"quantized_ttt_phased val_loss:[0-9.]+ val_bpb:([0-9.]+) eval_time:([0-9]+)ms",
    log_text
)
if not m:
    print("ERROR: quantized_ttt_phased line not found in log", flush=True)
    print("  Check for training crash or timeout in", sys.argv[1], flush=True)
    sys.exit(1)

bpb      = float(m.group(1))
eval_ms  = int(m.group(2))
eval_s   = eval_ms / 1000.0

# Artifact size
ms = re.search(r"Total submission size quantized\+\S+: ([0-9]+) bytes", log_text)
if not ms:
    print("ERROR: 'Total submission size' line not found in log", flush=True)
    sys.exit(1)
artifact_bytes = int(ms.group(1))

print(f"  val_bpb:       {bpb:.8f}   (published: 1.07216564  ceiling: {bpb_ceiling})")
print(f"  eval_time:     {eval_s:.1f}s   (limit: 600s)")
print(f"  artifact_size: {artifact_bytes} bytes   (limit: {limit})")

errors = []
if bpb > bpb_ceiling:
    errors.append(f"BPB {bpb:.8f} > {bpb_ceiling} (published + 0.003) — reproduction failed; "
                  f"investigate before spending more GPU time")
if eval_ms > 600_000:
    errors.append(f"eval_time {eval_s:.0f}s > 600s limit")
if artifact_bytes > limit:
    errors.append(f"artifact {artifact_bytes}B > {limit}B seed-0 headroom limit")

if errors:
    print("", flush=True)
    for e in errors:
        print(f"GATE_A: FAIL — {e}", flush=True)
    print(f"  Log: {sys.argv[1]}", flush=True)
    sys.exit(1)

summary = {
    "seed": 0, "bpb": bpb, "eval_ms": eval_ms,
    "artifact_bytes": artifact_bytes, "status": "PASS"
}
out = log_path.parent / "gate_a_summary.json"
out.write_text(json.dumps(summary, indent=2))

print(f"")
print(f"GATE_A: PASS")
print(f"  Summary written to {out}")
PY

echo ""
echo "02_gate_a: PASS"
