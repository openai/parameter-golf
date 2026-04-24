#!/usr/bin/env bash
# Usage: cd experiments/NNNN_<slug> && ../../run_experiment.sh
#
# Runs the experiment via `python train_gpt.py > run.log 2>&1`, parses
# metrics from the log, writes result.json, and appends one row to the
# repo-root results.tsv.

set -uo pipefail

if [[ ! -f result.json || ! -f train_gpt.py || ! -f env.sh ]]; then
  echo "Error: must be run from inside experiments/NNNN_<slug>/ (missing result.json, train_gpt.py, or env.sh)" >&2
  exit 1
fi

EXPERIMENT_ID=$(basename "$(pwd)")
REPO_ROOT="$(cd ../.. && pwd)"

# shellcheck source=/dev/null
source env.sh

echo "Running ${EXPERIMENT_ID}..."
set +e
python train_gpt.py > run.log 2>&1
RUN_RC=$?
set -e

# Parse metrics from run.log.
# Pre-quant val_bpb: last `step:N/N val_loss:X val_bpb:Y` line (the separate val_loss line, not the train_loss one).
VAL_BPB_PRE=$(grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+' run.log | tail -1 | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2 || echo "")
VAL_LOSS_PRE=$(grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+' run.log | tail -1 | grep -oE 'val_loss:[0-9.]+' | cut -d: -f2 || echo "")

# Post-quant val_bpb: prefer 8-digit `_exact` line, fall back to 4-digit.
VAL_BPB_POST=$(grep -oE 'final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:[0-9.]+' run.log | tail -1 | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2 || echo "")
VAL_LOSS_POST=$(grep -oE 'final_int8_zlib_roundtrip_exact val_loss:[0-9.]+' run.log | tail -1 | grep -oE 'val_loss:[0-9.]+' | cut -d: -f2 || echo "")
if [[ -z "$VAL_BPB_POST" ]]; then
  VAL_BPB_POST=$(grep -oE 'final_int8_zlib_roundtrip val_loss:[0-9.]+ val_bpb:[0-9.]+' run.log | tail -1 | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2 || echo "")
  VAL_LOSS_POST=$(grep -oE 'final_int8_zlib_roundtrip val_loss:[0-9.]+' run.log | tail -1 | grep -oE 'val_loss:[0-9.]+' | cut -d: -f2 || echo "")
fi

# Step timing: last `step:N/N train_loss:... step_avg:Yms`.
STEP_AVG_MS=$(grep -oE 'step:[0-9]+/[0-9]+ train_loss:[0-9.]+ train_time:[0-9]+ms step_avg:[0-9.]+ms' run.log | tail -1 | grep -oE 'step_avg:[0-9.]+' | cut -d: -f2 || echo "")
NUM_STEPS=$(grep -oE 'step:[0-9]+/[0-9]+ train_loss:[0-9.]+' run.log | tail -1 | grep -oE 'step:[0-9]+' | cut -d: -f2 || echo "")

# Artifact: decimal MB (1_000_000 bytes), per challenge spec.
ARTIFACT_BYTES=$(grep -oE 'Total submission size int8\+zlib: [0-9]+ bytes' run.log | tail -1 | grep -oE '[0-9]+' | head -1 || echo "")
CODE_BYTES=$(grep -oE 'Code size: [0-9]+ bytes' run.log | tail -1 | grep -oE '[0-9]+' | head -1 || echo "")
COMPRESSION_RATIO=$(grep -oE 'payload_ratio:[0-9.]+x' run.log | tail -1 | grep -oE '[0-9.]+' || echo "")

CRASHED="false"
if [[ -z "$VAL_BPB_POST" || $RUN_RC -ne 0 ]]; then
  CRASHED="true"
fi

QUANT_TAX=""
if [[ -n "$VAL_BPB_PRE" && -n "$VAL_BPB_POST" ]]; then
  QUANT_TAX=$(python3 -c "print(round(${VAL_BPB_POST} - ${VAL_BPB_PRE}, 6))")
fi

SIZE_VIOLATION="false"
ARTIFACT_MB=""
if [[ -n "$ARTIFACT_BYTES" ]]; then
  ARTIFACT_MB=$(python3 -c "print(round(${ARTIFACT_BYTES} / 1_000_000, 3))")
  if (( ARTIFACT_BYTES > 16000000 )); then
    SIZE_VIOLATION="true"
  fi
fi

# Update result.json.
VBP_PRE="$VAL_BPB_PRE" VBP_POST="$VAL_BPB_POST" VLP_PRE="$VAL_LOSS_PRE" VLP_POST="$VAL_LOSS_POST" \
  QT="$QUANT_TAX" SAM="$STEP_AVG_MS" NS="$NUM_STEPS" AB="$ARTIFACT_BYTES" AM="$ARTIFACT_MB" \
  CB="$CODE_BYTES" CR="$COMPRESSION_RATIO" CRASHED="$CRASHED" SV="$SIZE_VIOLATION" RC="$RUN_RC" \
python3 <<'PYEOF'
import json, os
def n(s):
    return float(s) if s not in ("", None) else None
with open("result.json") as f:
    r = json.load(f)
r["metrics"] = {
    "val_bpb_pre_quant":  n(os.environ["VBP_PRE"]),
    "val_bpb_post_quant": n(os.environ["VBP_POST"]),
    "val_loss_pre_quant":  n(os.environ["VLP_PRE"]),
    "val_loss_post_quant": n(os.environ["VLP_POST"]),
    "quant_tax":          n(os.environ["QT"]),
    "step_avg_ms":        n(os.environ["SAM"]),
    "num_steps":          n(os.environ["NS"]),
    "artifact_bytes":     n(os.environ["AB"]),
    "artifact_mb":        n(os.environ["AM"]),
    "code_bytes":         n(os.environ["CB"]),
    "compression_ratio":  n(os.environ["CR"]),
}
r["flags"] = {
    "crashed": os.environ["CRASHED"] == "true",
    "size_violation": os.environ["SV"] == "true",
    "exit_code": int(os.environ["RC"]),
}
with open("result.json", "w") as f:
    json.dump(r, f, indent=2)
PYEOF

# Append one row to repo-root results.tsv.
TSV="${REPO_ROOT}/results.tsv"
if [[ ! -f "$TSV" ]]; then
  printf "id\tparent\tval_bpb\tpre_quant_bpb\tquant_tax\tartifact_mb\tstep_avg_ms\tcrashed\tsize_violation\tstatus\tdescription\n" > "$TSV"
fi

PARENT=$(python3 -c "import json; print(json.load(open('result.json'))['parent'])")
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tTODO\tTODO\n" \
  "$EXPERIMENT_ID" "$PARENT" "${VAL_BPB_POST:-null}" "${VAL_BPB_PRE:-null}" "${QUANT_TAX:-null}" "${ARTIFACT_MB:-null}" "${STEP_AVG_MS:-null}" "$CRASHED" "$SIZE_VIOLATION" \
  >> "$TSV"

cat <<EOF

=== ${EXPERIMENT_ID} ===
  val_bpb_post_quant: ${VAL_BPB_POST:-null}
  val_bpb_pre_quant:  ${VAL_BPB_PRE:-null}
  quant_tax:          ${QUANT_TAX:-null}
  step_avg_ms:        ${STEP_AVG_MS:-null}
  num_steps:          ${NUM_STEPS:-null}
  artifact_mb:        ${ARTIFACT_MB:-null}
  crashed:            ${CRASHED}
  size_violation:     ${SIZE_VIOLATION}
  exit_code:          ${RUN_RC}

--- First 10 training steps (sanity-check the trajectory) ---
EOF
grep -E "^step:[0-9]+/[0-9]+ train_loss:" run.log | head -10
cat <<EOF

Next: review run.log if needed, fill status+description in ${REPO_ROOT}/results.tsv.
EOF
