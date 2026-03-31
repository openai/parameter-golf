#!/usr/bin/env bash
# new_leg.sh — scaffold a new dated experiment leg from the current track leader
# Usage: bash scripts/new_leg.sh <track> <name>
#   track: neural | crawler
#   name:  short hypothesis name (no spaces, use underscores)
# Example: bash scripts/new_leg.sh neural gptq_warmdown
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TRACK="${1:-}"
NAME="${2:-}"

[[ -n "${TRACK}" ]] || { echo "Usage: bash scripts/new_leg.sh <neural|crawler> <name>"; exit 1; }
[[ -n "${NAME}" ]]  || { echo "Usage: bash scripts/new_leg.sh <neural|crawler> <name>"; exit 1; }
[[ "${TRACK}" == "neural" || "${TRACK}" == "crawler" ]] || { echo "Track must be 'neural' or 'crawler'"; exit 1; }

TODAY="$(date +%Y-%m-%d)"
LEG_DIR="${REPO_ROOT}/${TRACK}/${TODAY}_${NAME}"

[[ ! -d "${LEG_DIR}" ]] || { echo "Already exists: ${LEG_DIR}"; exit 1; }
mkdir -p "${LEG_DIR}"

# Copy leader's train_gpt.py as starting point
LEADER_LEG=$(grep "^Leg:" "${REPO_ROOT}/${TRACK}/LEADER.md" | awk '{print $2}')
LEADER_TRAIN="${REPO_ROOT}/${LEADER_LEG}/train_gpt.py"
[[ -f "${LEADER_TRAIN}" ]] || { echo "Leader train_gpt.py not found: ${LEADER_TRAIN}"; exit 1; }
cp "${LEADER_TRAIN}" "${LEG_DIR}/train_gpt.py"

# 1. HYPOTHESIS
cat > "${LEG_DIR}/hypothesis.md" <<EOF
# Hypothesis: ${NAME}
Date: ${TODAY}
Track: ${TRACK}
Parent: ${LEADER_LEG}

## What changes (ONE variable only)
<!-- Describe exactly what is different from the parent leg. -->

## Why
<!-- What signal or reasoning motivates this change? -->

## Gate target
<!-- What step_avg / loss trend do you need to see at 2000 steps to proceed? -->
EOF

# 2. ABLATION log (filled during gate + run)
cat > "${LEG_DIR}/ablation.md" <<EOF
# Ablation: ${NAME}
Date: ${TODAY}
Track: ${TRACK}
Parent: ${LEADER_LEG}

## Gate (1-GPU, 2000 steps, seed=444)
Status: [ ] pending  [ ] pass  [ ] fail
step_avg:
loss @2000:
Notes:

## Full run (8×H100, 600s, seed=444)
Status: [ ] pending  [ ] pass  [ ] fail
step_avg:
steps:
val_bpb (post-EMA):
int6_sw_bpb:
artifact_bytes:
Code size:

## Confirmation (8×H100, 600s, seed=300)
Status: [ ] pending  [ ] pass  [ ] fail
int6_sw_bpb:
artifact_bytes:
EOF

# 3. RESULTS (filled after confirmation)
cat > "${LEG_DIR}/RESULTS.md" <<EOF
# Results: ${NAME}
Date: ${TODAY}
Track: ${TRACK}
Parent: ${LEADER_LEG}

## Verdict
[ ] PROMOTES  [ ] DOES NOT PROMOTE

## Scores
| Seed | int6_sw_bpb | artifact | vs leader |
|------|-------------|----------|-----------|
| 444  |             |          |           |
| 300  |             |          |           |
| mean |             |          |           |

## What we learned
<!-- Even if it doesn't promote, what does the result tell us? -->

## Next hypothesis
<!-- What should the next leg test, based on this result? -->
EOF

# Blank gate script stub
cat > "${LEG_DIR}/gate.sh" <<'EOF'
#!/usr/bin/env bash
# Gate: 1-GPU, 2000 steps. Run this BEFORE the 8x run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-200}"

env \
  SEED="${SEED}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
  SKIP_GPTQ=1 \
  SKIP_FINAL_EVAL=1 \
  python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
  "${SCRIPT_DIR}/train_gpt.py" \
  2>&1 | tee "${SCRIPT_DIR}/gate_seed${SEED}.log"

echo "--- gate done. check step_avg and loss trend before proceeding to run.sh ---"
EOF
chmod +x "${LEG_DIR}/gate.sh"

echo ""
echo "New leg created: ${LEG_DIR}"
echo ""
echo "  1. hypothesis.md  ← fill in: what changes + why + gate target"
echo "  2. train_gpt.py   ← make ONE change from parent"
echo "  3. gate.sh        ← commit+push, then run on pod (1-GPU, ~\$0.50)"
echo "  4. ablation.md    ← fill gate results"
echo "  5. run.sh         ← write it, commit+push, run 8x on pod (~\$3-4)"
echo "  6. ablation.md    ← fill full run + confirmation results"
echo "  7. RESULTS.md     ← verdict, what we learned, next hypothesis"
echo ""
echo "HYPOTHESIS → ABLATION → RESULTS. Gate before 8x. Always."
