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

# Blank hypothesis template
cat > "${LEG_DIR}/hypothesis.md" <<EOF
# Hypothesis: ${NAME}
Date: ${TODAY}
Track: ${TRACK}
Parent: ${LEADER_LEG}

## What changes
<!-- ONE variable. Describe exactly what is different from the parent leg. -->

## Why
<!-- What signal or reasoning motivates this change? -->

## Gate target
<!-- What do you need to see at 2000 steps to proceed to 8x? -->

## Result
<!-- Fill in after gate / full run -->
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
echo "  1. Edit ${LEG_DIR}/hypothesis.md — write what ONE thing changed"
echo "  2. Edit ${LEG_DIR}/train_gpt.py — make the change"
echo "  3. bash ${LEG_DIR}/gate.sh      — 1-GPU gate (~\$0.50)"
echo "  4. If gate passes: write run.sh and launch 8x"
echo ""
echo "GATE FIRST. THEN 8x."
