#!/usr/bin/env bash
# PreToolUse hook for the Agent tool (subagent spawning) — blocks the call if
# the referenced experiment's plan.md still has any of the four template
# <!-- ... --> sections unfilled. Mirrors the gate in run_experiment.sh so
# subagent edits and experiment runs are held to the same plan-completeness bar.

set -euo pipefail

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.tool_input.prompt // ""')

# Look for any reference to an experiments/NNNN_<slug> path in the prompt.
# If the subagent isn't being pointed at an experiment, allow.
EXP_DIR=$(echo "$PROMPT" | grep -oE 'experiments/[0-9]+_[a-zA-Z0-9_-]+' | head -1 || true)

if [[ -z "$EXP_DIR" ]]; then
  exit 0
fi

PLAN="${CLAUDE_PROJECT_DIR}/${EXP_DIR}/plan.md"

if [[ ! -f "$PLAN" ]]; then
  exit 0
fi

UNFILLED_PATTERNS=(
  '<!-- What are you actually asking'
  '<!-- Predicted direction and magnitude'
  '<!-- Exact env vars'
  '<!-- What outcome would falsify'
)
UNFILLED_COUNT=0
for pat in "${UNFILLED_PATTERNS[@]}"; do
  if grep -qF "$pat" "$PLAN"; then
    UNFILLED_COUNT=$((UNFILLED_COUNT + 1))
  fi
done

if (( UNFILLED_COUNT > 0 )); then
  jq -n --arg dir "$EXP_DIR" --arg n "$UNFILLED_COUNT" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: ("\($dir)/plan.md has \($n) unfilled <!-- ... --> template section(s). Fill Question, Hypothesis, Change, and Disconfirming with real content (replace the placeholders, do not just append below them) before spawning the subagent.")
    }
  }'
  exit 0
fi

exit 0
