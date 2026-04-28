#!/usr/bin/env bash
# Spawn a reviewer subagent via `claude -p` (headless mode). Reads the reviewer
# prompt from reviewer_prompt.md and runs a fresh Claude session against the
# project. The main agent that invokes this script never sees the reviewer
# prompt content — it only receives the review on stdout. Fresh eyes,
# uncontaminated by main-agent anchoring.
#
# Invoked by the agent (via Bash) when the outside-eyes skill is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PROMPT_FILE="$SCRIPT_DIR/reviewer_prompt.md"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "outside_eyes.sh: reviewer prompt not found at $PROMPT_FILE" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "outside_eyes.sh: 'claude' CLI not in PATH; cannot dispatch reviewer subagent" >&2
  exit 1
fi

cd "$PROJECT_DIR"

# Reviewer is restricted to read-only tools — its session-end won't add any
# research evidence to the Stop hook's scope check, so no recursion guard needed.
claude -p \
  --allowed-tools "Read Glob Grep" \
  --output-format text \
  "$(cat "$PROMPT_FILE")"
