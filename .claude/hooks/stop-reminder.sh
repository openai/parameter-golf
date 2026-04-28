#!/usr/bin/env bash
# Stop reminder — soft nudge for autoresearch sessions.
#
# Scope (all three gates must pass for the nudge to fire):
#   1. Branch matches `autoresearch-*`.
#   2. This session has done actual research work — at least one Edit/Write
#      under `experiments/NNNN_<slug>/`, OR a Bash invocation of
#      run_experiment.sh / await_steps.sh / new_experiment.sh. Read or `ls`
#      of experiments/ does NOT count: tooling sessions on the same branch
#      (like the one editing this hook) routinely inspect experiments
#      without doing research.
#   3. No training process is currently running (`python … train_gpt.py`).
#      The agent legitimately ends turns to await long background runs;
#      nudging mid-await would loop forever.
#
# Failure mode this guards against: an open-ended overnight loop on a
# non-research conversation on the same branch, where the hook keeps
# blocking stops and the agent keeps responding, burning tokens.
#
# Behavior on a passing session:
#   - Look at the most recent text content block across all assistant
#     messages in the transcript. If it contains SENTINEL → allow.
#     Otherwise → block with the reminder.
#
# Why "latest text content block" rather than "latest assistant message":
#   Claude Code splits a single agent turn into separate JSONL entries — one
#   for thinking, one for text, one per tool_use. They land at slightly
#   different timestamps. Filtering by "no tool_use" alone catches thinking
#   entries (which have no text), and the matcher can race with the writer
#   if the thinking line lands but the text line hasn't flushed yet. Working
#   directly off "latest text block ever spoken" sidesteps the race.
#
# Why `contains` not `==`: tool_use inputs aren't text blocks, so accidental
# quoting of the sentinel inside Bash/Edit args doesn't leak. Within a real
# spoken turn, accidental sentinel quotes are rare enough that contains is
# fine — and forgiving wording (e.g. "OK. Human asked me to stop. ...
# Stopping now.") still works.
#
# A small sleep below buffers against the writer race.
#
# Edit SENTINEL or REMINDER below to change wording. They must agree: the
# REMINDER instructs the agent to emit SENTINEL verbatim.

set -euo pipefail

BRANCH=$(git -C "${CLAUDE_PROJECT_DIR:-.}" branch --show-current 2>/dev/null || echo "")
[[ "$BRANCH" == autoresearch-* ]] || exit 0

# Gate 3: skip while an experiment is in flight. We match three layers:
#   - python … train_gpt.py     → the ~5 min training process (bulk of wallclock)
#   - run_experiment.sh         → ~1 s setup before python, plus post-python
#                                 metrics extraction / results.tsv append
#   - await_steps.sh            → mid-run gates the agent stacks while waiting
# Any one of these alive means the agent is legitimately mid-experiment and
# may end a turn to await completion — nudging here would loop forever.
if pgrep -f '(python[^[:space:]]* train_gpt\.py|run_experiment\.sh|await_steps\.sh)' >/dev/null 2>&1; then
  exit 0
fi

SENTINEL="Human asked me to stop. I have wrap-session finished. Stopping now."
REMINDER="You shouldn't be stopping unless the human explicitly told you to stop. If unclear or stuck, invoke outside-eyes for a fresh read on where to go next. If the human did say stop and you've wrapped, end your final spoken turn with this line: ${SENTINEL}"

INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // ""')
[[ -f "$TRANSCRIPT" ]] || exit 0

# Gate 2: this session has actually done research work. Scan all assistant
# tool_use entries for either an Edit/Write/NotebookEdit on an experiment
# file, or a Bash that invokes one of the experiment scripts.
ENGAGED=$(jq -rs '
  [ .[]
    | select(.type == "assistant")
    | (.message.content // [])[]?
    | select(.type == "tool_use")
    | select(
        ((.name == "Edit" or .name == "Write" or .name == "NotebookEdit")
          and ((.input.file_path // "") | test("/experiments/[0-9]{4}_")))
        or
        (.name == "Bash" and ((.input.command // "") | test("(run_experiment|await_steps|new_experiment)\\.sh")))
      )
  ] | length
' "$TRANSCRIPT" 2>/dev/null || echo 0)
[[ "${ENGAGED:-0}" -gt 0 ]] || exit 0

# Buffer against the harness writing the agent's text entry slightly after
# the Stop hook fires.
sleep 1.0

# Most recent text content block across all assistant messages.
LATEST_TEXT=$(jq -rs '
  [.[]
    | select(.type == "assistant")
    | (.message.content // [])[]?
    | select(.type == "text")
    | .text // ""
  ] | last // ""
' "$TRANSCRIPT" 2>/dev/null || echo "")

# Diagnostic log (one line per fire) — easy to remove later.
{
  echo "=== $(date +%H:%M:%S) transcript_lines=$(wc -l < "$TRANSCRIPT" 2>/dev/null) ==="
  echo "latest_text[0:200]: ${LATEST_TEXT:0:200}"
  if [[ "$LATEST_TEXT" == *"$SENTINEL"* ]]; then echo "match: YES → ALLOW"; else echo "match: NO → BLOCK"; fi
} >> /tmp/stop-reminder-debug.log

if [[ "$LATEST_TEXT" == *"$SENTINEL"* ]]; then
  exit 0
fi

jq -nc --arg r "$REMINDER" '{decision:"block", reason:$r}'
