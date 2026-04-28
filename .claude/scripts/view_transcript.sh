#!/usr/bin/env bash
# View a Claude Code transcript in a human-readable summary form.
# Default: the most recent transcript for the current project.
#
# Usage:
#   bash .claude/scripts/view_transcript.sh                  # most recent, full view
#   bash .claude/scripts/view_transcript.sh <path>           # specific file
#   bash .claude/scripts/view_transcript.sh --skills-only    # only Skill invocations
#   bash .claude/scripts/view_transcript.sh --tools-summary  # tool-use counts table
#   bash .claude/scripts/view_transcript.sh --evidence       # research-evidence audit
#                                                              (what the Stop hook scopes on)

set -euo pipefail

TRANSCRIPT=""
MODE="full"

for arg in "$@"; do
  case "$arg" in
    --skills-only)    MODE="skills" ;;
    --tools-summary)  MODE="tools" ;;
    --evidence)       MODE="evidence" ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    -*)  echo "Unknown flag: $arg" >&2; exit 1 ;;
    *)   TRANSCRIPT="$arg" ;;
  esac
done

# Resolve transcript: arg > most-recent for current project's .claude transcripts dir.
if [[ -z "$TRANSCRIPT" ]]; then
  PROJECT_KEY="$(pwd | sed 's|/|-|g')"
  PROJECT_DIR="$HOME/.claude/projects/${PROJECT_KEY}"
  TRANSCRIPT=$(ls -t "$PROJECT_DIR"/*.jsonl 2>/dev/null | head -1 || true)
fi

if [[ -z "$TRANSCRIPT" || ! -f "$TRANSCRIPT" ]]; then
  echo "No transcript found (looked in $PROJECT_DIR). Pass an explicit path." >&2
  exit 1
fi

LINES=$(wc -l < "$TRANSCRIPT" | tr -d ' ')
echo "Transcript: $TRANSCRIPT"
echo "Lines: $LINES"
echo "---"

case "$MODE" in
  tools)
    jq -rs '
      [.[] | select(.type == "assistant") | (.message.content // [])[]?
        | select(.type == "tool_use") | .name]
      | group_by(.) | map({name: .[0], count: length}) | sort_by(-.count)
      | "count  tool", "-----  ----",
        (.[] | "\(.count|tostring|. + (" "*(5-length)))  \(.name)")
    ' "$TRANSCRIPT"
    ;;

  skills)
    jq -rs '
      [.[] | select(.type == "assistant") | (.message.content // [])[]?
        | select(.type == "tool_use" and .name == "Skill")
        | (((.. | .timestamp? // empty) | tostring) // "") + " " + (.input.skill // "?")]
      | .[]
    ' "$TRANSCRIPT" 2>/dev/null \
    || jq -r 'select(.type == "assistant") | (.message.content // [])[]?
              | select(.type == "tool_use" and .name == "Skill")
              | "Skill: " + (.input.skill // "?")' "$TRANSCRIPT"
    ;;

  evidence)
    # Replicates exactly what the Stop hook scopes on.
    jq -rs '
      [.[]
        | select(.type == "assistant")
        | .timestamp as $ts
        | (.message.content // [])[]?
        | select(.type == "tool_use")
        | . as $t
        | if   ($t.name == "Edit" or $t.name == "Write" or $t.name == "NotebookEdit")
                and (($t.input.file_path // "") | test("(experiments/[0-9]{4}_|/walks/|/winners/|/journal\\.md$)"))
           then {ts: $ts, kind: "EDIT", detail: $t.input.file_path}
           elif ($t.name == "Skill")
                and (($t.input.skill // "") | test("^(launch-and-await|take-a-walk|promote|noise-floor-sentinel|subagent-handoff|wrap-session)$"))
           then {ts: $ts, kind: "SKILL", detail: $t.input.skill}
           else empty
           end
      ]
      | if length == 0 then "(no research evidence — Stop hook would be a no-op for this transcript)"
        else "Research evidence found — Stop hook would arm.\n",
             (.[] | "  [\(.ts // "?")] \(.kind): \(.detail)")
        end
    ' "$TRANSCRIPT"
    ;;

  full)
    jq -r '
      def trunc(s; n): if (s | length) > n then (s[0:n] + "...") else s end;
      def fmt_hms(ts):
        if (ts // "") == "" then ""
        else (ts | sub("T"; " ") | sub("\\.[0-9]+Z?$"; "") | sub(".*\\s"; ""))
        end;

      . as $e
      | fmt_hms($e.timestamp) as $hms
      | def line(prefix; text):
          "[\($hms)] \(prefix): \(trunc((text | gsub("\n"; " ")); 220))";

      if .type == "user" then
        .message.content as $c
        | if ($c | type) == "string" then
            if ($c | startswith("Stop hook feedback:"))
              then line("[HOOK BLOCK]"; ($c | split("\n")[0:2] | join(" | ")))
            elif ($c | test("^<command-name>"))
              then line("[SLASH CMD]"; ($c | capture("<command-name>(?<n>[^<]*)").n // "?"))
            elif ($c | startswith("<local-command-caveat>"))
              then empty
            else line("[USER]"; $c)
            end
          elif ($c | type) == "array" then
            ($c[]
              | if .type == "tool_result"
                then line("  -> tool_result"; (if (.content | type) == "string" then .content else (.content // [])[0].text // "(structured)" end))
                else empty end)
          else empty end
      elif .type == "assistant" then
        (.message.content // [])[]?
        | if .type == "text" then
            line("[ASSISTANT]"; (.text // ""))
          elif .type == "tool_use" then
            if .name == "Skill"      then line("  * SKILL"; (.input.skill // "?"))
            elif .name == "Bash"     then line("    Bash"; (.input.command // ""))
            elif .name == "Read"     then line("    Read"; (.input.file_path // ""))
            elif .name == "Edit"     then line("    Edit"; (.input.file_path // ""))
            elif .name == "Write"    then line("    Write"; (.input.file_path // ""))
            elif .name == "TodoWrite" then line("    TodoWrite"; ((.input.todos // []) | length | tostring) + " items")
            else line("    \(.name)"; (.input | tostring))
            end
          elif .type == "thinking" then empty
          else empty end
      else empty end
    ' "$TRANSCRIPT"
    ;;
esac
