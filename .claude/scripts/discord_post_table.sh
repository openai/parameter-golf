#!/usr/bin/env bash
# Post a monospace code-block (table / multi-line) to Discord.
# Usage:
#   discord_post_table.sh < path/to/file
#   cat table.txt | discord_post_table.sh
#   discord_post_table.sh "line1\nline2\nline3"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -ge 1 ]; then
  BODY="$(printf '%b' "$1")"
else
  BODY="$(cat)"
fi

# Wrap in triple-backtick code block for fixed-width rendering.
MSG="$(printf '```\n%s\n```' "$BODY")"

printf '%s' "$MSG" | "$HERE/discord_post.sh"
