#!/usr/bin/env bash
# Post a message to Discord. Text from $1 or stdin.
# Usage:
#   discord_post.sh "one-line message"
#   echo -e "multi\nline" | discord_post.sh
set -euo pipefail

SECRETS="${DISCORD_SECRETS:-/home/claude-user/.claude/skills/secrets.env}"
set -a; source "$SECRETS"; set +a

if [ $# -ge 1 ]; then
  MSG="$1"
else
  MSG="$(cat)"
fi

# Discord hard-caps message content at 2000 chars.
if [ "${#MSG}" -gt 1990 ]; then
  MSG="${MSG:0:1980}…(truncated)"
fi

curl -sS -X POST \
  -H "Authorization: Bot $DISCORD_BOT_TOKEN" \
  -H "Content-Type: application/json" \
  --data-raw "$(jq -n --arg c "$MSG" '{content:$c}')" \
  "https://discord.com/api/v10/channels/$DISCORD_CHANNEL/messages" > /dev/null
