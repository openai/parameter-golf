#!/usr/bin/env bash
# Rotate per-session journal entries out of journal.md into journals/YYYY-MM-DD_<slug>.md.
# Keeps everything ABOVE "## Entries (newest first)" in journal.md (Current threads,
# Confirmed-paying axes, Dead axes, Open questions) and moves all entries below it
# into the dated archive file.
#
# Usage: bash rotate.sh <slug>
#   slug — short kebab-case session identifier, e.g. "recurrence-swiglu"

set -euo pipefail

SLUG="${1:-}"
if [[ -z "$SLUG" ]]; then
  echo "Usage: $0 <slug>   (e.g. recurrence-swiglu)" >&2
  exit 1
fi

DATE=$(date +%Y-%m-%d)
JOURNAL="journal.md"
TARGET="journals/${DATE}_${SLUG}.md"

if [[ ! -f "$JOURNAL" ]]; then
  echo "Error: $JOURNAL not found. Run from repo root." >&2
  exit 1
fi

if [[ -f "$TARGET" ]]; then
  echo "Error: $TARGET already exists. Pick a different slug or remove it first." >&2
  exit 1
fi

ENTRIES_LINE=$(grep -n '^## Entries' "$JOURNAL" | head -1 | cut -d: -f1)
if [[ -z "$ENTRIES_LINE" ]]; then
  echo "Error: no '## Entries' section header found in $JOURNAL — nothing to rotate." >&2
  exit 1
fi

mkdir -p journals

{
  echo "# Journal · ${DATE} · ${SLUG}"
  echo ""
  echo "Rotated from journal.md on $(date '+%Y-%m-%d %H:%M %Z')."
  echo ""
  sed -n "${ENTRIES_LINE},\$p" "$JOURNAL"
} > "$TARGET"

KEEP_LINE=$((ENTRIES_LINE - 1))
{
  sed -n "1,${KEEP_LINE}p" "$JOURNAL"
  echo ""
  echo "## Entries (newest first)"
  echo ""
} > "${JOURNAL}.tmp"
mv "${JOURNAL}.tmp" "$JOURNAL"

ROTATED_LINES=$(wc -l < "$TARGET")
KEPT_LINES=$(wc -l < "$JOURNAL")
echo "Rotated ${ROTATED_LINES} lines into ${TARGET}"
echo "journal.md is now ${KEPT_LINES} lines (Current threads / axes / Open questions retained)"
echo ""
echo "Next step: curate Current threads — drop episodic Prior-winner lines, keep only durable knowledge."
