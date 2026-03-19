#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-openai/parameter-golf}"
OUTDIR="${2:-pr_diffs}"
LIMIT="${3:-60}"

mkdir -p "$OUTDIR"

echo "Fetching up to $LIMIT PRs from $REPO..."
gh pr list --repo "$REPO" --state all --limit "$LIMIT" --json number,title,state \
  --jq '.[] | "\(.number)\t\(.state)\t\(.title)"' | while IFS=$'\t' read -r num state title; do
  echo "  PR #$num ($state): $title"
  gh pr view --repo "$REPO" "$num" > "$OUTDIR/pr_${num}_view.txt"
  gh pr diff --repo "$REPO" "$num" > "$OUTDIR/pr_${num}_diff.txt"
done

echo "Done. Diffs saved to $OUTDIR/"
