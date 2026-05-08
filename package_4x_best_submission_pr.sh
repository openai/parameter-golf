#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
ONE_X_BATCH="${ONE_X_BATCH:-records/h100_candidate_batch/20260430T210340Z}"
FOUR_X_BATCH="${FOUR_X_BATCH:-records/h100_4x_best_candidate_batch/20260501T002059Z}"
AUTHOR="${AUTHOR:-Rustam Eynaliyev}"
GITHUB_ID="${GITHUB_ID:-tenet-diver}"
FORK_REPO="${FORK_REPO:-$GITHUB_ID/parameter-golf}"
BASE_REMOTE="${BASE_REMOTE:-upstream}"
BASE_REPO_URL="${BASE_REPO_URL:-https://github.com/openai/parameter-golf.git}"
BASE_BRANCH="${BASE_BRANCH:-main}"
WINNER_ID="${WINNER_ID:-best4x_ttt_disabled_qk525}"
WINNER_BRANCH="${WINNER_BRANCH:-submission/4xh100-qk525-ttt-disabled}"
PR_TITLE="${PR_TITLE:-Non-record: 4xH100 QK5.25 TTT-disabled promotion}"

declare -a ALL_KEYS=()
declare -A KIND=()
declare -A BATCH=()
declare -A NPROC=()
declare -A CANDIDATE_ID=()
declare -A SCORE=()
declare -A BRANCH=()
declare -A SLUG=()
declare -A SUBMISSION_NAME=()
declare -A DESCRIPTION=()
declare -A SUBMISSION_DIR=()

die() {
  echo "error: $*" >&2
  exit 1
}

safe_key() {
  printf '%s:%s' "$1" "$2"
}

batch_table() {
  local batch="$1"
  python3 - "$batch/results.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
completed = [
    row for row in rows
    if row.get("status") == "completed" and row.get("finalValBpb") is not None
]
for row in sorted(completed, key=lambda r: float(r["finalValBpb"])):
    cid = str(row.get("id") or "")
    bpb = float(row["finalValBpb"])
    desc = str(row.get("description") or row.get("hypothesis") or "").replace("\n", " ").strip()
    if len(desc) > 140:
        desc = desc[:137].rstrip() + "..."
    print(f"| `{cid}` | {bpb:.8f} | {desc} |")
PY
}

load_batch_metadata() {
  local kind="$1"
  local batch="$2"
  local nproc="$3"
  local branch_prefix="$4"
  local title_prefix="$5"

  [ -f "$batch/results.json" ] || die "missing batch results: $batch/results.json"
  [ -f "$batch/leaderboard.csv" ] || die "missing batch leaderboard: $batch/leaderboard.csv"

  while IFS=$'\t' read -r candidate_id score branch slug name description; do
    local key
    key="$(safe_key "$kind" "$candidate_id")"
    ALL_KEYS+=("$key")
    KIND[$key]="$kind"
    BATCH[$key]="$batch"
    NPROC[$key]="$nproc"
    CANDIDATE_ID[$key]="$candidate_id"
    SCORE[$key]="$score"
    BRANCH[$key]="$branch"
    SLUG[$key]="$slug"
    SUBMISSION_NAME[$key]="$name"
    DESCRIPTION[$key]="$description"
  done < <(
    python3 - "$batch/results.json" "$kind" "$branch_prefix" "$title_prefix" "$WINNER_ID" "$WINNER_BRANCH" <<'PY'
import json
import re
import sys

results_path, kind, branch_prefix, title_prefix, winner_id, winner_branch = sys.argv[1:7]
rows = json.load(open(results_path, encoding="utf-8"))

def safe_slug(raw: str, max_len: int = 72) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.lower()).strip("-._")
    value = re.sub(r"-+", "-", value)
    return (value[:max_len].strip("-._") or "submission")

completed = [
    row for row in rows
    if row.get("status") == "completed" and row.get("finalValBpb") is not None
]
for row in sorted(completed, key=lambda r: float(r["finalValBpb"])):
    cid = str(row.get("id") or "")
    score = float(row["finalValBpb"])
    score_slug = f"{score:.5f}".replace(".", "p")
    if kind == "4x" and cid == winner_id:
        branch = winner_branch
    else:
        branch = f"evidence/{branch_prefix}-{safe_slug(cid, 58)}"
    slug = f"2026-05-01_{safe_slug(branch_prefix + '-' + cid, 82)}_{score_slug}"
    desc = str(row.get("description") or row.get("hypothesis") or cid).strip().rstrip(".")
    name = f"{title_prefix}: {desc}"
    desc = desc.replace("\t", " ").replace("\n", " ")
    print("\t".join([cid, f"{score:.8f}", branch, slug, name, desc]))
PY
  )
}

package_record() {
  local key="$1"
  local output
  local dir

  output="$(
    python3 fastest/scripts/package_competition_submission.py \
      --batch-dir "${BATCH[$key]}" \
      --candidate-id "${CANDIDATE_ID[$key]}" \
      --records-root records/track_non_record_16mb \
      --track non-record-unlimited-compute-16mb \
      --nproc-per-node "${NPROC[$key]}" \
      --author "$AUTHOR" \
      --github-id "$GITHUB_ID" \
      --name "${SUBMISSION_NAME[$key]}" \
      --slug "${SLUG[$key]}" \
      --force
  )"

  echo "$output"
  dir="$(printf '%s\n' "$output" | awk -F= '/^submission_dir=/{print $2}')"
  [ -n "$dir" ] || die "could not parse submission_dir for ${CANDIDATE_ID[$key]}"
  [ -d "$dir" ] || die "packaging did not create directory: $dir"

  cp "${BATCH[$key]}/leaderboard.csv" "$dir/batch_leaderboard.csv"
  cp "${BATCH[$key]}/summary.md" "$dir/batch_summary.md"
  cp "${BATCH[$key]}/results.json" "$dir/batch_results.json"

  python3 -m json.tool "$dir/submission.json" >/dev/null
  python3 -m py_compile "$dir/train_gpt.py"
  SUBMISSION_DIR[$key]="$dir"
}

commit_push_branch() {
  local key="$1"
  local branch="${BRANCH[$key]}"
  local dir="${SUBMISSION_DIR[$key]}"

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git switch "$branch"
  else
    git switch -c "$branch" "$BASE_REMOTE/$BASE_BRANCH"
  fi

  git add "$dir"
  git diff --check

  if git diff --cached --quiet; then
    echo "No staged changes for $branch; continuing."
  else
    if [ "${KIND[$key]}" = "4x" ] && [ "${CANDIDATE_ID[$key]}" = "$WINNER_ID" ]; then
      git commit -m "Add ${SUBMISSION_NAME[$key]} non-record submission"
    else
      git commit -m "Add ${SUBMISSION_NAME[$key]} evidence package"
    fi
  fi

  git push -u origin "$branch"
}

evidence_links_table() {
  local winner_key="$1"
  local key
  for key in "${ALL_KEYS[@]}"; do
    if [ "$key" = "$winner_key" ]; then
      continue
    fi
    printf '| `%s` | %s | %s | [branch](https://github.com/%s/tree/%s) |\n' \
      "${CANDIDATE_ID[$key]}" \
      "${KIND[$key]}" \
      "${SCORE[$key]}" \
      "$FORK_REPO" \
      "${BRANCH[$key]}"
  done
}

write_winner_pr_body() {
  local winner_key="$1"
  local path="$2"
  local winner_dir="${SUBMISSION_DIR[$winner_key]}"
  local winner_score="${SCORE[$winner_key]}"
  local four_x_table
  local one_x_table
  local evidence_table
  four_x_table="$(batch_table "$FOUR_X_BATCH")"
  one_x_table="$(batch_table "$ONE_X_BATCH")"
  evidence_table="$(evidence_links_table "$winner_key")"

  cat > "$path" <<PRBODY
Adds the single primary non-record 16MB submission from the deadline search.

We did not get access to an 8xH100 box before the deadline, so this is intentionally submitted to the non-record / unlimited-compute track rather than as a leaderboard-record claim. The goal of this PR is to submit the strongest completed result and link the supporting evidence branches from the wider search.

Search process:
- Started with 12 implementation-backed candidates on 1xH100.
- Promoted the four most promising completed 1xH100 candidates into a 4xH100 batch.
- Could not take the strongest 4xH100 candidate further to a compliant 8xH100 / 3-seed record attempt before the deadline.

Primary packaged result:
- candidate_id: \`$WINNER_ID\`
- val_bpb: \`$winner_score\`
- track: \`non-record-unlimited-compute-16mb\`
- hardware: \`4x NVIDIA H100 80GB HBM3\`
- artifact: under 16MB

Approach:
- Uses the QK-gain 5.25 autoregressive control stack.
- Primary candidate disables test-time training, testing whether saved runtime is more valuable than legal score-first TTT for this smaller SP1024 stack.
- The promoted 4x batch compares this against legal TTT, parallel-residual + legal TTT, and a dense optimizer/tuning baseline.

4xH100 promoted batch:
| Candidate | val_bpb | Description |
|---|---:|---|
$four_x_table

1xH100 screening batch:
| Candidate | val_bpb | Description |
|---|---:|---|
$one_x_table

Evidence branches:
| Candidate | Batch | val_bpb | Link |
|---|---|---:|---|
$evidence_table

The evidence branches are not separate submission PRs. They are pushed only so reviewers can inspect the individual packaged records behind the search.

Included in this PR:
- \`submission.json\`
- self-contained \`train_gpt.py\`
- primary train log and result JSON
- \`batch_leaderboard.csv\`
- \`batch_summary.md\`
- \`batch_results.json\`

Validation:
- \`python3 -m json.tool $winner_dir/submission.json >/dev/null\`
- \`python3 -m py_compile $winner_dir/train_gpt.py\`
- \`git diff --check\`
PRBODY
}

create_or_update_winner_pr() {
  local body_file="$1"
  local url
  local number

  if url="$(
    gh pr create \
      --repo openai/parameter-golf \
      --base "$BASE_BRANCH" \
      --head "$GITHUB_ID:$WINNER_BRANCH" \
      --title "$PR_TITLE" \
      --body-file "$body_file" 2>/tmp/pg_gh_pr_error.txt
  )"; then
    printf '%s\n' "$url" | tail -1
    return 0
  fi

  number="$(
    gh pr list \
      --repo openai/parameter-golf \
      --head "$GITHUB_ID:$WINNER_BRANCH" \
      --json number \
      --jq '.[0].number' 2>/dev/null || true
  )"
  if [ -n "$number" ] && [ "$number" != "null" ]; then
    gh pr edit "$number" --repo openai/parameter-golf --body-file "$body_file" >/dev/null
    gh pr view "$number" --repo openai/parameter-golf --json url --jq '.url'
    return 0
  fi

  cat /tmp/pg_gh_pr_error.txt >&2
  return 1
}

cd "$REPO_DIR"

[ -f fastest/scripts/package_competition_submission.py ] || die "missing packaging script; run from codex/frontier-deadline-experiments"
[ -f "$ONE_X_BATCH/results.json" ] || die "missing 1x batch results: $ONE_X_BATCH/results.json"
[ -f "$FOUR_X_BATCH/results.json" ] || die "missing 4x batch results: $FOUR_X_BATCH/results.json"

echo "1xH100 batch leaderboard:"
cat "$ONE_X_BATCH/leaderboard.csv"
echo
echo "4xH100 batch leaderboard:"
cat "$FOUR_X_BATCH/leaderboard.csv"

if ! git remote get-url "$BASE_REMOTE" >/dev/null 2>&1; then
  git remote add "$BASE_REMOTE" "$BASE_REPO_URL"
fi
git fetch "$BASE_REMOTE" "$BASE_BRANCH:refs/remotes/$BASE_REMOTE/$BASE_BRANCH"

load_batch_metadata "1x" "$ONE_X_BATCH" "1" "h100" "1xH100 screening"
load_batch_metadata "4x" "$FOUR_X_BATCH" "4" "4xh100" "4xH100 promotion"

for key in "${ALL_KEYS[@]}"; do
  package_record "$key"
done

winner_key="$(safe_key "4x" "$WINNER_ID")"
[ -n "${SUBMISSION_DIR[$winner_key]:-}" ] || die "winner key missing: $winner_key"

for key in "${ALL_KEYS[@]}"; do
  commit_push_branch "$key"
done

winner_body="/tmp/pg_${WINNER_ID}_winner_body.md"
write_winner_pr_body "$winner_key" "$winner_body"

gh auth status
winner_pr_url="$(create_or_update_winner_pr "$winner_body")"
echo "winner_pr_url=$winner_pr_url"
