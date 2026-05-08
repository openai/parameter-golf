#!/usr/bin/env bash
set -euo pipefail

cd /workspace/parameter-golf

BRANCH="submission/nonrecord-ttt-disabled-runtime"
SUBMISSION_DIR="records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421"
PR_BODY="/tmp/pg_pr_body.md"

git fetch origin "$BRANCH"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
else
  git switch -c "$BRANCH" --track "origin/$BRANCH"
fi
git pull --ff-only

python3 -m json.tool "$SUBMISSION_DIR/submission.json" >/dev/null
python3 -m py_compile "$SUBMISSION_DIR/train_gpt.py"
git diff --check
gh auth status

cat > "$PR_BODY" <<'EOF'
Adds a single non-record 16MB submission documenting the best completed candidate from the 1xH100 deadline batch.

Final candidate:
- candidate_id: `gpuq_ttt_disabled_runtime_control`
- val_bpb: `1.44421409`
- val_loss: `2.438495`
- hardware: `1x NVIDIA H100 80GB HBM3`
- seed: `1337`
- train time: `668.949s`
- model bytes: `10,095,619`
- code bytes: `104,891`
- total package bytes: `10,200,510`

This is submitted as non-record because it was run on 1xH100 rather than 8xH100 SXM, includes one seed, and the packaged run exceeded 600 seconds.

The result is useful as a documented deadline-batch control: disabling the runtime TTT path was the strongest completed candidate in this batch.

Validation:
- `python3 -m json.tool records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/submission.json >/dev/null`
- `python3 -m py_compile records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/train_gpt.py`
- `git diff --check`
EOF

gh pr create \
  --repo openai/parameter-golf \
  --base main \
  --head "tenet-diver:$BRANCH" \
  --title "Non-record: TTT-disabled runtime control (1xH100, val_bpb 1.4442)" \
  --body-file "$PR_BODY"
