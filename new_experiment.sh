#!/usr/bin/env bash
# Usage: ./new_experiment.sh <slug> [<parent_id>]
#
# Creates experiments/NNNN_<slug>/, forked from the canonical train_gpt.py
# at repo root by default. If a parent_id is given (e.g. "0007_big_mlp"),
# forks from experiments/<parent_id>/train_gpt.py instead.
#
# Must be invoked from repo root.

set -euo pipefail

SLUG="${1:-}"
PARENT_ARG="${2:-}"

if [[ -z "$SLUG" ]]; then
  echo "Usage: $0 <slug> [<parent_id>]" >&2
  exit 1
fi

if ! [[ "$SLUG" =~ ^[a-z0-9_-]+$ ]]; then
  echo "Slug must be lowercase alphanumeric, with - or _ only (got: '$SLUG')" >&2
  exit 1
fi

REPO_ROOT="$(pwd)"
if [[ ! -f "$REPO_ROOT/train_gpt.py" ]]; then
  echo "Error: must be run from repo root (train_gpt.py not found here)" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/experiments"

# Reject if this slug is already used at any NNNN_<slug>.
if compgen -G "$REPO_ROOT/experiments/[0-9][0-9][0-9][0-9]_${SLUG}" >/dev/null; then
  echo "Slug '${SLUG}' is already used:" >&2
  ls -d "$REPO_ROOT"/experiments/[0-9][0-9][0-9][0-9]_"${SLUG}" >&2
  exit 1
fi

# Next id = max existing NNNN + 1.
NEXT_ID=1
if compgen -G "$REPO_ROOT/experiments/[0-9][0-9][0-9][0-9]_*" >/dev/null; then
  LAST=$(ls -d "$REPO_ROOT"/experiments/[0-9][0-9][0-9][0-9]_* | sort | tail -1)
  LAST_BASE=$(basename "$LAST")
  NEXT_ID=$((10#${LAST_BASE:0:4} + 1))
fi
ID=$(printf "%04d" "$NEXT_ID")
NAME="${ID}_${SLUG}"
DIR="$REPO_ROOT/experiments/${NAME}"

# Resolve parent source.
if [[ -n "$PARENT_ARG" ]]; then
  PARENT_DIR="$REPO_ROOT/experiments/${PARENT_ARG}"
  if [[ ! -f "$PARENT_DIR/train_gpt.py" ]]; then
    echo "Parent experiment not found: experiments/${PARENT_ARG}/train_gpt.py" >&2
    exit 1
  fi
  SOURCE_FILE="$PARENT_DIR/train_gpt.py"
  PARENT="$PARENT_ARG"
else
  SOURCE_FILE="$REPO_ROOT/train_gpt.py"
  PARENT="canonical"
fi

mkdir "$DIR"
cp "$SOURCE_FILE" "$DIR/train_gpt.py"

cat > "$DIR/plan.md" <<EOF
# Experiment ${NAME}

Parent: ${PARENT}

## Question
<!-- What are you actually asking? Be specific. -->

## Hypothesis [CONJECTURE]
<!-- Predicted direction and magnitude of val_bpb change, with confidence tag. -->

## Change
<!-- Exact env vars / code edits. -->

## Disconfirming
<!-- What outcome would falsify the hypothesis? -->

## Notes from execution
<!-- Filled during edit, or by subagent if invoked. Note any deviations. -->
EOF

cat > "$DIR/env.sh" <<EOF
# Source this from inside the experiment folder before running.
export RUN_ID="${NAME}"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
export WARMDOWN_ITERS=40
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=8192
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
export VAL_TOKENS=16384
# Linear LR warmup: required on MPS to avoid first-step Adam overshoot.
# See program.md "MPS stability" and the git log of train_gpt.py for rationale.
export LR_WARMUP_STEPS=10
# Experiment-specific overrides go below:
EOF

CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$DIR/result.json" <<EOF
{
  "id": "${NAME}",
  "parent": "${PARENT}",
  "created_at": "${CREATED_AT}",
  "metrics": null,
  "flags": {},
  "status": null,
  "description": null
}
EOF

echo "Created experiments/${NAME} (parent: ${PARENT})"
echo ""
echo "Next steps:"
echo "  1. Edit experiments/${NAME}/plan.md   (hypothesis, change, disconfirming)"
echo "  2. Edit experiments/${NAME}/train_gpt.py and/or env.sh"
echo "  3. cd experiments/${NAME} && ../../run_experiment.sh"
