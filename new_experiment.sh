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
  PARENT="$PARENT_ARG"
else
  PARENT_DIR=""
  PARENT="canonical"
fi

mkdir "$DIR"

# Copy source files. From a parent: train_gpt.py, env.sh, and modules/ (if
# present) ride along — these are the parent's tuned overrides and any extra
# code the experiment depends on. Generated artifacts (plan.md, result.json,
# run.log, logs/, *.pt, *.ptz) are NOT inherited; plan.md and result.json
# are freshly generated below.
if [[ -n "$PARENT_DIR" ]]; then
  cp "$PARENT_DIR/train_gpt.py" "$DIR/train_gpt.py"
  if [[ -f "$PARENT_DIR/env.sh" ]]; then
    cp "$PARENT_DIR/env.sh" "$DIR/env.sh"
    # Rewrite RUN_ID so logs/results land under the new experiment name.
    # Preserve everything else — overrides like SEED, LR_*, WARMDOWN_ITERS
    # are exactly what the agent wants forked.
    sed -i.bak -E "s|^export RUN_ID=.*|export RUN_ID=\"${NAME}\"|" "$DIR/env.sh"
    rm -f "$DIR/env.sh.bak"
  fi
  if [[ -d "$PARENT_DIR/modules" ]]; then
    cp -R "$PARENT_DIR/modules" "$DIR/modules"
  fi
else
  cp "$REPO_ROOT/train_gpt.py" "$DIR/train_gpt.py"
fi

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

# Generate canonical env.sh only when forking from canonical (no parent).
# When a parent_id is given, the parent's env.sh has already been copied
# above with RUN_ID rewritten — overwriting it would defeat the purpose.
if [[ -z "$PARENT_DIR" ]]; then
  cat > "$DIR/env.sh" <<EOF
# Source this from inside the experiment folder before running.
export RUN_ID="${NAME}"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
# WARMDOWN_ITERS >= ITERATIONS triggers the step-based warmdown from step 0,
# yielding an effective LR ramp of (1200 - step) / 1200 across the whole run
# (0.167 at step 0, ~0 by step 200). This is the regime the Apr-18 reference
# baseline ran in (ITERATIONS=200, WARMDOWN_ITERS unset → defaulted to 1200),
# and it's what our smoke needs: full canonical LR (warmdown_iters << iterations)
# is too aggressive for MPS bf16 numerics and NaNs around step 165.
# To run an experiment at full canonical LR, override per-experiment with a
# small WARMDOWN_ITERS plus an explicit LR_WARMUP_STEPS (10–20).
export WARMDOWN_ITERS=1200
# Wallclock cap disabled so lr_mul uses the step-based warmdown branch (the
# wallclock branch's formula doesn't fire for short smokes — see git log).
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=8192
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
# 16384-token val cap keeps eval ~1 s. Do NOT set to 0 — full val on MPS
# takes 60-120 min per experiment (eval is called twice: pre-quant + post-
# int8-quant). Use SEED=42 re-run for marginal-result confirmation instead.
export VAL_TOKENS=16384
# Dense step logs to catch divergence early; default 200 prints only steps
# 1–10 and step 200, leaving the bulk of training invisible.
export TRAIN_LOG_EVERY=5
# Experiment-specific overrides go below:
EOF
fi

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
