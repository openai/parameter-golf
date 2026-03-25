# Podracer RED Hypothesis (safe-only lane)

Date: 2026-03-25

## Goal
Improve Podracer II without enabling any test-time training behavior.

## Hypothesis
If we keep the same base model but increase n-gram headroom in uncertain regions and reduce hash collisions, we can gain a small but real BPB improvement with no TTT.

Changes in this lane:
- Keep multi-order backoff at order 7.
- Raise `NGRAM_EVAL_ALPHA_MAX` from `0.60` to `0.70` so uncertain tokens can lean more on backoff.
- Lower `NGRAM_EVAL_ENTROPY_CENTER` from `4.0` to `3.0` so adaptive mixing engages earlier.
- Double `NGRAM_EVAL_BUCKETS` from `4,194,304` to `8,388,608` to reduce collisions.

## Safety Guardrails
- `TTT_EVAL_ENABLED=0`
- `TTT_EPOCHS=0`
- `TTT_MAX_TRAIN_CHUNKS=0`
- No oracle routing or min-NLL branch selection.
- No leaderboard-driven online adaptation in this run recipe.

## Expected Gain Band (vs Podracing II baseline)
- Likely: `+0.001` to `+0.006` BPB improvement
- Neutral band: `-0.002` to `+0.001`
- Downside tail: up to `-0.004`
