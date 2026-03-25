# Podracer RED Hypothesis (racing profile lane)

Date: 2026-03-25

## Goal
Target the proven backoff profile that produced ~0.962 BPB while keeping eval legal and TTT disabled.

## Hypothesis
If we keep the same base model and run the proven 7-gram adaptive profile (order 7, alpha 0.30, alpha_max 0.60, entropy center 4.0, buckets 4,194,304), we should reproduce the ~0.962 band on strong seeds. A safe optional edge is cubric-lite per-order alpha scaling (cadence-based updates using already-scored tokens only).

Changes in this lane:
- Keep multi-order backoff at order 7.
- Keep `NGRAM_EVAL_ALPHA=0.30`, `NGRAM_EVAL_ALPHA_MIN=0.05`, `NGRAM_EVAL_ALPHA_MAX=0.60`.
- Keep adaptive entropy schedule centered at `NGRAM_EVAL_ENTROPY_CENTER=4.0` with scale `2.0`.
- Keep `NGRAM_EVAL_BUCKETS=4,194,304` (the setting used in the `.962` logs).
- Add optional `CUBRIC_CADENCE` (default `32` in run script, `0` disables) for per-order alpha multipliers.

## Safety Guardrails
- `TTT_EVAL_ENABLED=0`
- `TTT_EPOCHS=0`
- `TTT_MAX_TRAIN_CHUNKS=0`
- No oracle routing or min-NLL branch selection.
- No leaderboard-driven online adaptation in this run recipe.

## Expected Gain Band (vs plain sliding-window eval)
- Strong seeds: around `0.962` to `0.964` BPB
- Typical spread: up to about `+0.06` BPB worse when seed/config drifts (e.g., lower order profile)
- Key risk: config drift from the proven 7-gram profile, not eval throughput
- Cubric-lite expectation: neutral to small gain; disable with `CUBRIC_CADENCE=0` if it regresses on a seed.
