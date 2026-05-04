# Super Long Context — `KS_LONG_CONTEXT` + `EVAL_SEQ_LEN`

OpenAI Requests-for-PRs item: *"State-space models, E2E TTT, super
long context for evaluation or training."*

## Where this already lives

PR #1953 already pushes eval seq_len from the 2048 default to **2560**
combined with `TTT_MASK=no_qv` (disable Q/V LoRA) — net gain
~−0.0006 BPB. This is the documented "super long context for
evaluation" implementation in the lineage.

## What this submission contributes

The `KS_LONG_CONTEXT=1` flag is a *documentation* toggle that surfaces
in the hparam log when `EVAL_SEQ_LEN > 2560`. The actual lever is
already there — push `EVAL_SEQ_LEN` and `TTT_EVAL_SEQ_LEN` to whatever
fits the 600s eval budget.

Empirically observed budget on our 8×H100 SXM (Hopper, FA3):
- 2560: ~362s of 600s eval budget used
- 4096: estimated ~580s (extrapolating quadratic-ish FA3 cost)
- 8192: would exceed budget without chunked / linear attention.

## To go further than 4096

Three paths, none in this submission:

1. **Chunked/linear attention at long context.** Replace FA3 quadratic
   attention with a linear-attention variant for the long tail. Runs
   in O(T) at the cost of attention quality.
2. **State-space models** (see `ssm.md`) — natively O(T), arbitrary
   context length.
3. **Sliding-window attention with bigger window**, with the prefix
   computed once and reused for many scoring positions. Requires
   careful KV-cache management.

## Why it's flagged

`KS_LONG_CONTEXT` is a hparam-log signal: it makes the long-context
contribution visible in run logs even when the value is just a
larger `EVAL_SEQ_LEN`, so reviewers don't miss the lever in among
the 30+ other env vars.
