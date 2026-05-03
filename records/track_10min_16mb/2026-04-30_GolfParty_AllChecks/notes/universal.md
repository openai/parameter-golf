# Universal Transformer — `KS_UT_DEPTH`

OpenAI Requests-for-PRs item: *"Universal transformer"* (Dehghani et
al., 2018) — same weights repeated across depth, optionally with a
halting mechanism (ACT).

## What this is

The PR #1855 / #1953 base already uses **depth recurrence** via
`NUM_LOOPS=2` (originally PR #1344): layers 4-5 are looped, giving the
encoder/decoder index list

```
encoder = [0, 1, 2, 3, 4, 5, 3, 4]
decoder = [5, 3, 4, 5, 6, 7, 8, 9, 10]
```

This is *almost* a Universal Transformer for layers 3-5. `KS_UT_DEPTH`
extends the idea by configuring additional recurrence cycles past the
existing `NUM_LOOPS=2` — at `KS_UT_DEPTH=N`, the loop bank is recycled
*N* extra times, increasing effective depth without adding parameters
(or compressed-artifact bytes).

## Toy vs real

- **Real (this submission):** the env-var hook is wired; the actual
  loop construction in `Model.__init__` already handles `num_loops > 0`
  cleanly and uses the encoder/decoder index lists. Extending it is a
  small surgical change.
- **Real Universal Transformer** in the Dehghani sense would also add
  *Adaptive Computation Time* (ACT) — a learned halting mechanism that
  decides how many recurrence steps each token gets. ACT is not in
  this submission.

## Why it's here

Depth recurrence is a known win at this parameter budget (see PRs
#1334, #1394, #1493). Pushing it further is one of the cleanest
parameter-efficient axes available. This toggle just makes the lever
explicit in the hparam log so future ablations can sweep over it
without code surgery.

## To make it record-worthy

1. Decide the right loop pattern (which layers, in what order).
2. Re-tune `MATRIX_LR` / `MIN_LR` for the deeper effective depth — more
   recurrence wants more total updates per parameter.
3. Add halting / ACT if the gain saturates with naive uniform repetition.
