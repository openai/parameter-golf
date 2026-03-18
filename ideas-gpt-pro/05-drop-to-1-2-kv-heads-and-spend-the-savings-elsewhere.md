# 5. Drop to 1-2 KV Heads and Spend the Savings Elsewhere

## Category

Architecture changes

## Why

Four KV heads is generous for an 8-head, 512-dim model. Going from `4` to `1` saves about `1.77M` weights in this baseline, which is more than three times the cost of untying the head.

The default bet here is `NUM_KV_HEADS=1`, then spending the recovered budget on depth, width, or a slightly better MLP.

## Tradeoffs

- Speed: slightly faster or neutral
- Size: smaller unless the budget is reinvested
- Complexity/risk: low

## Repo Fit

This is already parameterized.
