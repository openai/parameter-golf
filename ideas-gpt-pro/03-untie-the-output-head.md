# 3. Untie the Output Head

## Category

Architecture changes

## Why

In this exact script, tying is buying very little. The vocab is only `1024`, so untying adds just `1024 x 512 = 524,288` weights, which is small relative to the transformer blocks.

Separate input and output embeddings usually help more than that costs.

## Tradeoffs

- Speed: basically unchanged
- Size: slightly larger
- Complexity/risk: trivial

## Repo Fit

This is already implemented, so it is an immediate toggle plus maybe a head-LR sweep.
