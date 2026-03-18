# 12. Add a Tiny Causal Depthwise Conv or Token-Mixing Branch

## Category

Architecture changes

## Why

FineWeb has lots of short-range regularity. A causal depthwise conv costs almost nothing in parameters and can improve local modeling without paying full attention costs.

## Tradeoffs

- Speed: slightly slower
- Size: almost unchanged
- Complexity/risk: moderate because the gain may not justify the added kernel

## Repo Fit

This still fits cleanly inside `Block`.
