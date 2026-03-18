# 10. Replace Deterministic Sequential Streaming With Randomized Block Sampling

## Category

Optimization and training changes

## Why

The current loader is a single token stream that wraps. Once the dataset loops, the model sees the same prefix again in the same order.

Cheap random shard or block offsets should improve gradient diversity and reduce ordering artifacts.

## Tradeoffs

- Speed: about the same if the loader stays simple
- Size: unchanged
- Complexity/risk: low-moderate

## Repo Fit

This belongs in `TokenStream` or `DistributedTokenLoader`.
