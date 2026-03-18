# 9. Pairwise Shared-Block Recurrence With Per-Layer Gates Left Untied

## Category

Architecture changes

## Why

This is the kind of parameter-golf move the challenge explicitly invites. Tie blocks in pairs or repeat a smaller bank of blocks, but keep per-layer control tensors such as `attn_scale`, `mlp_scale`, `resid_mix`, and maybe `q_gain` untied so the repeated block can behave differently at different depths.

That buys effective depth per byte.

## Tradeoffs

- Speed: unchanged if depth stays fixed, slower if the savings are cashed into more unrolls
- Size: much better byte efficiency
- Complexity/risk: moderate

## Repo Fit

The current per-layer control tensors make this much more plausible than it would be in a plain GPT.
