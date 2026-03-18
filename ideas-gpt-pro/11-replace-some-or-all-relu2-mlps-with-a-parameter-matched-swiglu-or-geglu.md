# 11. Replace Some or All ReLU2 MLPs With a Parameter-Matched SwiGLU or GEGLU

## Category

Architecture changes

## Why

Gated MLPs are usually better per parameter than plain ReLU2. The proposed approach is not to blindly inflate MLP width, but to use a parameter-matched hidden size or only switch the upper layers where semantics matter more.

## Tradeoffs

- Speed: slightly slower
- Size: neutral to slightly larger
- Complexity/risk: low-moderate

## Repo Fit

This is an easy change inside `MLP`.
