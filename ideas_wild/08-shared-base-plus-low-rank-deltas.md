# 8. Shared Base Plus Low-Rank Deltas

## Core Thesis

If the hypernetwork version is too ambitious, a simpler variant is a heavily shared base model with tiny low-rank per-position adapters.

## What It Changes

Most weights are shared. Each unrolled position only gets:

- a small low-rank delta
- or a tiny adapter-like correction

This is a much simpler way to preserve some layer diversity.

## Why It Might Improve `val_bpb`

It combines:

- the byte efficiency of tied weights
- some of the representational flexibility of untied weights

This feels especially compatible with the current baseline, because the present model already has several scalar/vector control parameters that suggest lightweight per-position modulation is acceptable.

## Why It Is Risky

The low-rank deltas may be too weak if the shared base is not already good. But the failure mode is cleaner than with a full hypernetwork.

## First Useful Experiment

Tie blocks aggressively, then add rank-4 or rank-8 deltas only on the output projections and compare against a pure tied-block baseline.
