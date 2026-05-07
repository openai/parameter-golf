# Mixture of Depth (MoD): Per-Layer Token Routing

**Author:** genji0306 (Opensens Research)
**Date:** 2026-03-27
**Track:** Non-record / Exploration
**Status:** Pending H100 validation

## Hypothesis

Mixture of Depth (MoD) applies a lightweight binary router at each transformer layer, allowing "easy" tokens to skip the layer entirely while "hard" tokens pass through the full computation. Unlike Mixture of Experts (MoE), MoD adds zero parameters to the exported model — the router is a single linear layer per block. This provides adaptive compute depth per token, reducing average step time by 15-25% and yielding additional training steps within the 10-minute budget.

## Approach

Add a per-layer binary router that outputs a skip/compute decision per token. Easy tokens (confident predictions) bypass the layer via the residual stream. Hard tokens (high-entropy predictions) receive full attention and MLP computation. The router is trained jointly with a small auxiliary load-balancing loss.

## Verification Plan

- Measure step time reduction with MoD enabled vs. baseline
- Compare val_bpb at matched wall-clock time (more steps but shallower average depth)
- Validate that the router learns meaningful easy/hard token distinctions

## Related Work

- Raposo et al. (2024), "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"

## Compute Needed

Estimated 4-5 runs on 1xH100 (~60 min total).
