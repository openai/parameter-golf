# 4. Stateful Recurrent Memory

## Core Thesis

A tiny learned memory state may buy more effective computation than another fully materialized block under the same byte budget.

## What It Changes

Instead of only using residual stream accumulation, add a compact recurrent state:

- per token position
- per head
- or globally per sequence chunk

That state would be updated across repeated passes and used to influence later computation.

## Why It Might Improve `val_bpb`

Artifact bytes pay for parameters, not for transient compute state. A small learned memory mechanism could increase effective representational capacity with minimal byte overhead, especially in a recurrent/shared-weight architecture.

## Why It Is Risky

This is a genuine architecture departure. The optimization behavior is much less predictable than simpler tying or low-rank schemes.

## First Useful Experiment

Add a tiny gated memory vector per block application and reuse the same block multiple times. Keep the memory dimension very small so the artifact impact is negligible.
