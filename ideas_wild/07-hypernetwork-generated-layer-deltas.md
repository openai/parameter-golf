# 7. Hypernetwork-Generated Layer Deltas

## Core Thesis

There may be a better point between fully tied blocks and fully untied blocks: a shared base block plus tiny structured per-layer deltas.

## What It Changes

Store:

- one or a few shared base blocks
- a very small hypernetwork or decoder
- compact per-layer codes that generate deltas

This gives layer diversity without storing every block densely.

## Why It Might Improve `val_bpb`

The resulting representation may compress better than independently stored dense matrices, especially after zlib, while still preserving some of the benefits of depth specialization.

## Why It Is Risky

There are two moving parts:

- the shared base
- the delta generator

If either is wrong, you get the complexity of both tying and untied depth with little gain.

## First Useful Experiment

Use an extremely small learned code per unrolled position and linearly decode it into rank-limited deltas for only the projection matrices.
