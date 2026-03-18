# 5. Mixture of Depth

## Core Thesis

Conditional compute is attractive here, but conditional experts are too byte-expensive, so conditional depth is the version worth trying.

## What It Changes

Instead of a fixed number of block applications for every token, let the model decide whether to do more recurrent passes:

- per token
- per position band
- or per sequence

The key is that all extra passes reuse existing weights.

## Why It Might Improve `val_bpb`

You get some of the benefits of adaptive compute without paying the parameter cost of MoE. Under a fixed artifact cap, that is a much better trade than adding experts.

## Why It Is Risky

Routing decisions can become noisy or unstable. If the gating logic is not very small and very disciplined, complexity grows quickly.

## First Useful Experiment

Start with a sequence-level gate:

- do either 1 or 2 extra shared passes
- use a tiny controller on the pooled hidden state

This is crude, but it tests the core idea without major complexity.
