# 3. Eval-Time Iterative Refinement

## Core Thesis

This challenge appears to leave a lot of unused evaluation-time compute on the table, and one of the cleanest ways to spend it is extra shared-weight refinement passes.

## What It Changes

After the normal forward pass, run one or more extra refinement passes through shared blocks or a tied tail stack before producing logits.

## Why It Might Improve `val_bpb`

The recorded roundtrip evaluation time is roughly `1.4` seconds, which is tiny relative to the full challenge budget. That means even materially more expensive evaluation might still be acceptable. If the extra passes reuse the same weights, the byte cost is almost zero.

This is one of the rare settings where test-time compute is not a gimmick but a likely first-order lever.

## Why It Is Risky

You may need to train with awareness of the refinement process or it may do little or even hurt. It also introduces more evaluation-specific behavior, which tends to be brittle.

## First Useful Experiment

Take a shared-weight tail or last block and apply it one extra time only at evaluation. Measure:

- exact roundtrip `val_bpb`
- total evaluation time

If it helps at all, push to more deliberate recurrent refinement.
