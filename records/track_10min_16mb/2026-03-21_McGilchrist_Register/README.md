# McGilchrist Register Token — Negative Result

**Status:** Complete. Non-record submission (artifact 17.1MB > 16MB; register overhead costs too many training steps).

## Results

| Metric | Value |
|--------|-------|
| val_bpb (int8+zlib) | **1.16166728** |
| Artifact size (int8+zlib) | 17,132,420 bytes (over 16MB limit) |
| Training steps | 4,259 (~13% fewer than SOTA due to overhead) |
| Step time | ~141ms (vs ~122ms for SOTA) |
| Peak VRAM | 23.7 GB |
| Eval time | 233s (~3.9 min) |

## Method

Each transformer block gets a single learnable **register token** — a global bottleneck that attends over all sequence tokens, synthesizes a summary, then injects it back via cross-attention. Inspired by McGilchrist's theory that the right hemisphere holds holistic/contextual understanding while the left processes local detail: the register token acts as the "right hemisphere," observing the whole document before the local attention proceeds.

## Why It Didn't Win

The register mechanism is sample-efficient: at equal step counts it beats the baseline by ~0.006 bpb (step 2000: 1.2555 vs 1.2620). But the 19ms/step overhead reduces training from ~4917 steps to 4259 steps in the fixed 10-minute window. The lost training time outweighs the per-step gain.

Additionally the artifact exceeds the 16MB limit — the register parameters inflate the model.

## What Would Fix It

1. **bf16 cumsum** (already implemented locally) — cuts step time from 141ms to ~127ms, recovering ~465 steps
2. **Reduce model size** — trim layers or MLP hidden to bring artifact under 16MB
3. **Larger stride in eval** — eval ran at stride=128/batch=32 (3.9 min); with batch=256 + stride=128 this drops to ~15s

## Inspiration

> "Some things are destroyed by direct gaze. The quality in a piece of writing that vanishes when you try to name it."
> — McGilchrist, *The Master and His Emissary*

The register token is peripheral attention: it holds what the model senses but cannot yet say.
