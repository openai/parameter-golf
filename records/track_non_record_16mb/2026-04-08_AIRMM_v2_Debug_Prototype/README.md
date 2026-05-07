# AIR-MM v2 Debug Prototype — Experimental Non-Record Submission

**Status:** Experimental prototype. Not a leaderboard claim.

**Author:** John McCray (@Jmccrayiii)

**Date:** 2026-04-08

---

## What is AIR-MM?

**Adaptive Importance Routing Memory Model (AIR-MM)** is a novel selective-compute mechanism for parameter-constrained transformers. Instead of treating every token equally — giving each position the same attention computation, MLP pass, and residual update — AIR-MM estimates which tokens carry more information and selectively scales their residual updates accordingly.

The core insight: under tight parameter and compute budgets, most tokens in a sequence are predictable filler. A handful carry the real signal. AIR-MM introduces a lightweight mechanism that routes more update strength to the tokens that matter and softens updates for the rest.

This is **not** pruning, distillation, or mixture-of-experts. Every token is still processed. The question is how much weight each token's update carries.

## Motivation

The Parameter Golf challenge imposes hard size (16 MB artifact) and training-time (10 min wallclock) constraints. The standard approach is to shrink a known transformer architecture and hope the pieces fit. AIR-MM takes a different direction: rather than just making the model smaller, give it a new degree of freedom — the ability to decide, per token and per layer, how strongly to apply its own updates.

The goal is to maximize intelligence per byte and per second by moving from **uniform compute allocation** to **importance-based compute allocation**.

## Architecture

### v1: Learned Token Importance Gating

A small two-layer MLP per transformer block scores each token's importance from its hidden representation:

```
hidden_state → Linear(dim, 32) → ReLU → Linear(32, 1) → Sigmoid
```

The output is a scalar in (0, 1), mapped to `[min_scale, 1.0]` to ensure no token is ever fully silenced. This gate is applied multiplicatively to both attention and MLP residual updates:

```
importance = min_scale + (1 - min_scale) * sigmoid(gate(x))
x = x + importance * attn_out
x = x + importance * mlp_out
```

Parameter overhead: ~800 parameters per block. Negligible relative to the millions of parameters in attention and MLP weights.

### v2: Learned Gate + Recency Signal + Learned Combiner

v2 adds a second signal: **positional recency**. This is a normalized position ramp `[0, 1]` across the sequence — earlier tokens get lower values, later tokens get higher values. The idea is that more recent tokens are more likely to carry actionable information.

A tiny learned combiner (`Linear(2, 1)` — 3 parameters per block) blends the learned importance signal and the recency signal:

```
combined = sigmoid(combiner([learned, recency]))
importance = min_scale + (1 - min_scale) * combined
```

Total additional parameters for v2 over v1: 12 (3 per block × 4 blocks in debug config).

### Baseline Preservation

When `USE_IMPORTANCE_ROUTING=0`, the gate is not created and the forward pass is identical to the original baseline. This makes ablation a one-line change.

## Local FAST_DEBUG Results

All results below are from a **CPU-only fast-debug configuration** (128-dim, 4-layer model, 10 training steps, 8192 batch tokens, seq_len=128). These are **not** representative of full-scale GPU performance and should not be compared to leaderboard scores.

| Configuration | val_bpb (step 10) | train_loss (step 10) | Parameters | Runtime |
|---|---|---|---|---|
| AIR-MM v1 (learned gate, min_scale=0.25) | 4.0261 | 6.8806 | 608,656 | ~7s |
| AIR-MM v2 (learned + recency, min_scale=0.10) | 4.0254 | 6.8788 | 608,668 | ~7s |

**Key observations:**
- Training is stable with no loss divergence or spikes
- Baseline behavior is fully preserved when routing is disabled
- v2 shows a small directional improvement over v1
- The importance gate learns non-trivial per-token distributions (learned signal ~0.46, combined ~0.62, recency spans full [0, 1] range)
- Gate behavior is relatively uniform across blocks at this scale and iteration count

**Honest assessment:** These results are promising but early. The mechanism is active and stable, but the small debug model and 10-step training horizon are insufficient to demonstrate meaningful separation from baseline. The real test requires GPU-scale runs with the full model configuration.

## What This Is and Is Not

**This is:**
- A serious experimental direction exploring selective compute allocation
- A working prototype with clean ablation controls
- A demonstration of training stability and mechanism activity
- An honest non-record submission for community visibility

**This is not:**
- A leaderboard claim
- Evidence of SOTA performance
- A finalized architecture

## Configuration

Key hyperparameters (v2 debug):
- `USE_IMPORTANCE_ROUTING = 1`
- `IMPORTANCE_HIDDEN_DIM = 32`
- `IMPORTANCE_MIN_SCALE = 0.10`
- `USE_RECENCY_SIGNAL = 1`
- `IMPORTANCE_COMBINE_MODE = "learned"`
- `RECENCY_STRENGTH = 1.0`
- `IMPORTANCE_LOG_EVERY = 1`

## Included Files

- `README.md` — This file
- `submission.json` — Submission metadata
- `requirements.txt` — Dependencies (standard PyTorch stack only)
- `train_gpt.py` — Code snapshot with AIR-MM v2 implementation
- `train_debug_v2.log` — Training log from AIR-MM v2 FAST_DEBUG run (CPU, 10 steps)
