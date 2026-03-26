# [Non-Record] JEPA Self-Distillation with EMA Target Encoder + VICReg

**Track:** Non-record, unlimited compute, 16MB artifact
**Author:** Manav Pandey (MVPandey)
**val_bpb:** TBD (3 seeds, 2h each on 8xH100)
**Artifact:** ~14MB (int6 + LZMA)

## Summary

This submission applies full JEPA (Joint Embedding Predictive Architecture) with an EMA target encoder as an auxiliary self-distillation objective for autoregressive language modeling. The context encoder is a standard 20L/512d causal transformer; the target encoder is an EMA copy (decay=0.9995) that provides slowly-evolving prediction targets. A small predictor network learns to forecast what the target encoder will represent at the next token position.

My previous experience with JEPA was in constraint satisfaction (Sudoku solving via energy-based inference with Langevin dynamics, [github.com/MVPandey/Enso](https://github.com/MVPandey/Enso)). Adapting the framework from structured discrete puzzles to autoregressive token prediction required rethinking the target/context encoder relationship and discovering which components (VICReg placement, loss weighting, EMA schedule) actually matter for this domain.

Only the context encoder is saved in the 16MB artifact. The target encoder, predictor, and projection heads are purely training-time overhead with zero inference cost.

## Architecture

```
Context encoder (causal transformer, 20L/512d, trainable):
    h_ctx = transformer(x_{1..T})

Target encoder (same architecture, EMA-updated, no gradient):
    h_tgt = transformer_ema(x_{1..T})

Predictor (MLP: 256d -> 256d):
    z_pred = predictor(ctx_proj(h_ctx[t]))

Loss = CE(softmax(h_ctx @ W_emb), target)                  # standard LM loss
     + 0.3 * MSE(z_pred, tgt_proj(h_tgt[t+1]).detach())    # JEPA loss
     + VICReg(z_tgt)                                        # collapse prevention

Scoring: standard tied-embedding softmax (JEPA is training-only)
```

### Backbone
- 20 layers, 512d, 8 attention heads (GQA with 4 KV heads)
- LeakyReLU(0.5)^2 MLP (3x expansion)
- Partial RoPE (16/64 dims), QK RMSNorm, learned q_gain
- XSA on last 4 layers, U-Net skip connections
- BigramHash embeddings, logit softcapping at 30.0
- Muon optimizer for 4 weight banks, AdamW for scalars/embeddings

### JEPA Components (~393K params, not saved)
- Predictor: 2-layer MLP (256 -> GELU -> 256)
- ctx_proj: Linear(512, 256), projects context encoder output to latent space
- tgt_proj: Linear(512, 256), projects target encoder output to latent space

## The Journey

### Starting Point: Energy-Based Output Heads

The initial idea was to replace the standard softmax output with energy-based scoring in a learned latent space, inspired by the CLIP-style approach: project encoder output and token embeddings into a shared space, score via cosine similarity with a learnable temperature.

I tried several variants:
- **Cosine similarity head with separate energy embeddings** (not shared with input tok_emb). The decoupled geometry was supposed to let the output space optimize purely for discrimination. Result: identical BPB to standard softmax (2.62 vs 2.62 at step 200). With V=1024, a 512d dot product already has more than enough capacity.
- **VICReg on the energy head output** to prevent representation collapse. The L2-normalized representations had inherently low std (~0.04) on the unit sphere, making the VICReg variance hinge threshold of 1.0 meaningless. VICReg at weight=0.1 did nothing; at weight=10.0 it dominated the loss without fixing the underlying issue.
- **Deterministic sharpening Langevin** at eval time: iteratively refine predictions by moving toward squared-probability-weighted centroids. This turned out to be mean-seeking rather than mode-seeking, and with K=3 steps it just blurs predictions.

The core realization: softmax IS already an energy model with E(v) = -logit(v). Adding nonlinearity to the energy function doesn't help when you have 1024 classes and abundant data.

### Pivot: Actual JEPA with Target Encoder

The energy head experiments weren't JEPA at all; they were just different output projections. Real JEPA needs a target encoder (EMA copy) and a predictor operating in latent space, with the target providing a slowly-evolving learning signal.

Key design decisions that came from experimentation:

**VICReg placement matters.** I initially applied VICReg to the predictor output (z_pred). This is wrong. The predictor receives gradient from the MSE loss, so it doesn't collapse. The target encoder is the tensor at risk because it only receives EMA updates, never supervised gradient. When I switched VICReg to z_tgt, representation health (measured by per-dimension std) went from 0.05 (collapsed) to ~1.05 (healthy) and stayed there throughout training.

**JEPA weight sensitivity.** My first run used jepa_weight=1.0. The JEPA loss started at ~1.0 and climbed to ~2.0 during training, meaning the predictor couldn't keep up with the evolving target encoder. This actively hurt the CE loss by competing for gradient. Dropping to 0.3 and annealing to 0 during warmdown fixed this: the JEPA loss stayed stable (~1.3-1.5) and the CE loss converged better.

**Target EMA decay.** Initial decay of 0.996 was too fast; the target encoder evolved faster than the predictor could track, causing the rising JEPA loss. Increasing to 0.9995 stabilized training.

**LR warmup.** The original code had no LR ramp (only a compile warmup that resets weights). Adding 200-step linear warmup eliminated early training instability where the loss would spike from 6.9 to 14.7 before recovering.

### The Quantization Bug

My first "successful" 20L run reported 1.1483 BPB with an artifact of 39.7MB, well over the 16MB limit. The issue: the `_cls` function that classifies parameters for int6 vs int8 quantization checked for `.attn.` and `.mlp.` in tensor names, but the `_unbank` function creates names like `b.0.a.q` and `b.0.m.u`. Every bank weight fell through to int8 instead of int6, killing LZMA compression. Fixing the name matching brought the artifact from 39.7MB to ~14MB.

The silver lining: the 1.15 BPB result was achieved with int8 quantization (less lossy than int6), so the actual model quality was real. Switching to int6 adds a small quantization gap (~0.01-0.02 BPB).

## Results

| Seed | Steps | val_bpb | Artifact |
|------|-------|---------|----------|
| 42   | TBD   | TBD     | TBD      |
| 1234 | TBD   | TBD     | TBD      |
| 5678 | TBD   | TBD     | TBD      |
| **Mean** | | **TBD** | |

## What I Learned

1. **SSL auxiliary losses have limited value when CE already operates at the right abstraction level.** In vision, JEPA helps because pixel prediction is wasteful. In language with a 1024-token vocabulary, CE already predicts semantic units. The self-distillation signal from the EMA target encoder provides some regularization but doesn't fundamentally change what the model learns.

2. **The energy landscape framing doesn't buy much for discrete classification.** With V=1024 tokens, a linear dot product has sufficient capacity. Nonlinear energy functions (MLP projectors, cosine similarity, Langevin refinement) add computation without improving discrimination.

3. **JEPA weight and EMA decay are tightly coupled.** High JEPA weight + fast EMA = predictor can't track target = rising JEPA loss = gradient competition. Low weight + slow EMA = stable auxiliary signal that doesn't interfere with CE.

## Concurrent Work

PR #832 independently explores JEPA for language modeling with a byte-level transformer. The key difference is that my approach uses a full EMA target encoder as the backbone rather than chunk-level prediction on top of a standard transformer.

## Reproduction

```bash
SEED=42 ITERATIONS=200000 MAX_WALLCLOCK_SECONDS=7200 \
VAL_LOSS_EVERY=1000 WARMDOWN_ITERS=3000 \
RUN_ID=jepa_20L_final_s42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
