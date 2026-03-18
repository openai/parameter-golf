# Recurrent MQA Transformer — Depth Recurrence + Weight Tying

**Author:** nidhilak-Aquarius  
**GitHub:** nidhilak-Aquarius  
**Status:** WIP — local implementation complete, awaiting compute grant  
**Track:** 10min / 16MB  
**Date:** 2026-03-19

---

## The Philosophy Behind the Architecture

My approach draws from two ideas separated by 2,000 years.

The **Chakravyuha** in the Mahabharata is a spiral military formation — one
repeating structural unit creating depth far beyond its apparent size. Not 12
different armies. One disciplined unit, looping inward. The power comes from
the geometry of repetition, not the addition of mass.

**Kalaripayattu**, Kerala's ancient martial art, teaches that maximum force
comes from finding the exact pressure point (marma), not from raw strength.
A Kalari master does not overpower — they apply precise energy at the exact
point where the system is most sensitive.

These are not metaphors. They are the actual engineering principles at work.

---

## Core Idea

Instead of 9 unique transformer blocks (baseline), use **one shared
TransformerBlock looped 12 times** — Universal Transformer style.

```
Baseline:    [Block_1] → [Block_2] → ... → [Block_9]   (9× unique params)
This model:  [Block]   → [Block]   → ... → [Block]      (1× unique params, 12× depth)
```

Same computational depth. 12× fewer unique parameters.

The **marma insight**: weight sharing acts as a regularizer. The same weights
must generalize across ALL depths simultaneously — forcing more robust,
invariant representations than unique per-layer weights, which are free to
overfit to their position in the stack.

This is analogous to resonance in physics: a single eigenstate representing
infinite depth without growing in mass.

---

## Architecture

| Component | Choice | Reason |
|-----------|--------|--------|
| Core structure | 1 shared block × 12 loops | 12× param savings, regularization via sharing |
| Position encoding | RoPE | Zero learned parameters (Aryabhata principle) |
| Attention | MQA: 8Q / 1KV heads | 43% fewer attention params, minimal quality loss |
| FFN | SwiGLU | Consistently outperforms GELU (Shazeer 2020) |
| Output projection | Weight-tied to embedding | Zero extra parameters |
| Normalization | RMSNorm | More stable than LayerNorm in deep recurrence |
| Optimizer | AdamW (β=0.9/0.95) | Cosine LR with 100-step warmup |

---

## Local Results (Smoke Test)

| Metric | Value |
|--------|-------|
| Unique parameters | ~3.5M |
| Compressed artifact | ~5.2MB |
| 16MB budget used | 32.5% |
| Unused budget | 10.8MB |
| val_bpb on FineWeb | **Pending GPU run** |

Smoke test confirms: clean training, decreasing loss, artifact under 5.3MB.
First real val_bpb score requires GPU — pending compute grant.

---

## Hypothesis

I hypothesize recurrence depth **N=12 outperforms N=8** at identical
parameter count, with diminishing returns beyond N=16.

This grant will map the curve empirically:
- N=8 vs N=12 vs N=16 vs N=24 at fixed parameter budget
- dim=384 vs dim=512 vs dim=768 sweeps
- LR sensitivity: 1e-3 vs 3e-3 vs 5e-3

---

## Phase 2: BitNet Ternary Quantization

The 10.8MB of unused artifact budget will fund Phase 2:

BitNet-style ternary weights constrain each weight to {-1, 0, +1}.
- float16: 16 bits per weight
- Ternary: log2(3) = **1.58 bits** per weight
- Compression ratio: 16 / 1.58 = **~10×**

Same 5.2MB artifact. Effectively 10× more expressive parameters.
Trained with straight-through estimator for gradient flow through
the non-differentiable quantization step.

This is Nagarjuna's alchemy from Kerala's Rasavidya tradition: transform
the base substance (float weights) into gold (ternary) while preserving
the essential nature (svabhava) through the training process.

---

## Why This Approach Is Promising

1. **Parameter efficiency**: 3.5M unique params behave like 42M effective
   params (12 loops × 3.5M) in terms of computational depth
2. **Artifact budget**: 5.2MB leaves 10.8MB free — more room than any
   baseline submission
3. **Regularization**: weight sharing prevents depth-specific overfitting
4. **Phase 2 headroom**: BitNet can fit 10× more in the freed space

---

## Background

- 12 years IAM systems engineering — designing minimal, efficient systems
  under hard constraints. Directly analogous to parameter budget optimization.
- Trained GANs in DeepFaceLab (encoder-decoder architecture, GPU training)
- Optimized voice ML inference pipelines (Okada) — sequential data = text
- Strong Python, familiar with PyTorch training loops and loss debugging

---

## How to Reproduce

```bash
# Clone and install
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (no GPU needed)
python3 train_gpt.py --smoke

# Single H100 (experiments)
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Full leaderboard run (8xH100)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## References

- Universal Transformers: https://arxiv.org/abs/1807.03819
- Multi-Query Attention: https://arxiv.org/abs/1911.02150
- RoPE: https://arxiv.org/abs/2104.09864
- SwiGLU: https://arxiv.org/abs/2002.05202
- BitNet: https://arxiv.org/abs/2310.11453
- modded-nanogpt (inspiration): https://github.com/KellerJordan/modded-nanogpt
