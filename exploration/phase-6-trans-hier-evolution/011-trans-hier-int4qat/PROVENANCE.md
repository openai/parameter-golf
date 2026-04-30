# Provenance: What's borrowed, what's ours

## From the competition baseline (train_gpt.py)
These are standard components that every submission uses:
- Muon optimizer (Newton-Schulz orthogonalization)
- Adam for scalar/embedding parameters
- GQA attention (8 heads, 4 KV heads)
- RoPE positional embeddings
- Parallel block schedule (attn + MLP from shared residual)
- Encoder/decoder split with skip connections
- Logit softcap (tanh, cap=30)
- BPB evaluation with SentencePiece byte counting
- Distributed training with torchrun

## From the leaderboard (known PRs and submissions)
Borrowed techniques that appeared in prior submissions:
- SmearGate (token blending with previous position) — from 2026-03-20
- BigramHashEmbedding (hash consecutive token pairs) — from 2026-03-20
- SwiGLU MLP activation — from 2026-03-20
- SWA (Stochastic Weight Averaging) during warmdown — from 2026-03-20
- Muon weight decay — from 2026-03-20

## NOVEL to this submission

### 1. Hierarchical Macro Sidechannel with Causal Self-Distillation
The core architectural innovation. No prior submission has this.

**Concept**: At the encoder/decoder boundary, extract interval-level summaries (last
token of every 16-token block). Run a dual-pass processing:
- **Teacher** (non-causal oracle): enriches current-interval summaries via a learned
  prediction head. Runs under no_grad — never broadcast to tokens.
- **Student** (causal): enriches PREVIOUS-interval summaries via the SAME prediction
  head, with gradients. This is what tokens attend to.

Tokens cross-attend to past student summaries via learned Q/K/V projections with
causal masking. A per-dimension sigmoid gate controls injection strength.

**Cosine distillation loss**: The student at interval c is trained to match the
teacher at interval c (same position, different inputs). The gradient teaches the
prediction head to anticipate what the current interval's representation looks like
from past context alone. The model learns to shortcut its own future self.

At inference, the teacher path is dropped. The student has already internalized the
anticipatory pattern.

### 2. Int4 QAT (Quantization-Aware Training)
All linear projections use fake-quantized 4-bit weights during training (STE for
backward). Per-group scales with group_size=128. The model learns to compensate for
quantization noise from step 1, so the int4 roundtrip at submission has minimal
degradation.

This replaces the int5/int6/int8 post-training quantization used by the leaderboard
baseline (which produces ~22MB artifacts — over the 16MB budget for this model size).

### 3. Dense-to-Sparse Monarch Factorization (d2s)
Optional: MLP expansion layers can be factored into block-diagonal Monarch structure
mid-training via SVD, with learning rate healing. Reduces parameter count while
preserving capacity. Currently disabled (d2s_enabled=False) pending validation.

## Architecture summary
```
Input → tok_emb (int4 QAT) + bigram + smear + RMSNorm
  → Encoder (5 parallel blocks, GQA + SwiGLU, int4 QAT)
  → Macro Sidechannel (dual-pass self-distillation, causal cross-attention)
  → Decoder (5 parallel blocks + skip connections, int4 QAT)
  → RMSNorm → logit softcap → loss + distillation loss
```

## What makes this distinct from every leaderboard entry
1. **Hierarchical temporal processing** — nobody else has a Macro pathway operating
   on interval summaries
2. **In-flight self-distillation** — the model trains against its own non-causal
   oracle. No external teacher, no EMA, no separate training phase.
3. **Cross-attention routing** — tokens selectively attend to past interval summaries
   weighted by learned relevance, not hard-broadcast
4. **Int4 QAT** on a transformer (most submissions use int6+ post-training)
