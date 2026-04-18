# Disable Layer 0 Attention

**Status:** candidate
**Expected Δ:** +0.001 to +0.003 (unverified)
**Source:** 2026-03-31 submission (1.1063 bpb) used this.

## Idea
The first transformer layer's attention is often dominated by noise — tokens have fresh embeddings, no useful context to attend to yet. Skipping attention in layer 0 (keep only the MLP) frees a small amount of compute and parameter capacity for elsewhere.

## Why it might help
- Empirically observed in some architectures that layer-0 attention entropy is near-uniform (= uninformative).
- Frees ~50K–100K params (depending on attn projections) to reallocate — either into a wider MLP, an extra layer, or leave for quantization headroom.
- Zero runtime cost; likely a small win.

## Code-change sketch
- In `train_gpt_sota.py`'s block class, add a config flag `disable_layer0_attn`.
- If set, layer 0's forward pass becomes `x = x + MLP(norm(x))` without the attention term.
- Don't allocate attention projections for layer 0 when flag is set (to actually reclaim params).

## Risks / open questions
- The SOTA submission from 2026-04-09 does NOT use this. Why? Was it tried and rejected, or just not tried at that point? Need to check `sota_analysis.md` or ask.
- Reclaimed params have to go somewhere useful for the Δ to materialize. Options: slightly wider MLP in layer 0, or added depth elsewhere.
- Interacts with depth recurrence — if layer 0 has no attention, the recurrence schedule (which involves layer 3+) is unaffected.

## If this works
Could stack with progressive-recurrence and swa-plus-ema.
