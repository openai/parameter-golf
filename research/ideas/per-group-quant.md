# Per-group quantization bit allocation

**Status:** candidate
**Expected Δ:** +0.001 to +0.003 (estimate; marginal)
**Source:** General idea from mixed-precision quantization literature; SOTA currently uses uniform INT6 matrix / INT8 embed.

## Idea
Current SOTA uses uniform 6 bits for all matrix weights, 8 bits for embeddings. Asymmetric allocation might help:
- More bits for layers whose weights span a wider distribution (early layers, attention QK).
- Fewer bits for layers that compress well (MLP up-projections, which have highly redundant structure).

Goal: same 16MB total budget, better quality at the same budget (or same quality with headroom to add BigramHash etc.).

## Why it might help
- Hessian-sensitive layers (where small weight changes cause big loss changes) are underserved by uniform precision.
- GPTQ's error is dominated by a few outlier layers; giving them an extra bit often wins back 0.001–0.003 bpb.
- Free to attempt — the quantization logic in `train_gpt_sota.py` already has `matrix_bits` as a scalar; extending to per-group is mechanical.

## Code-change sketch
- Replace `matrix_bits` scalar with a dict or list: `matrix_bits_per_layer` or `matrix_bits_per_group` (by matrix type).
- Re-budget: compute total bytes, ensure ≤ 16MB after quantization.
- Simplest first pass: INT7 for attention QK matrices (all layers), INT5 for MLP up-projection matrices. Everything else stays INT6.

## Risks / open questions
- Budget math: 11 layers × 2 QK matrices × 512² × 7/8 bytes ≈ extra cost. Need to verify total stays ≤ 16MB.
- GPTQ per-group calibration may behave differently per bit-width — might need separate calibration passes.
- Interacts with BigramHash (if we add it, less budget to play with).
- Already tried with Hessian-aware SDClip — user's `sota_analysis.md` notes it didn't beat plain SDClip. Need to understand whether that's a dead end or just one specific attempt.

## If this works
Important for stacking multiple features within 16MB. Without extra bits reallocated wisely, BigramHash may not fit.
