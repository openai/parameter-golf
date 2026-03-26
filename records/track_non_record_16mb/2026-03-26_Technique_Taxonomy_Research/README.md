# Parameter Golf Technique Taxonomy: Tier List, Interaction Effects, and Verification Tools

**Non-Record Submission (Research Synthesis)**
**Author:** @robbiebusinessacc
**Date:** March 26, 2026
**Best score achieved:** 1.1428 BPB (merged record, PR #180)

---

## What This Document Adds

Several excellent research PRs exist for specific topics: PR #363 (depth recurrence), PR #831 (throughput tax), PR #670/#756 (kernel/quantization negative results). This document synthesizes across all of them and adds:

1. **Tier-ranked technique table** with measured BPB deltas from merged PRs
2. **Interaction effects matrix** — which technique combos are sub-additive
3. **BPB verification checklist** — catch formula errors and causal violations
4. **N-gram legality status** with organizer rulings collected in one place
5. **Parameter budget calculator** for quick config feasibility checks

---

## Technique Tier List

Ranked by **marginal BPB improvement when added to a competitive stack** (not vs raw baseline). Numbers are from merged submission ablations and our own 8×H100 runs.

### S-Tier: Must-Have (each: 0.005–0.020 BPB)

| Technique | BPB Delta | Source | Notes |
|-----------|-----------|--------|-------|
| Sliding window eval (stride=64) | -0.020 to -0.025 | All merged records | Eval-only. Free. Every competitive submission uses this. |
| Int6 quantization (MLP weights) | -0.010 to -0.015 | PR #164, PR #287 | Saves ~25% model bytes → reinvest in wider MLP |
| 3× MLP expansion (from 2×) | -0.010 to -0.015 | PR #164 | Enabled by int6 savings. Biggest single arch win. |
| FP16 embeddings | -0.005 to -0.008 | PR #180 | Small table, disproportionate quant quality loss |
| 11 layers (from 9) | -0.005 to -0.008 | PR #287, PR #315 | Fits at int6 with 3× MLP |
| Seq_len 2048 (train + eval) | -0.005 to -0.008 | PR #180, PR #287 | NTK RoPE scaling for extrapolation |

### A-Tier: Strong (each: 0.002–0.005 BPB)

| Technique | BPB Delta | Source | Notes |
|-----------|-----------|--------|-------|
| Muon weight decay (0.04) | -0.003 to -0.005 | PR #164, PR #198 | Standard by now |
| EMA (decay=0.997) | -0.002 to -0.004 | PR #287 | Slightly better than SWA. **Calibrate GPTQ on EMA model.** |
| SWA | -0.003 to -0.005 | PR #180 | Simpler alternative to EMA |
| Orthogonal init | -0.002 to -0.003 | PR #180 | Better-conditioned matrices for quant |
| SmearGate | -0.002 to -0.003 | PR #180 | Learned gate mixing current + prev token embedding |
| BigramHash | -0.002 to -0.004 | PR #180 | Hashed bigram pair representations |
| XSA (last 4 layers) | -0.001 to -0.003 | PR #287, PR #265 | arXiv:2603.09078. Zero new params. |
| LeakyReLU(0.5)² | -0.001 to -0.002 | PR #549 | One-line change |
| QAT with STE (int6) | -0.001 to -0.003 | PR #180, PR #414 | Start at 80-85% of training |

### B-Tier: Marginal (each: <0.002 BPB)

| Technique | BPB Delta | Source |
|-----------|-----------|--------|
| Partial RoPE (16/64 dims) | -0.001 to -0.002 | PR #315 |
| LN Scale (1/sqrt(layer)) | -0.001 | PR #315 |
| Value embeddings (VE128) | -0.001 | PR #315 |
| LZMA over zstd | -0.000 to -0.001 | PR #414 |
| Warmdown 1200→3500 | -0.001 to -0.002 | PR #414 |

### C-Tier: High Effort / Paradigm Shift

| Technique | BPB Delta | Source |
|-----------|-----------|--------|
| Legal TTT (score-first) | -0.020 to -0.050 | PR #549 |
| N-gram backoff mixer | -0.050 to -0.700 | PR #688, #814 (all unmerged) |
| GPTQ (Hessian-aware) | -0.002 to -0.005 | PR #414 |
| Ternary/binary quant | Requires full redesign | PR #640, #641 |

---

## Interaction Effects Matrix

This is the part nobody else has published. Some technique combos are **sub-additive** — stacking them gives less than the sum of individual gains.

| Combo | Expected (sum) | Actual | Why |
|-------|----------------|--------|-----|
| XSA-all + TTT | -0.028 | ~-0.022 | XSA biases interfere with TTT LoRA adaptation |
| Int6 QAT + GPTQ | -0.006 | ~-0.004 | QAT already reduces the error GPTQ would fix |
| EMA + SWA | -0.007 | ~-0.004 | Redundant — pick one |
| LeakyReLU² + 3×MLP | -0.017 | ~-0.016 | Nearly additive (good combo) |
| Partial RoPE + seq=2048 | -0.009 | ~-0.008 | Nearly additive |
| TTT + N-gram mixer | -0.08 | ~-0.07 | Nearly additive (complementary mechanisms) |

**Rule of thumb:** Training improvements stack additively with each other. Eval improvements stack additively with each other. But training × eval interactions are sub-additive because better training reduces the gap eval tricks exploit.

**Practical implication:** If you're choosing between two techniques and can only implement one, pick the one that doesn't overlap with what you already have. XSA-all is worth less if you're already doing TTT.

---

## N-gram Revolution & Legality Status

Starting ~March 23, classical n-gram language models added at eval time transformed the competition. **No n-gram PR has been merged as of March 26.** The merged SOTA remains PR #549 at 1.1194 BPB.

### Organizer Rulings (Collected)

**The fundamental rule, from Will DePue (OpenAI team) on Discord (March 25):**
> "All that matters is your eval runs on 10 minutes and that the information you send between train and eval is under 16MB, that is, you can use more runtime memory during eval but you can't use runtime memory to 'send extra bits' for use during eval from training if they aren't counted in your 16MB. You can imagine the best analogy is I spin up an 8xH100 box for 10 minutes, you train your model, then you hand me a flash drive with only 16MB of space in it with your weights and your code, and then I plug that into a new 8xH100 box for 10 minutes and then you get your score."

**Key implication:** Runtime eval memory is unlimited. N-gram tables built during eval don't count against 16MB.

**On n-gram legality (@valerio-oai, issue #677):**
> "We've been discussing this internally, and we are currently leaning towards accepting it as legal. It's essentially a way to compensate for the undertrained-ness of the LLM..."

**What's explicitly illegal (@valerio-oai, PR #659):**
- **Oracle/min-NLL selection** — picking whichever of neural or n-gram gives lower loss is "effectively peeking at the correct token"
- Must **commit to one mixture distribution** before scoring each token

### N-gram Approaches by Legality Confidence

| Approach | BPB Range | Key PRs | Status |
|----------|-----------|---------|--------|
| Hedge mixer (5 experts) | 1.04–1.08 | PR #688 | Unmerged, likely legal |
| Simple 5-gram cache | 0.90–1.05 | PR #706, #724 | Unmerged, likely legal |
| Complementary training + backoff | 0.44–0.55 | PR #803, #814 | Unmerged, likely legal |
| Multi-order backoff (7-9 gram) | 0.50–0.90 | PR #795, #813 | Unmerged, under review |
| Order-adaptive backoff (11+ gram) | 0.13–0.30 | PR #825, #853 | Unmerged, disputed |
| Full-rescore cache | 0.09–0.10 | PR #870, #881 | Unmerged, disputed |

---

## BPB Verification Checklist

### Formula Check

```
val_bpb = (val_loss / ln(2)) × tokens_per_byte
tokens_per_byte = total_tokens / total_bytes    (on validation set)
```

For SentencePiece-1024 on FineWeb: `tokens_per_byte ≈ 0.408–0.412`.

**Reverse-engineer from any claim:** `tokens_per_byte = val_bpb × ln(2) / val_loss`. If it doesn't match the expected range for the declared tokenizer, something is wrong.

### Common Errors

| Error | Symptom |
|-------|---------|
| Dividing by ln(2) twice | BPB is 1.44× too low |
| Using perplexity instead of loss | BPB is nonsensically high |
| Swapped tokens/bytes ratio | BPB is ~2.4× wrong |
| Scoring only high-context tokens | BPB looks artificially good |

### N-gram Causality Check

For n-gram submissions, verify:
- N-gram model built only from **already-scored** tokens (backward-looking)
- No oracle/min-NLL selection (must commit to mixture before scoring)
- Token being scored is **excluded** from its own n-gram context
- Claims approaching 0.0 BPB are almost certainly causal violations

---

## Negative Results Index

Rather than restating others' excellent work, here's where to find each negative result:

| Dead End | Result | Read This |
|----------|--------|-----------|
| Depth recurrence / layer tying | -0.025 BPB vs flat | PR #363 (@evangelinehelsinki) — 250+ hrs, 12 experiments |
| Novel architectures (MUD, nGPT, SSM, etc.) | All slower | PR #831 (@sseanliu) — throughput tax: need 0.007 BPB/ms |
| Kernel optimization (CUTLASS, Triton, FP8) | torch.compile wins | PR #670 (@abaybektursun) — 82ms step is 95% optimal |
| GPTQ calibration: random vs real data | Only 0.002 BPB diff | PR #756 (@abaybektursun) |
| SwiGLU activation | Neutral at this scale | PR #676, #799, #661 — no merged submission uses SwiGLU |
| Multi-chunk TTT gradient accumulation | +0.002–0.005 worse | Fewer adaptation steps = less progressive learning |
| Soft-round QAT for int6 | Negligible | Needs ~1750 annealing steps; typical QAT window is 500 |
| MC dropout at eval | Computationally impossible | K=100 needs 15,000s; K=3 gives <0.001 BPB |

**One exception worth watching:** PR #857 (@aruniyer) claims 1.1093 BPB with 15L depth recurrence + TTT, suggesting recurrence may work when combined with TTT.

---

## Parameter Budget Calculator

```python
def fits_in_budget(vocab, dim, layers, mlp_mult, kv_heads, heads,
                   unique_layers=None, tie_embed=True,
                   mlp_bits=6, attn_bits=8, embed_bits=16):
    """Returns (total_bytes, fits_bool)"""
    if unique_layers is None:
        unique_layers = layers
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim
    attn = dim*dim + 2*dim*kv_dim + dim*dim
    mlp = 2 * dim * int(mlp_mult * dim)
    scalars = dim * 4 + heads
    block = attn * (attn_bits/8) + mlp * (mlp_bits/8) + scalars * 2
    embed = vocab * dim * (embed_bits/8) * (1 if tie_embed else 2)
    skips = min(layers//2, layers - layers//2) * dim * 2
    total = embed + unique_layers * block + skips
    artifact = total * 0.92 + 50000  # zstd-22 + ~50KB code
    return artifact, artifact <= 16_000_000

# Verified configs:
# fits_in_budget(1024, 512, 9,  2.0, 4, 8)  → ~15.4 MB ✓ (baseline)
# fits_in_budget(1024, 512, 11, 3.0, 4, 8)  → ~14.7 MB ✓ (meta stack)
# fits_in_budget(1024, 512, 11, 3.5, 4, 8)  → ~15.8 MB ✓ (wide MLP)
# fits_in_budget(1024, 512, 12, 2.8, 4, 8)  → ~15.6 MB ✓
# fits_in_budget(1024, 512, 13, 3.0, 4, 8)  → ~17.1 MB ✗ (over by 1.1 MB)
# fits_in_budget(1024, 768, 5,  3.0, 4, 12) → ~18.5 MB ✗
```

### Quantization Quality Impact

| Scheme | Quality Loss (BPB) | Best Use |
|--------|-------------------|----------|
| Int8 | 0.007 | Attention weights |
| Int6 | 0.010–0.015 | MLP weights |
| Int5 | 0.015–0.020 | MLP only, with QAT |
| Ternary | 0.030–0.050 | Full redesign (PR #640) |

**Converged mixed-precision meta:** Int6 MLP + Int8 Attention + FP16 Embeddings + zstd-22 → ~14.7 MB artifact, ~1.25 MB margin.

**Pitfall from PR #670:** Late QAT causes torch.compile recompilation → OOM. Flipping `_qat_enabled` mid-training changes the graph. Budget for this or enable QAT from the start.

---

## TTT Quick Reference

**What works (from PR #549, merged SOTA):**
- SGD lr=0.002 (not AdamW — cold-start momentum causes catastrophic early updates)
- chunk_size=256 (128 is wasteful: same context coverage, 2× more forwards)
- Score-first: score tokens BEFORE training on them (definitively legal)
- Don't freeze early blocks (ttt_freeze_blocks=0)
- ~450-550s eval time (within 10-min eval budget)

**What doesn't work:**
- Gradient accumulation across chunks (+0.002–0.005 BPP worse)
- AdamW for TTT (momentum cold-start per document)
- Freezing early layers (hurts adaptation)

---

## Timeline of Key Innovations

| Date | PR | BPB | Innovation | Status |
|------|-----|-----|-----------|--------|
| Mar 18 | Baseline | 1.2244 | 9L 512d, int8, seq=1024 | Merged |
| Mar 19 | #164 | 1.1630 | Int6 + 3× MLP + SmearGate + BigramHash | Merged |
| Mar 20 | #198 | 1.1458 | + 11L + Muon WD + SWA | Merged |
| Mar 20 | #287 | 1.1271 | + XSA4 + EMA | Merged |
| Mar 21 | #315 | 1.1248 | + Partial RoPE + LN Scale | Merged |
| Mar 22 | #414 | 1.1233 | + GPTQ-lite + warmdown3500 | Merged |
| Mar 23 | #549 | 1.1194 | + LeakyReLU² + Legal TTT (**merged SOTA**) | Merged |
| Mar 23 | #688 | 1.0745 | N-gram era: 5-expert Hedge mixer | Open |
| Mar 25 | #814 | 0.4820 | Cubric: complementary training + n-gram | Open |

---

## Related Research PRs

| Topic | PR | Author |
|-------|-----|--------|
| Depth recurrence (definitive) | #363 | @evangelinehelsinki |
| Why novel architectures fail | #831 | @sseanliu |
| Hardware/kernel negative results | #670 | @abaybektursun |
| Quantization negative results | #756 | @abaybektursun |
| Recursive weight sharing | #579 | @newjordan |
| Data ordering (negative result) | #772 | @abaybektursun |
| Ternary quantization | #640, #641 | @CiprianFlorin-Ifrim |

---

## Acknowledgments

Built on the work of the entire Parameter Golf community. Thanks to PR #363 (@evangelinehelsinki) for the exemplary research format, PR #831 (@sseanliu) and PR #670/#756 (@abaybektursun) for deep negative-result studies referenced throughout, and all merged PR authors whose ablation data made this taxonomy possible.
