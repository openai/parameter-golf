# Parameter Golf Technique Taxonomy: Tier List, Interaction Effects, and Verification Tools

**Non-Record Submission (Research Synthesis)**
**Author:** @robbiebusinessacc
**Date:** March 26, 2026

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
| CROWN-Q + Full GPTQ | -0.004 to -0.008 | PR #693 |
| Ternary/binary quant | Requires full redesign | PR #640, #641 |

---

## Interaction Effects Matrix

Some technique combos are **sub-additive** — stacking them gives less than the sum of individual gains.

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

## N-gram Caching: Status & Validity Crisis

Starting ~March 23, eval-time n-gram caches appeared to transform the competition, with claimed BPB scores dropping from 1.12 to 0.09. **No n-gram PR has been merged.** The merged SOTA remains PR #549 at 1.1194 BPB.

**As of March 27, most n-gram scores have been shown to be invalid** (see [Validity Crisis](#n-gram-validity-crisis) below).

### Organizer Rulings (Collected)

**The fundamental rule, from Will DePue (OpenAI team) on Discord (March 25):**
> "All that matters is your eval runs on 10 minutes and that the information you send between train and eval is under 16MB, that is, you can use more runtime memory during eval but you can't use runtime memory to 'send extra bits' for use during eval from training if they aren't counted in your 16MB. You can imagine the best analogy is I spin up an 8xH100 box for 10 minutes, you train your model, then you hand me a flash drive with only 16MB of space in it with your weights and your code, and then I plug that into a new 8xH100 box for 10 minutes and then you get your score."

**Key implication:** Runtime eval memory is unlimited. N-gram tables built during eval don't count against 16MB.

**On n-gram concept (@valerio-oai, issue #677):**
> "We've been discussing this internally, and we are currently leaning towards accepting it as legal. It's essentially a way to compensate for the undertrained-ness of the LLM..."

**What's explicitly illegal (@valerio-oai, PR #659):**
- **Oracle/min-NLL selection** — picking whichever of neural or n-gram gives lower loss is "effectively peeking at the correct token"
- Must **commit to one mixture distribution** before scoring each token

**Organizer response to validity crisis (@valerio-oai, issue #677, March 27):**
> "I agree there are likely issues with the current implementations of EvalCaches, especially regarding hashing and renormalization -- we've been investigating, and that is a large part of the reason why we haven't merged anything for the past couple of days."

### N-gram Validity Crisis

PR #886 (@abaybektursun) and analysis by @Eppie (issue #677) demonstrated that **most n-gram cache implementations produce invalid probability distributions:**

**The core problem:** Most implementations only compute the blended probability for the **correct token**. The other 1,023 tokens are never scored. If you scored all of them, the distribution sums to ~410, not 1.0. The reported BPB is not a valid information-theoretic measurement.

**The hash collision proof:** The n-gram "improvement" tracks hash collision density, not prediction quality:
- 1 hash bucket: `P(cache_bin) = T/T = 1.0` for every lookup → BPB approaches 0 with α=1
- 256M buckets (near collision-free): scores 1.11, **same as neural-only baseline**
- The gap between 1 and 256M buckets comes entirely from collision aggregation, not linguistic signal

**Two-pass rescoring is also invalid:** PRs #846, #853, #868, #870, #881, #888 violate causality — pass 2 rescores token #100 using a cache built from tokens #101 through #62M.

**The decodability test (@Eppie):** A valid BPB claim implies you can compress the validation set to that size. If you claim 0.1 BPB on 151M bytes, you're claiming ~1.9 MB of compressed data. If you cannot produce a decoder that reconstructs the original data from an artifact of that size, the score is invalid.

**What might still be valid:** Implementations using proper Dirichlet-Multinomial posterior predictive mixing (PR #900, @Robby955) produce normalized distributions by construction. However, even these rely on hash tables that introduce collision-based distortion — the score degrades toward baseline as collisions are removed.

**Proposed fixes (from issue #677):**
1. Verify the blended distribution sums to 1 over all vocab tokens — one `torch.sum` per position
2. Make causality an explicit rule
3. Cap auxiliary eval-time state (≤32 MB proposed)
4. Cap per-token overhead (≤1.5× base forward pass proposed)

### FineWeb N-gram Repetition Rates

Actual FineWeb validation set n-gram statistics — this data shows why n-gram caching was expected to be effective:

| Order | Unique N-grams | % Positions Repeated |
|-------|---------------|---------------------|
| 2-gram | 294K | 99.5% |
| 3-gram | 5.9M | 90.5% |
| 4-gram | 21.1M | 66.1% |
| 5-gram | 50.8M | 18.1% |
| 6-gram | 56.9M | 8.3% |
| 7-gram | 59.6M | 3.9% |
| 8-gram | 60.8M | 2.0% |
| 9-gram | 61.3M | 1.1% |

### N-gram Approaches — Current Status

| Approach | Claimed BPB | Key PRs | Status |
|----------|------------|---------|--------|
| Hedge mixer (5 experts) | 1.04–1.08 | PR #688 | Unmerged, likely valid distribution |
| Dirichlet-Multinomial mixing | ~0.32 | PR #900 | Unmerged, valid by construction but hash collisions inflate score |
| Complementary training + backoff | 0.44–0.55 | PR #803, #814 | Unmerged, distribution validity unknown |
| Multi-order backoff (7-9 gram) | 0.50–0.90 | PR #795, #813 | Unmerged, likely invalid distribution |
| Order-adaptive backoff (11+ gram) | 0.13–0.30 | PR #825, #853 | Unmerged, invalid distribution + causality violations |
| Full-rescore cache | 0.09–0.10 | PR #870, #881 | Unmerged, invalid distribution + causality violations |

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

### N-gram Validity Check

For n-gram submissions, verify **distribution validity first, then causality:**
- **Distribution sums to 1:** The blended distribution must be computed over ALL vocab tokens, not just the correct one. Sum the blend over all 1,024 tokens — if it's not ~1.0, the BPB score is meaningless (PR #886)
- **Hash collision test:** Run the same code with 256M buckets (near collision-free). If the score jumps back to baseline (~1.11), the "improvement" was collision noise, not prediction quality
- **Decodability:** A valid 0.1 BPB claim implies ~1.9 MB compressed representation of 151M bytes. Can you actually decode it? (@Eppie)
- **Causality:** N-gram model built only from already-scored tokens. No two-pass rescoring with future tokens.
- **No oracle selection:** Must commit to one mixture distribution before scoring each token

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

**CROWN-Q (PR #693):** Training-time curvature-weighted penalty applied during warmdown: `lambda * mean(w²) * delta² / 12`. Pushes weights into flat minima where int6 quantization causes less damage. Combined with full Cholesky GPTQ (act-order), achieves 1.1186 BPB **without TTT** — comparable to TTT-based submissions. Zero eval-time cost. Notable finding: AdamW TTT destroys GPTQ-quantized weights (+0.077 BPB degradation), so CROWN-Q + GPTQ is best used without TTT.

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
| Mar 27 | #886 | — | N-gram validity crisis: most scores shown invalid | Open |

**Key moments:**
1. **Mar 18–23:** Neural optimization (1.2244 → 1.1194, all merged)
2. **Mar 23–26:** N-gram era (claimed 1.07 → 0.09, none merged)
3. **Mar 27:** PR #886 + @Eppie show most n-gram scores are invalid distributions. Organizers investigating.

---

## Related Research PRs

| Topic | PR | Author |
|-------|-----|--------|
| Depth recurrence (definitive) | #363 | @evangelinehelsinki |
| Why novel architectures fail | #831 | @sseanliu |
| Hardware/kernel negative results | #670 | @abaybektursun |
| Quantization negative results | #756 | @abaybektursun |
| Recursive weight sharing | #579 | @newjordan |
| N-gram validity analysis | #886 | @abaybektursun |
| Data ordering (negative result) | #772 | @abaybektursun |
| Ternary quantization | #640, #641 | @CiprianFlorin-Ifrim |

---

## Acknowledgments

Built on the work of the entire Parameter Golf community. Thanks to PR #363 (@evangelinehelsinki) for the exemplary research format, PR #831 (@sseanliu) and PR #670/#756 (@abaybektursun) for deep negative-result studies referenced throughout, and all merged PR authors whose ablation data made this taxonomy possible.
