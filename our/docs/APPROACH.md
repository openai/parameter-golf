# Parameter Golf — Our Approach

## Challenge Summary

Train the best language model that fits in a **16MB artifact** (code + int8+zlib compressed weights), trainable in **<10 min on 8×H100s**, scored by **bits-per-byte (BPB)** on FineWeb validation set. Lower BPB = better.

- Baseline: **1.2244 BPB** (9 layers, dim=512, vocab=1024, tied embeddings, 17M params)
- Leaderboard: https://github.com/openai/parameter-golf

## What We've Done

### 1. Baseline Smoke Test (MacBook, 200 iters, 1 shard)

Verified the setup works. Got **val_bpb = 2.3033** (post int8 roundtrip). Much higher than the H100 leaderboard score because we trained for only 200 steps on 1 data shard vs 20K steps on 80 shards.

### 2. Depth Recurrence Experiment (first improvement)

**Idea:** Instead of 9 unique transformer blocks, use **3 unique blocks looped 3 times**. Same effective depth (9), but ~1/3 the parameters for transformer layers. Use the freed parameter budget to **widen the model** from dim=512 to dim=768.

**Config:**
- 3 unique layers × 3 recurrences = 9 effective depth
- dim=768, 12 heads, 4 KV heads, MLP mult=2
- ~12.6M params (vs 17M baseline)

**Results (200 iters, 1 shard, MacBook):**

| Model | Params | BPB (post int8 roundtrip) |
|-------|--------|--------------------------|
| Baseline (9 layers, dim=512) | 17M | 2.3033 | 9.3MB |
| **Recurrent (3×3, dim=768)** | **12.6M** | **2.2714** | **6.9MB** |

**Improvement: 0.032 BPB better with 26% fewer params and 6.9MB compressed (tons of room under 16MB).**

## Next Steps (prioritized)

### Near-term (local MacBook experiments)

1. **Larger vocabulary** — Current 1024 vocab is very small. A 4K-8K vocab means fewer tokens per byte → directly lowers BPB. Trade-off: embedding table uses more of the 16MB budget. Need to retrain the tokenizer or use a larger one.

2. **Tune recurrence depth vs width** — Try other configs:
   - 4 unique × 3 recurrences = 12 effective depth (deeper)
   - 2 unique × 5 recurrences = 10 effective depth (more sharing)
   - Vary dim (640, 768, 896) to find the sweet spot under 16MB

3. **Quantization-Aware Training (QAT)** — Simulate int8 rounding during training so the model learns weights that survive compression better. Reduces the gap between pre- and post-roundtrip BPB.

4. **Low-rank factorization** — Factor Q/K/V weight matrices as products of two smaller matrices. Reduces param count further while preserving expressiveness.

### Medium-term (requires H100 compute)

5. **Full training runs** — Validate that local improvements transfer to 8×H100 with 20K iterations on 80 shards.

6. **Test-time compute** — Use extra compute at evaluation time (allowed up to 10 min) to improve predictions, e.g., self-consistency or iterative refinement.

7. **Hyperparameter sweep** — Tune learning rates, warmup, warmdown, batch size on H100s.

### Longer-term / speculative

8. **Custom tokenizer** — Design a tokenizer optimized for compression (larger vocab, better BPE merges for FineWeb distribution). Risky: must prove BPB is correctly calculated.

9. **Mixture of Experts (MoE)** — Multiple small expert MLPs with a router. More capacity per parameter, but adds routing overhead.

10. **BitNet / 1-bit weights** — Train with ternary weights {-1, 0, 1}. Extreme compression but needs careful training.

## Key Insight

The 16MB limit is on **unique parameters**, but compute is on **effective depth**. Depth recurrence exploits this gap: reuse the same weights multiple times to get more compute per parameter. This is likely the single highest-leverage optimization for this challenge.

## Grant Application Strategy

Once we confirm the recurrent approach works at scale (with a longer local run or a cheap 1×H100 test), apply for the largest compute grant with evidence:
- Concrete BPB improvement from depth recurrence
- Clear roadmap of stacking optimizations (recurrence + vocab + QAT)
- Estimated target BPB and how it would rank on the leaderboard
