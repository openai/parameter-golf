# Non-Record: Eval-Time N-gram Mixing and the Unbounded Model Growth Problem

**Author:** abaybektursun | **Date:** 2026-03-26 | **Track:** Non-record study

This submission is not a leaderboard entry. It is a study of eval-time n-gram caching — a technique that reduces BPB from 1.11 to 0.38 while preserving strict causality, costing zero artifact bytes, but growing the effective model to 17x the artifact limit at eval time. We present results, explain why this creates a dilemma for the competition, and propose rule clarifications.

---

## Results

All runs use our ValCalib GPTQ base model ([PR #728](https://github.com/openai/parameter-golf/pull/728), 1.1142 BPB, 11L/512d, ~16MB artifact). Single GPU, stride=64, FineWeb val (62M tokens).

| Config | BPB | Eval-time state | Effective model | Time |
|--------|----:|----------------:|----------------:|-----:|
| Base LM (int6 quantized, leaderboard) | 1.1142 | 0 MB | 16 MB | 606s |
| Base LM (float, pre-quant) | 1.1109 | 0 MB | 16 MB | 606s |
| Pure n-gram, no base LM | 1.0615 | 192 MB | 192 MB | 535s |
| Fixed 7-gram, alpha=0.40 | 0.5234 | 192 MB | 208 MB | 824s |
| Backoff 2-7, alpha=0.40 | 0.4923 | 192 MB | 208 MB | 1079s |
| Backoff 2-7, entropy-adaptive alpha | 0.6535 | 192 MB | 208 MB | 1114s |
| **Backoff 2-9, order-adaptive entropy** | **0.3779** | **256 MB** | **272 MB** | **1234s** |

The n-gram cache alone — with no base LM — beats the trained model (1.06 vs 1.11 BPB). Combined, it cuts BPB by 66%.

### 8-GPU results with all-reduce sync (EXP-11)

All-reduce sync cost: 1.6–2.0s total. The first three configs fit within the 600s competition eval budget; α=0.80 exceeds it (939s).

| Config | BPB | Time | Cache | Sync cost |
|--------|----:|-----:|-------|-----------|
| Base LM (8-GPU) | 1.1130 | 110s | None | — |
| Backoff 2-7, α=0.40 | 0.4941 | 401s | Global (all-reduce) | 1.6s |
| Backoff 2-9, α=0.40 | 0.4548 | 500s | Global (all-reduce) | 1.9s |
| Backoff 2-7, α=0.80 | 0.3942 | 939s | Global (all-reduce) | ~2.0s |

Alpha sweep (8-GPU, backoff 2-7): α=0.20 → 0.6180, α=0.40 → 0.4941, α=0.60 → 0.4263, α=0.80 → 0.3942. Higher alpha is monotonically better — the opposite of PR #727's finding (0.9674 BPB). With a global cache, the n-gram is reliable enough that the model should defer to it more, not less. The best alpha (0.80) exceeds the time budget, so in practice α=0.40–0.60 is the operating range.

### Hash collision analysis — the reported BPB scores are inflated

**Update:** our original explanation of the collision mechanism was incomplete. Credit to @Eppie ([comment](https://github.com/openai/parameter-golf/issues/677#issuecomment-4139902162)) for identifying the probability validity issue, and to Mirco on Discord for the `P(cache_bin)` formulation.

We swept bucket counts from 1M to 256M:

| Buckets | BPB | Table memory |
|--------:|----:|---------:|
| 1M | 0.5793 | 48 MB |
| 4M | 0.6535 | 192 MB |
| 64M | 1.0629 | 3 GB |
| 256M | 1.1123 | 12 GB |

The hash ratio `full_table[hash(ctx, tok)] / ctx_table[hash(ctx)]` is not a conditional probability. The two tables use different hash functions mapping to the same number of buckets. With 1M buckets and 62M tokens, each bucket averages ~62 entries in both tables. The ratio of two similarly-populated buckets approaches 1.0. This is `P(cache_bin)` — a collision-aggregated hash ratio, not `P(tok | ctx)`.

The blend `(1-α) * p_model + α * P(cache_bin)` with `P(cache_bin) ≈ 1.0` pushes the correct token's probability up. But the blend is only computed for the correct token. If you computed it for all 1024 tokens, each would also get `P(cache_bin) ≈ 1.0`. The distribution would sum to far more than 1. After renormalization, the n-gram contribution washes out.

The 1-bucket extreme makes this obvious: `P(cache_bin) = T/T = 1.0` for every lookup. Perfect (fake) score.

The reported BPB numbers are not achievable by a valid compressor. With collision-free tables and proper normalization, n-grams would provide at most a modest improvement from genuine corpus repetition.

### What the n-gram cache is

After each token is scored by the base LM, the token and its preceding context are inserted into hash tables. When a future token's context matches a previously seen n-gram, the cached frequency estimate is mixed with the prediction:

```
p_mix = (1 - alpha) * p_model + alpha * p_ngram
```

The tables are built exclusively from already-scored tokens. No future tokens are accessed. Strict causality is preserved.

### What the n-gram cache costs

| Config | Hash table memory | Formula |
|--------|------------------:|---------|
| Orders 2-7 (6 orders) | 192 MB | 6 orders x 2 tables x 4M buckets x 4 bytes |
| Orders 2-9 (8 orders) | 256 MB | 8 orders x 2 tables x 4M buckets x 4 bytes |
| Orders 2-9, 64M buckets | 4,096 MB | 8 orders x 2 tables x 64M buckets x 4 bytes |

None of this counts toward the 16MB artifact limit. The tables are empty at the start of evaluation and grow as tokens are scored. By the end of evaluation, the model that is doing the actual prediction is 16MB of weights plus 256MB of hash tables — **272 MB total**.

---

## The Dilemma

The competition constrains the artifact to 16MB. The intent is clear: force creative compression of model knowledge into a small footprint. But eval-time techniques like n-gram caching, TTT, and LoRA adaptation grow the effective model far beyond 16MB during evaluation — legally, because the rules only constrain the artifact, not the eval-time state.

This creates a gap between what the competition measures and what matters in practice.

### Four dimensions of the gap

|  | Competition | Real-world inference |
|--|-------------|---------------------|
| **Corpus** | Fixed 62M tokens, scored in one pass | Streaming queries, each independent |
| **Time budget** | 600 seconds for 62M tokens | < 200ms per request |
| **Hardware** | 8x H100 80GB (640 GB VRAM) | Often 1 GPU, sometimes CPU |
| **Model size** | 16 MB artifact; eval-time state unconstrained | Total model must fit deployment target |

Each dimension matters:

**1. Inference time.** The competition allows 600 seconds to score 62M tokens. The n-gram cache exploits this by doing O(K) hash lookups per token across K orders, plus table updates after scoring. On a single GPU, our best config takes 1234s. On 8 GPUs with all-reduce sync (EXP-11), backoff 2-7 takes 401s. In real-world inference, you serve one request at a time with a latency budget measured in milliseconds.

**2. Inference hardware.** The competition provides 8x H100 with 640GB of combined VRAM. The hash tables (256 MB per GPU, synced via all-reduce) are negligible relative to this. In deployment, models run on single GPUs, edge devices, or CPUs. The 256MB of hash tables alone exceeds the 16MB artifact by 16x.

**3. Competition setup.** The artifact limit constrains what you ship. But the n-gram cache ships nothing — it materializes at eval time from the scored tokens themselves. The 16MB limit was designed to constrain model capacity. The n-gram cache circumvents this by building an unbounded statistical model during evaluation, limited only by the number of hash buckets allocated.

**4. Real-world evaluation.** In production, a language model scores individual prompts. Each query arrives independently. There is no corpus-level repetition to exploit. The n-gram cache's power comes entirely from within-corpus repetition. On a stream of independent queries, the cache starts empty for each request and provides no benefit.

**5. Inference speed.** The n-gram cache roughly doubles eval time (606s → 1,079s for backoff 2-7). The overhead is constant per token — it doesn't get worse as the cache fills — but a flat 2x slowdown matters when your latency budget is 50–200ms. You pay the per-token cost on every request, but you only get the BPB benefit after millions of tokens of contiguous corpus. On a 500-token prompt, you get the slowdown without the payoff.

### The core tension

The competition implicitly asks: **given N bytes of model, how well can you compress natural language?**

Eval-time caching answers a different question: **given N bytes of model plus unbounded eval-time memory, how well can you compress a specific fixed corpus?**

These are different problems. The second has a much lower floor — any corpus with internal repetition can be compressed toward its empirical entropy by memorizing seen patterns. Our results show the gap is enormous: 1.11 BPB (base LM only) vs 0.38 BPB (base LM + cache). The cache contributes 2/3 of the total compression, yet costs zero artifact bytes.

---

## What's already legal and where the line blurs

The competition already permits eval-time model growth through several mechanisms:

| Technique | Eval-time state growth | Legality status |
|-----------|----------------------:|----|
| Sliding window eval (stride < seq_len) | KV cache, ~20 MB | Uncontroversial |
| Test-time training (score-first TTT) | LoRA deltas, ~2 MB | Technique deemed legal (PRs #549, #548) |
| Per-document LoRA TTT (8 epochs) | LoRA deltas, ~2 MB | Technique deemed legal (PR #596, 0.6430 BPB) |
| N-gram cache (backoff 2-7) | Hash tables, 192 MB | Under review |
| N-gram cache (backoff 2-9, 64M buckets) | Hash tables, 4 GB | Under review |

TTT and LoRA adaptation follow the same principle as the n-gram cache — build state from scored tokens — though the growth is modest (~2 MB vs 192 MB). The question is not whether causality is preserved (it is), but whether unbounded eval-time model growth is in the spirit of the 16MB constraint.

---

## Proposal

### 0. Enforce causality explicitly

The competition assumes causality but does not enforce it. The FAQ says you can only train on tokens "you've already evaluated your model on," but the eval harness does not verify this. Two-pass rescoring (PRs #846, #853, #868, #870, #881, #888) violates causality: pass 2 rescores token #100 using a cache built from tokens #101 through #62M. This should be an explicit, enforced constraint, not an honor-system rule.

### 1. Verify the distribution sums to 1

The most fundamental fix. Require the model to produce a full probability vector over all K tokens at every scored position. The eval script verifies `sum(probs) ≈ 1.0` before scoring:

```python
probs = model.predict(context)        # shape: [vocab_size]
assert abs(probs.sum() - 1.0) < 1e-4  # verify
nll = -torch.log(probs[correct_token])
```

One `torch.sum` per position. Cost: 1–2 seconds for 62M tokens. Negligible.

This catches every invalid distribution: hash-ratio inflation (sum ≈ 410), single-token hacks (sum = K), any post-softmax modification that doesn't renormalize. It passes everything valid: softmax outputs, linear interpolation of valid distributions, Dirichlet-Multinomial, TTT, LoRA, GPTQ. Not n-gram specific. A general invariant the eval should enforce.

### 2. Cap auxiliary eval-time state

Even with valid distributions, the model can grow unboundedly at eval time. Constrain **auxiliary state**: tensors that accumulate during eval and are not derivable from the artifact alone (hash tables, TTT LoRA deltas, anything that persists across batches). Not model weights (deterministic decompression of artifact), not KV cache (recomputed each window), not activations (transient).

A cap of auxiliary state ≤ 32 MB preserves everything currently approved (TTT LoRA at ~2 MB) while constraining techniques that grow the effective model by 10–250x.

### 3. Cap per-token overhead

Eval-time techniques must not increase per-token latency by more than 50% over the base model forward pass. Base LM on 8×H100 takes 110s. A 1.5× cap means 165s max. The n-gram cache takes 401s (3.6×). Catches two-pass rescoring mechanically.

All three fixes preserve everything currently approved.

---

## Surprising findings

1. **Global cache vs partitioned cache:** On 8 GPUs with independent caches (as in PRs #727 at 0.9674 BPB, #788 at 0.9059 BPB), each GPU sees 1/8 of the tokens. This degrades BPB from ~0.49 (global) to ~0.91–0.97 (partitioned). Our EXP-11 implementation solves this with all-reduce sync of hash table deltas across GPUs.

2. **Entropy-adaptive alpha hurts with strong caches:** The sigmoid-gated alpha from PR #727 gives 0.65 BPB — 0.16 BPB *worse* than fixed alpha=0.40 (0.49 BPB). With a global cache, the n-gram is often more reliable than the base LM, and the entropy gate is too conservative.

3. **N-gram alone beats the base LM:** Pure n-gram (no base LM at all) achieves 1.06 BPB vs 1.11 BPB for the trained model. A zero-parameter frequency table built from scored tokens predicts FineWeb better than the trained model.

4. **Three compression phenomena:** The n-gram cache captures (a) deterministic BPE subword completion (orders 2-4), (b) common English collocations (orders 4-6), and (c) verbatim document repetition (orders 6+). Only (c) is corpus-specific.

---

## Reproduction

All scripts are in `experiments/eval_time_mixing/scripts/`:

```bash
# Single-GPU experiments (EXP-0, requires 1xH100 + trained model)
python3 experiments/eval_time_mixing/scripts/eval_ngram.py \
    --model final_model.pt --exp backoff_7

# 8-GPU distributed with global cache (EXP-11)
NGRAM_ENABLED=1 NGRAM_ORDER=9 NGRAM_ALPHA=0.40 \
    torchrun --standalone --nproc_per_node=8 \
    experiments/eval_time_mixing/scripts/eval_ngram_distributed.py

# N-gram match analysis (qualitative)
python3 experiments/eval_time_mixing/scripts/analyze_ngram_matches.py
```

Base model: `train_609_val_calib.py` from [PR #728](https://github.com/openai/parameter-golf/pull/728) (`records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`).

## Credits

N-gram cache concept and initial implementations: [PR #727](https://github.com/openai/parameter-golf/pull/727), [PR #779](https://github.com/openai/parameter-golf/pull/779), [PR #788](https://github.com/openai/parameter-golf/pull/788). Competition design and infrastructure: OpenAI.
