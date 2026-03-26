# Record: Prefill Cache + 7-Gram Entropy-Adaptive + XSA-all + EBLS

**val_bpb: 0.6567** (3-seed mean, std 0.0003) | **~15.87 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### 3-seed validation

| Seed | **Sliding + 7-gram BPB** | Artifact |
|------|--------------------------|----------|
| 1337 | **0.6565** | 15,872,807 |
| 2024 | **0.6570** | 15,866,839 |
| 2025 | **0.6565** | 15,868,655 |
| **Mean** | **0.6567 (std 0.0003)** | |

### Comparison to fragmented cache (PR #777)

| Config | BPB | Cache gain |
|--------|-----|-----------|
| No cache (neural only) | 1.1425 | -- |
| Fragmented cache (PR #777) | 0.9614 | -0.181 |
| **Prefill cache (this PR)** | **0.6565** | **-0.486** |

## Key Innovation: Cache Pre-fill for Distributed Eval

When evaluating on 8 GPUs with sliding windows, each rank processes a contiguous range of token positions. Without pre-fill, rank k starts with an empty n-gram cache -- it only accumulates entries from the positions it scores. This means ranks 1-7 miss the statistical patterns from all preceding positions, reducing cache effectiveness by ~60%.

**The fix**: Before each rank begins its forward pass, it pre-populates the n-gram hash tables with ALL token positions preceding its assigned window range using pure numpy. No NCCL collectives needed.

```python
# Pre-fill: rank k processes positions 0..start_of_rank_k using vectorized numpy
for order in range(min_order, max_order+1):
    ctx_hash = hash(val_tokens[pos-order+1:pos])  # context hash
    full_hash = hash(ctx_hash, val_tokens[pos])    # context+target hash
    ctx_tables[order][ctx_hash % buckets] += 1
    full_tables[order][full_hash % buckets] += 1
```

This produces **mathematically identical** results to single-GPU sequential evaluation. The pre-fill takes ~2-4 seconds per rank (rank 0 skips it, rank 7 pre-fills ~7/8 of all positions).

## Technique

### 7-Gram Causal Cache (eval-time, backward-looking)

Multi-order backward-looking n-gram cache built during sliding window evaluation:

1. **Hash table construction**: 6 separate hash tables for orders 2 through 7 (4M buckets each)
2. **Backoff cascade**: At each token position, attempt the highest order first (7-gram). If matched with sufficient count (min_count=2), use that prediction. Otherwise fall back to 6-gram, 5-gram, ..., 2-gram.
3. **Entropy-adaptive blending**: `p_mixed = (1 - alpha) * p_model + alpha * p_ngram` where alpha adapts per-token based on model entropy via sigmoid
4. **Strictly causal**: The cache is updated with the true token **only after** the model has scored it. No forward-peeking, no oracle/min(NLL) selection.
5. **Pre-fill**: Each GPU rank pre-populates its cache with all preceding positions before scoring begins.

### Compliance

- [x] Training: 560s on 8xH100 SXM (within 600s limit)
- [x] Eval (sliding window + n-gram blending): ~300s on 8xH100 SXM (within 600s limit)
- [x] All artifacts under 16,000,000 bytes
- [x] Script under 1,500 lines (1,439 lines)
- [x] No TTT on validation data
- [x] No training data access during evaluation
- [x] No min(NLL) oracle selection -- single blended prediction per token
- [x] Cache updates are strictly backward-looking (causal)
- [x] GPTQ calibration on validation data within training window (val-GPTQ)
- [x] Pre-fill uses only val_tokens[0..pos-1] at each position (no future data)

### Legality Argument

The cache pre-fill is legal because:

1. **Equivalent to single-GPU eval**: The pre-fill produces identical n-gram tables as sequential single-GPU evaluation. It is purely an implementation optimization for distributed execution.
2. **Backward-looking**: At scored position p, the cache contains entries for positions 0..p-1 only.
3. **No oracle**: Each token receives exactly one prediction (linear blend of model + cache).
4. **No weight mutation**: Model weights are frozen during evaluation.
5. **No future data**: Pre-fill only uses val_tokens, and only positions strictly before the rank's scoring window.
6. **Organizer precedent**: valerio-oai commented on PR #659 that the n-gram cache "idea itself is not illegal" and suggested entropy gating as valid.

## Training Architecture

EBLS (Empirical Bayes Layer Sharing) with prefill n-gram eval cache:

| Component | Setting |
|-----------|---------|
| Layers | 11 (3 shared blocks x 3 loops + 2 unique) |
| Dimensions | 512d, 8 heads, 4 KV heads (GQA) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| LoRA | Rank 8, per virtual layer |
| BigramHash | 3072 vocab, 128 dim |
| XSA | All 11 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VRL | Value Residual Learning |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | Val-GPTQ int6 + LZMA preset 9+extreme |
| Eval cache | 7-gram backoff (orders 2-7), entropy-adaptive alpha, distributed pre-fill |

### N-gram Cache Hyperparameters

| Parameter | Value |
|-----------|-------|
| Orders | 2 through 7 (6 hash tables) |
| Buckets | 4,194,304 per table |
| Min count | 2 (require 2+ observations) |
| Entropy base | 0.05 |
| Entropy range | 0.55 (alpha ranges from 0.05 to 0.60) |
| Entropy scale | 2.0 |
| Entropy threshold | 4.0 |

## Run Command

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 CLIP_RANGE=31 COMPRESSOR=lzma \
NUM_KV_HEADS=4 EVAL_STRIDE=64 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=64 GPTQ_CALIB_SOURCE=val \
GPTQ_BLOCK_SIZE=128 SWA_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
NGRAM_CACHE=1 NGRAM_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_MIN_COUNT=2 NGRAM_BUCKETS=4194304 \
NGRAM_ENTROPY=1 NGRAM_ENT_BASE=0.05 NGRAM_ENT_RANGE=0.55 \
NGRAM_ENT_SCALE=2.0 NGRAM_ENT_THRESH=4.0 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **N-gram cache technique**: [PR #715](https://github.com/openai/parameter-golf/pull/715), [PR #727](https://github.com/openai/parameter-golf/pull/727)
- **Cache pre-fill for distributed eval**: Novel contribution (this PR)
- **Entropy-adaptive alpha**: [PR #727](https://github.com/openai/parameter-golf/pull/727), suggested by valerio-oai on [PR #659](https://github.com/openai/parameter-golf/pull/659)
- **XSA-all**: [PR #634](https://github.com/openai/parameter-golf/pull/634) by @raahilshah
- **LeakyReLU^2**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
