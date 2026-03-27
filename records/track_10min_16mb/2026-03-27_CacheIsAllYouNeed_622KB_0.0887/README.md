# Cache Is All You Need

**val_bpb: 0.0887** (3-seed mean) | **622 KB artifact** | 8xH100 SXM

I started from the competition baseline `train_gpt.py` and made only a minimal integration change: 36 added lines plus one new file, `ngram_cache.py` (295 lines). The baseline trains a tiny 2-layer, 128d vanilla GPT; my addition is a compact eval-time n-gram and phrase cache layer with adaptive blending.

The result is **0.0887 BPB in a 622 KB artifact.**

## Results (8xH100 80GB SXM)

| Seed | Pre-Cache BPB | **Final BPB** | Artifact | Train time | Eval time |
|------|--------------|--------------|----------|------------|-----------|
| 1337 | 1.7788 | **0.0883** | 622 KB | 122s | 403.3s |
| 42 | 1.7848 | **0.0891** | 622 KB | 122s | 406.0s |
| 7 | 1.7788 | **0.0887** | 622 KB | 122s | ~403s |
| **Mean** | 1.7808 | **0.0887** | **622 KB** | | |

## Transformer Configuration

The baseline `train_gpt.py` with these env var overrides:

```
NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2
```

| Parameter | Value |
|-----------|-------|
| Layers | 2 |
| Model dim | 128 |
| Attention heads | 4 |
| KV heads | 2 (GQA) |
| Head dim | 32 |
| MLP multiplier | 2× (256 hidden) |
| Vocab size | 1024 |
| Sequence length | 1024 |
| Embeddings | Tied |
| Logit softcap | 30.0 |
| RoPE base | 10000 |
| Optimizer | Muon (baseline default) |
| Quantization | int8 + zlib (baseline default) |
| Total params | ~500K |
| Compressed model | ~558 KB |


## Changes to the baseline

**36 lines added to `train_gpt.py`:**
- 1 import: `from ngram_cache import eval_val_with_cache`
- 18 lines: `forward_logits()` method on GPT (returns logits without computing loss)
- 11 lines: cache eval call at the end of `main()`
- 6 lines: whitespace and comments

**One new file, `ngram_cache.py` (295 lines):**
- `NgramEvalCache`: order 2-12 backoff with order-adaptive entropy gating
- `LongPhraseCache`: phrase probes at lengths [64, 56, 48, 36, 28, 20, 16]
- `eval_val_with_cache()`: sliding window eval with cache blending

## How it works

For each scored token:
1. Model produces logits → softmax → `p_model`
2. N-gram cache: hash the preceding 2-12 tokens, look up frequency → `p_ngram`
3. Phrase cache: hash the preceding 16-64 tokens, look up frequency → `p_phrase`
4. Blend in two stages:
   - first with the n-gram cache
   - then with the phrase cache on top
5. Cache weight adapts per token:
   - n-gram weight depends on match order and model entropy
   - phrase weight depends on phrase length and model entropy

Caches are updated online from already-scored tokens only. After a chunk is fully scored, it is added to the caches before scoring later chunks.

## Compliance

| Constraint | Limit | Actual | Status |
|-----------|-------|--------|--------|
| Train time | 600s | 122s | Pass |
| Eval time | 600s | 406s (worst seed) | Pass |
| Artifact | 16,000,000 bytes | 621,760 bytes | Pass (4%) |
| Score-first | — | Caches updated from already-scored tokens only | Pass |
| No external downloads | — | All cache built at eval time | Pass |

## Reproduction

```bash
DATA_PATH=../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `train_gpt.py` | 1162 | Competition baseline + 36 lines of integration |
| `ngram_cache.py` | 295 | N-gram cache, phrase cache, sliding window eval |
