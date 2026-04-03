# Non-Record: Exact Sequence Matching + TTT on PR #549 (Previous SOTA)

**Base: [PR #549](https://github.com/openai/parameter-golf/pull/549)** (LeakyReLU + Legal TTT + Parallel Muon, 1.1194 BPB) by @abaybektursun

**Sequence matching + TTT BPB: 1.1177** | TTT alone: 1.1195 | Delta from seq match: **-0.0018 BPB** | 8xH100 SXM, 600s training

## Summary

Exact sequence matching is a pure eval-time trick. As the sliding window pass moves left to right, it stores exact 8-12 token contexts from already-scored positions and the token that followed them. When the same context shows up again, it mixes that cached next-token prediction into the model's output.

This PR shows that it still helps after TTT. TTT adapts the model during eval; sequence matching then picks up literal repetition that TTT doesn't explicitly memorize. I tried the same code on two different base submissions -- this PR applies it to PR #549 + TTT; the companion PR applies it to PR #1019 (current SOTA) without TTT.

## Results

| Metric | Score |
|--------|-------|
| Pre-quant BPB (post-EMA) | 1.1371 |
| Sliding window BPB (stride=64) | 1.1218 |
| TTT BPB | 1.1195 |
| **TTT + Sequence matching BPB** | **1.1177** |
| Improvement from sequence matching (over TTT) | **-0.0018 BPB** |
| Improvement from sequence matching (over sliding window) | **-0.0041 BPB** |
| Artifact size | 15,882,529 bytes |
| Training steps | 7,146 (84.0 ms/step) |
| Training time | 600s |

### Eval Time Breakdown

| Pass | Time | BPB |
|------|------|-----|
| Sliding window (stride=64) | 97s | 1.1218 |
| TTT (3 epochs, lr=0.002) | 422s | 1.1195 |
| Sequence matching (on TTT-adapted model) | 297s | 1.1177 |
| **Total eval time** | **~816s** | |

This is non-record because the current combined eval path takes about 816s. I haven't optimized that path yet; sequence matching currently runs as a separate pass on top of TTT.

### Sequence Matching Diagnostics

| Metric | Value |
|--------|-------|
| Match rate | 5.03% (390,344 / 7,754,688 positions) |
| Match accuracy | 65.57% |
| Average confidence | 92.79% |

## How It Works

1. **Build**: As the sliding window pass scores each chunk, all N-grams (orders 8-12) and their next tokens are inserted into a hash table.
2. **Query**: For each position being scored, check if the preceding 8-12 tokens exactly match a previously seen sequence. Use the longest match found.
3. **Blend**: If a match is found, create a one-hot distribution for the predicted next token and blend it with the model's softmax:

```
p_final = (1 - lambda * confidence) * p_model + (lambda * confidence) * p_match
```

Where confidence is `match_length / max_order` (longer matches = higher confidence), lambda=0.15, and the blend weight is clamped at 0.5 to prevent overriding the model.

4. **No match, no change**: If there's no match, we leave the model output alone.

## Why It Helps

TTT and exact matching are doing different jobs. TTT adapts the model to the local distribution of the eval stream; exact matching helps when a long token sequence repeats verbatim. That's why sequence matching can still add -0.0018 BPB even after TTT.

- No retraining required -- purely eval-time.
- The same eval code worked on both base submissions I tried.
- The blend is capped, so a bad match usually doesn't hurt much.

## Base Model

This submission uses the previous SOTA (PR #549 by @abaybektursun) as the base, with TTT enabled. The only addition beyond TTT is the `ExactSequenceCache` class and a separate eval function that applies the cache after TTT.

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| BigramHash | 1536 |
| Attention | XSA on last 4 layers |
| Quantization | int6 + LZMA |
| Optimizer | Parallel Muon + Parameter Banking |
| TTT | 3 epochs, lr=0.002, all blocks unfrozen |

## Run Command

```bash
BIGRAM_VOCAB_SIZE=1536 TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
ITERATIONS=9000 CACHE_ENABLED=1 CACHE_LAMBDA_CACHE=0.15 \
CACHE_MIN_ORDER=8 CACHE_MAX_ORDER=12 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Requirements

Flash Attention 3 (Hopper) is required.

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard
```

## Files

- `train_gpt.py` -- Training script (base: PR #549 + TTT + ExactSequenceCache)
- `submission.json` -- Submission metadata
- `train_seed1337.log` -- Full training and evaluation log
- `README.md` -- This file
