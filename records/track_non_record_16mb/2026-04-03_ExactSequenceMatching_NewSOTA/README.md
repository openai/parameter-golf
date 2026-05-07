# Non-Record: Exact Sequence Matching on PR #1019 (Current SOTA)

**Base: [PR #1019](https://github.com/openai/parameter-golf/pull/1019)** (AR Self-Gen GPTQ + XSA-all + BigramHash 3072, 1.1147 BPB) by @abaybektursun

**Sequence matching BPB: 1.1143** | Sliding window BPB: 1.1152 | Delta: **-0.0009 BPB** | 8xH100 SXM, 600s training

## Summary

Exact sequence matching is a pure eval-time trick. As the sliding window pass moves left to right, it stores exact 8-12 token contexts from already-scored positions and the token that followed them. When the same context shows up again, it mixes that cached next-token prediction into the model's output.

No retraining needed. I tried the same code on two different base submissions -- this PR applies it to PR #1019 (current SOTA); the companion PR applies it to PR #549 + TTT.

## Results

| Metric | Score |
|--------|-------|
| Pre-quant BPB (post-EMA) | 1.1345 |
| Sliding window BPB (stride=64) | 1.1152 |
| **Sequence matching BPB** | **1.1143** |
| Improvement from sequence matching | **-0.0009 BPB** |
| Artifact size | 15,842,788 bytes |
| Training steps | 6,887 (87.1 ms/step) |
| Training time | 600s |

### Sequence Matching Diagnostics

| Metric | Value |
|--------|-------|
| Match rate | 5.03% (390,344 / 7,754,688 positions) |
| Match accuracy | 65.57% |
| Average confidence | 92.79% |

Only 5% of positions trigger the cache, but when they do, the cached next token is right 65.6% of the time. That's good enough for a small blend to help overall.

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

- No retraining required -- purely eval-time.
- The same eval code worked on both base submissions I tried.
- On the TTT base, it still adds another -0.0018 BPB (see companion PR).
- The blend is capped, so a bad match usually doesn't hurt much.

## Base Model

This submission uses the current SOTA (PR #1019 by @abaybektursun) as the base, with no modifications to training. The only addition is the `ExactSequenceCache` class and a separate eval function that applies the cache during sliding window scoring.

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| BigramHash | 3072 x dim=112 |
| Attention | XSA on all 11 layers |
| Quantization | Full Hessian GPTQ int6 (AR self-gen calibration) |
| Compression | LZMA preset=9 |
| Optimizer | Parallel Muon + Parameter Banking |

## Run Command

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
SEQ_MATCH_ENABLED=1 SEQ_MATCH_LAMBDA=0.15 \
SEQ_MATCH_MIN_ORDER=8 SEQ_MATCH_MAX_ORDER=12 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Requirements

Flash Attention 3 (Hopper) is required.

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard
```

## Files

- `train_gpt.py` -- Training script (base: PR #1019 + ExactSequenceCache + eval_val_sliding_seq_match)
- `submission.json` -- Submission metadata
- `train_seed314.log` -- Full training and evaluation log
- `README.md` -- This file
