# Order-12 N-gram Backoff + 256K Chunks + Alpha 0.70

**val_bpb: 0.2834** (2-seed mean, std 0.0001) | **~13.4 MB** | 8xH100 SXM equivalent

## Results (8x L20Z 80GB, PyTorch 2.10+cu128)

| Seed | steps | Pre-Quant BPB | Roundtrip BPB | **N-gram BPB** | Artifact |
|------|-------|---------------|---------------|---------------|----------|
| 1337 | 6,192 | 1.1454 | 1.1643 | **0.2835** | 13,412,672 |
| 42 | 6,185 | 1.1454 | 1.1643 | **0.2833** | ~13.4M |
| **Mean** | **6,189** | **1.1454** | **1.1643** | **0.2834 (std 0.0001)** | |

## Key Improvements over PR #809 (0.2952 BPB)

Three eval-time modifications that together yield **-0.0118 BPB** improvement:

### 1. Extended N-gram Order (9 -> 12)

Added 6 new large primes to the hash function array:
```python
_NGRAM_PRIMES = [
    36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721, 347237,
    436273, 524287, 650009, 786433, 917503, 1048573  # NEW: orders 10-15
]
```

Orders 10-12 capture longer context windows (9-11 preceding tokens) which match multi-word phrases and boilerplate patterns. Per-order multipliers maintained at 2.0x for all high orders.

### 2. Smaller Chunk Tokens (1M -> 256K)

Reduced `NGRAM_EVAL_CHUNK_TOKENS` from 1,000,000 to 262,144. This means the n-gram cache refreshes 4x more frequently:
- Old: 62 chunks, each scored against cache from prior chunks
- New: ~237 chunks, cache updated every 256K tokens

The first chunk (scored against empty cache) is now only 256K tokens instead of 1M, reducing the cold-start penalty. By chunk 5, the cache has seen ~1.3M tokens and already captures common patterns.

### 3. Higher Alpha Max (0.60 -> 0.70)

Increased `NGRAM_EVAL_ALPHA_MAX` from 0.60 to 0.70, allowing stronger n-gram mixing when the model is uncertain. After per-order multipliers (2.0x), effective alpha can reach 0.95 (unchanged clip).

## Ablation

| Configuration | BPB | Delta |
|---|---|---|
| Baseline (order 9, 1M chunks, alpha_max 0.60) | 0.2952 | -- |
| + Order 12 only | 0.2888 | -0.0064 |
| + 256K chunks + alpha_max 0.70 | **0.2834** | **-0.0118** |

## N-gram Eval Configuration

| Parameter | Old (PR #809) | New |
|-----------|--------------|-----|
| NGRAM_EVAL_MAX_ORDER | 9 | **12** |
| NGRAM_EVAL_CHUNK_TOKENS | 1,000,000 | **262,144** |
| NGRAM_EVAL_ALPHA_MAX | 0.60 | **0.70** |
| NGRAM_EVAL_BUCKETS | 4,194,304 | 4,194,304 |
| NGRAM_EVAL_ALPHA_MIN | 0.05 | 0.05 |
| NGRAM_EVAL_ENTROPY_CENTER | 3.0 | 3.0 |
| NGRAM_EVAL_ENTROPY_SCALE | 2.0 | 2.0 |
| NGRAM_EVAL_ORDER_MULTS | 0.3,0.3,0.97,2.0x8 | 0.3,0.3,0.97,2.0x8,**2.0,2.0,2.0** |

## Score-First Compliance

All changes are eval-time only. The n-gram cache remains backward-looking:
- Cache updated AFTER scoring each chunk
- First chunk scored against empty cache
- No test-time training used in this submission
- No training-time changes from PR #809

## Timing Budget

| Phase | Time | Budget |
|---|---|---|
| Training | 525s | 600s |
| GPTQ calibration + serialize | ~66s | 600s |
| **Training total** | **591s** | **600s** |
| Roundtrip eval | 6s | 600s |
| N-gram eval | 431s | 600s |
| **Eval total** | **437s** | **600s** |

## Architecture

Identical to PR #809 (no training changes):
- 11L 512d GQA (8 query, 4 KV), MLP 3.0x LeakyReLU(0.9)^2
- BigramHash(4096), SmearGate, XSA-4, Partial RoPE (16/64)
- GPTQ int5, LZMA compression, ~13.4MB artifact

## Run Command

```bash
MODEL_PRESET=frontier_lean RUN_PROFILE=full_8gpu_600s \
SEED=1337 QAT_MODE=off ENABLE_COMPILE=1 \
MAX_WALLCLOCK_SECONDS=525 TTT_ENABLED=0 \
NGRAM_EVAL_ENABLED=1 \
NGRAM_EVAL_MAX_ORDER=12 \
NGRAM_EVAL_BUCKETS=4194304 \
NGRAM_EVAL_CHUNK_TOKENS=262144 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.70 \
NGRAM_EVAL_ORDER_MULTS='0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0' \
EVAL_BATCH_SEQS=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base submission (PR #809)**: Order-9 chunk-based n-gram backoff by @AayushBaniya2006
- **Architecture stack (PR #414)**: BigramHash, SmearGate, XSA, U-Net, by multiple contributors
- **N-gram backoff concept**: PRs #769, #779, #796
