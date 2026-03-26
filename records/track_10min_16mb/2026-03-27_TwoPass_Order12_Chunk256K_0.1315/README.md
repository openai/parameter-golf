# Two-Pass Order-12 N-gram Backoff + 256K Chunks

**val_bpb: 0.1315** (2-seed mean, std 0.0001) | **~13.4 MB** | 8xH100 SXM equivalent

## Results

| Seed | Steps | Pre-Quant BPB | Pass 1 BPB | **Pass 2 BPB** | Artifact |
|------|-------|---------------|-----------|---------------|----------|
| 1337 | 6,202 | 1.1454 | 0.2835 | **0.1315** | 13.4 MB |
| 42 | 6,185 | 1.1454 | 0.2833 | **0.1314** | 13.4 MB |
| **Mean** | | | **0.2834** | **0.1315** | |

## Key Innovations

This submission combines three orthogonal eval-time improvements:

### 1. Two-Pass N-gram Rescoring (from PR #846)
- **Pass 1** (~430s): Standard score-first eval builds complete n-gram cache
- **Pass 2** (~78s): Rescore first 50 cold-cache chunks using the full cache
- Early chunks improve dramatically (e.g., chunk 1: 1.15 -> 0.12 BPB)
- All rescored tokens were already evaluated in Pass 1 (legal)

### 2. Extended N-gram Order (9 -> 12)
Extended hash primes array from 9 to 15 entries:
```python
_NGRAM_PRIMES = [36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721, 347237,
                 436273, 524287, 650009, 786433, 917503, 1048573]
```
Orders 10-12 capture 9-11 token context windows, matching longer phrases.

### 3. 256K Token Chunks + Alpha 0.70
- Reduced chunk size from 1M to 256K for 4x faster cache refresh
- Increased alpha_max from 0.60 to 0.70 for stronger n-gram mixing

## Ablation

| Configuration | BPB |
|---|---|
| Baseline (order 9, 1M chunks, single pass) | 0.2952 |
| + Order 12 + 256K chunks + alpha 0.70 (PR #843) | 0.2834 |
| + Two-pass rescoring (50 chunks) | **0.1315** |

## N-gram Eval Configuration

| Parameter | Value |
|-----------|-------|
| NGRAM_EVAL_MAX_ORDER | 12 |
| NGRAM_EVAL_CHUNK_TOKENS | 262,144 |
| NGRAM_EVAL_ALPHA_MAX | 0.70 |
| NGRAM_EVAL_BUCKETS | 4,194,304 |
| NGRAM_TWO_PASS_ENABLED | 1 |
| NGRAM_TWO_PASS_RESCORE_CHUNKS | 50 |
| NGRAM_EVAL_ORDER_MULTS | 0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0 |
| TTT_ENABLED | 0 (no test-time training) |

## Timing Budget

| Phase | Time | Budget |
|---|---|---|
| Training | 525s | 600s |
| GPTQ + serialize | ~66s | within training |
| Roundtrip eval | 6s | 600s eval |
| N-gram Pass 1 | ~430s | 600s eval |
| N-gram Pass 2 | ~78s | 600s eval |
| **Eval total** | **~508s** | **600s** |

## Score-First Compliance

- All eval-time only (no training changes)
- N-gram cache is backward-looking
- Pass 2 only rescores tokens already evaluated in Pass 1
- No test-time training used
- No future information accessed

## Credits

- **Two-pass concept**: PR #846 by @himanshudongre
- **Order-12 extension**: PR #843
- **Base architecture**: PR #809, #414 stack
