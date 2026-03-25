# Podracing: 5-gram Eval + LeakyReLU² + GPTQ

## Results

| Seed | Sliding BPB | 5-gram BPB | Artifact |
|------|-------------|-----------|----------|
| 1337 | 1.1190 | **1.0451** | 15.63 MB |
| 42 | 1.1217 | **1.0471** | 15.59 MB |
| 2045 | 1.1200 | **1.0460** | 15.64 MB |
| **Mean** | **1.1202** | **1.0461** | — |

## Architecture

11L/512d U-Net, 8H/4KV, LeakyReLU² (slope 0.5), XSA last 4, BigramHash 1536,
VE128 on layers 9-10, partial RoPE (24/64 dims), tied embeddings. 26.93M params.

## 5-gram Eval (score-first, legal)

Fixed-weight hashed n-gram interpolation during sliding window eval:
- Cache built from already-scored tokens only (backward-looking)
- Fixed alpha=0.20: `p_final = 0.80 * p_model + 0.20 * p_ngram`
- No safety gate, no target-aware selection, no min-NLL comparison
- Hashed count-min sketch (4M buckets), min_count=2
- N-gram concept credited to @deanbrr (PR #659)

## Reproduce

```bash
SEED=2045 MLP_ACT=leaky_relu_sq MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=1536 ROPE_DIMS=24 \
NGRAM_EVAL_ORDER=5 NGRAM_EVAL_ALPHA=0.20 \
NGRAM_EVAL_MIN_COUNT=2 NGRAM_EVAL_BUCKETS=4194304 \
torchrun --nproc_per_node=8 train_gpt.py
```

8xH100 SXM, 600s training + ~190s eval.
