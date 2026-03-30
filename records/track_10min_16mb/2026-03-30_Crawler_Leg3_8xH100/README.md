# Crawler — val_bpb PLACEHOLDER (3-seed mean)

**Micro Crawler**: 4 flat XSA layers + 1 shared crawler block × 3 loops, mlp_mult=6.0. QAT via CRAWLER_QUANT_INT8=1. Naive int6 + zstd, ~8.8MB.

## Results

| Seed | val_bpb (int6 SW exact) | Steps | Size |
|------|------------------------|-------|------|
| 1337 | 1.18720375             | 8087  | 8,842,981 bytes |
| 42   | PLACEHOLDER            | PLACEHOLDER | PLACEHOLDER bytes |
| 300  | PLACEHOLDER            | PLACEHOLDER | PLACEHOLDER bytes |
| **mean** | **PLACEHOLDER**    |       | **PLACEHOLDER bytes (max)** |

Hardware: 8×H100 SXM, 600s wallclock cap.

## Config

- 4 flat XSA layers + 1 crawler block × 3 loops
- CRAWLER_MLP_MULT=6.0
- CRAWLER_QUANT_INT8=1 (QAT)
- GQA: 8 heads, 4 KV heads
- Bigram hash table: 2048
- RoPE: 16
- WARMDOWN_ITERS=2000
- SWA_EVERY=50
- SKIP_GPTQ=1 — naive int6 quantization, zstd compressed
- SKIP_EMA=1
- NGRAM_EVAL_ORDER=0 (legal — no ngram)
- 14.5M parameters

## Reproduce

```bash
SEED=1337 NPROC_PER_NODE=8 bash experiments/Crawler_Leg_3/run.sh
NPROC_PER_NODE=8 bash experiments/Crawler_Leg_3/run_multi_seed.sh
```

See logs in `experiments/Crawler_Leg_3/results/`.
