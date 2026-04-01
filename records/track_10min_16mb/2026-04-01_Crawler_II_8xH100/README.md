# Crawler II

Adds a fifth flat transformer layer on each side of the crawler bottleneck (5F+1C+5F vs 4F+1C+4F), with shared TAP encoder connections to each crawler loop.

## Results

| Seed | val_bpb (sliding window) | Steps | Size |
|------|--------------------------|-------|------|
| 444  | 1.17651313               | 7074  | 10048191 B |
| 300  | PENDING                  | —     | — |
| **mean** | **PENDING**          |       | **PENDING** |

Hardware: 8×H100 SXM · 600s wallclock · `bytes_code`: 119294

## Architecture changes

- `NUM_FLAT_LAYERS`: 4 → 5 (one additional flat transformer layer on each side of the crawler)

## Reproduce

```bash
# From repo root, with flash-attention/hopper on PYTHONPATH
SEED=444 NPROC_PER_NODE=8 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-01_Crawler_II_8xH100/train_gpt.py
```

Full env (copy-paste ready):

```bash
env \
    SEED=444 \
    MAX_WALLCLOCK_SECONDS=600 \
    WARMDOWN_ITERS=2000 \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    LATE_QAT_THRESHOLD=0 \
    MATRIX_LR=0.03 \
    TORCHDYNAMO_OPTIMIZE_DDP=0 \
    COMPILE_FULLGRAPH=1 \
    NGRAM_EVAL_ORDER=0 \
    MODEL_DIM=512 \
    USE_CRAWLER=1 \
    NUM_FLAT_LAYERS=5 \
    NUM_CRAWLER_LAYERS=1 \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8=1 \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=32 \
    CRAWLER_TAP_LOOP_SPECIFIC=0 \
    CRAWLER_TAP_LAYERS=all \
    ANCHOR_DIM=0 \
    FLAT_WEIGHT_SHARE=0 \
    NPROC_PER_NODE=8 \
    torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-01_Crawler_II_8xH100/train_gpt.py
```
