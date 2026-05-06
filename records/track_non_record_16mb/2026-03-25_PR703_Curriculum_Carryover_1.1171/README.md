# PR703 + Shard-Order Curriculum + GPTQ Cache-Backout

This is a `non-record-16mb` submission built from the PR703-style quant/cache-backout branch and improved with a score-ranked shard curriculum.

This run is being submitted as **non-record** for two reasons:

- it is a single-seed result, not a multi-seed statistically significant record package
- the improvement over the current accepted README leaderboard is well below the `0.005`-nat record threshold described in the root `README.md`

## Result

- `final_int6_sliding_window_exact`: `1.11709895`
- `final_int6_roundtrip_exact`: `1.14068680`
- `post_ema`: `1.1368`
- `step_stop`: `6918`
- `step_avg`: `86.75ms`
- `total submission size`: `15,909,560` bytes
- `bytes under 16MB`: `90,440`

## Core Change Relative to the Forked PR703 Base

The base PR703 carryover result was:

- `1.11748714`
- `15,963,300` bytes

This submission improves that branch mainly by:

1. `Shard-order curriculum`
   Training shards are reordered by a lightweight scorer so the run sees harder shards earlier.

2. `Tighter final compression`
   Final int6 payload uses a stronger `lzma` preset, preserving the same core model family while giving more artifact headroom.

The winning object is still the same general PR703-style branch:

- 11-layer trunk
- cache/backout path
- full-Hessian GPTQ over the banked-attn/MLP surface
- `BIGRAM_VOCAB_SIZE=1536`
- no TTT

## Reproduction

This submission depends on a generated `shard_order.json`. The run used the same shard scorer included here as `score_shards.py`.

First generate shard order:

```bash
python score_shards.py --data-dir ./data/datasets/fineweb10B_sp1024 --device cuda:0 --seq-len 1024 --train-steps 500 --max-batches 50 --batch-size 16 --output shard_order.json
```

Then launch training:

```bash
SHARD_ORDER_FILE=./shard_order.json SEED=2025 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 MILE_GAMMA=1.1 MUON_QUANT_MOMENTUM=1 CACHE_LAYER=7 BACKOUT_LAMBDA_INIT=0.1 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`: exact code snapshot used by the winning run
- `score_shards.py`: shard-order scorer used to generate the curriculum input
- `train.log`: exact controller log for the submitted run
- `submission.json`: leaderboard metadata
