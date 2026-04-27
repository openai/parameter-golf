# Experiment 004 — TTT chunk-size sweep

**Date:** TBD (Day 2)
**Hypothesis:** The current `TTT_CHUNK_TOKENS=32768` was tuned for full-param TTT. With a smaller adapt surface, smaller chunks (= more frequent param updates) may extract more signal, while larger chunks may be wasteful.
**Baseline:** Best LR/EPOCHS combo from Experiment 003
**Cost:** 4 cells × ~2 min = ~$1

## Grid

```
TTT_CHUNK_TOKENS ∈ {8192, 16384, 32768, 65536}
```

Trade-off:
- Smaller chunks → more SGD steps, finer adaptation, more `compile_logits` recompiles (caching helps).
- Larger chunks → fewer steps, more stable but coarser.
- Eval throughput is dominated by sliding-window scoring, not TTT — so chunk-size mainly affects TTT step count.

## Commands

```bash
LR=$WINNER_LR; EP=$WINNER_EP   # from experiment 003
FILTER=$WINNER_FILTER         # from experiment 002

for CHUNK in 8192 16384 32768 65536; do
  TAG="chunk${CHUNK}"
  TTT_ENABLED=1 TTT_PARAM_FILTER=$FILTER \
    TTT_LR=$LR TTT_EPOCHS=$EP TTT_CHUNK_TOKENS=$CHUNK \
    SEED=42 \
    LOAD_CHECKPOINT=$CKPT \
    RUN_ID=opus_e004_${TAG} \
    torchrun --standalone --nproc_per_node=2 \
      Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/004_${TAG}.log
done
```

## Result

| Chunk | val_bpb_ttt | Δ vs 32K | TTT eval time |
|-------|-------------|----------|---------------|
|  8192 | | | |
| 16384 | | | |
| 32768 | | reference | |
| 65536 | | | |

## Decision

Lock in the chunk size that minimizes `val_bpb_ttt`, with eval time as tiebreaker. Combined with the LR/epochs/filter winners, this becomes the **Day 2 locked config** that goes into the 3-seed validation in Experiment 005.
