
# Run Tracker

## Our Runs

| Run ID | Date | BPB | Size (MB) | Config | Notes |
|--------|------|-----|-----------|--------|-------|
| run-001 | 2026-03-22 | pending | pending | 11L unique, int4 QAT, all quick wins | First full run, in progress |

---

## Run Log

### Run: run-001 (IN PROGRESS)
**Date:** 2026-03-22
**Config:**
```
NUM_LAYERS=11
RECURRENCE=1
MLP_MULT=3.0
QAT_ENABLED=1
QAT_CLIP_RANGE=7
BIGRAM_VOCAB_SIZE=10240
BIGRAM_DIM=128
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WEIGHT_DECAY=0.04
GRAD_CLIP_NORM=0.3
WARMDOWN_ITERS=3000
TRAIN_SEQ_LEN=1024
```
**Result:** BPB= pending  |  Size= pending MB
**Train time:** pending
**Changes from baseline:** Int4 QAT on MLP, 11 layers (was 9), 3x MLP (was 2x), BigramHash, SmearGate, Muon WD 0.04, grad clip 0.3, warmdown 3000, momentum 0.99, magnitude pruning 3%
**Observations:**

---

## A/B Tests To Run

| Test | Config A | Config B | Hypothesis |
|------|----------|----------|------------|
| Depth recurrence vs unique layers | NUM_LAYERS=6 RECURRENCE=2 | NUM_LAYERS=11 RECURRENCE=1 | Unique layers win but recurrence is cheaper |

## Best Config So Far
```
# paste winning env vars here once we have runs
```
