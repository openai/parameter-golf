
# Run Tracker

## Our Runs

| Run ID | Date | BPB | Size (MB) | Config | Notes |
|--------|------|-----|-----------|--------|-------|
| run-001 | 2026-03-22 | 1.3477 | 14.04 | 11L unique, int4 QAT, all quick wins | 1xH100, only 1037 steps — model undertrained |

---

## Run Log

### Run: run-001
**Date:** 2026-03-22
**GPU:** 1xH100
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
**Result:** BPB=1.3477  |  Size=14.04 MB (under 16MB cap, ~2MB headroom)
**Train time:** 600s (hit wallclock cap)
**Steps:** 1037/20000 @ 578ms/step
**Peak memory:** 13906 MiB
**Changes from baseline:** Int4 QAT on MLP, 11 layers (was 9), 3x MLP (was 2x), BigramHash, SmearGate, Muon WD 0.04, grad clip 0.3, warmdown 3000, momentum 0.99, magnitude pruning 3%
**Observations:**
  - Model severely undertrained — 1xH100 gets ~1037 steps vs ~20K expected on 8xH100
  - Warmdown never kicked in (needs 3000 iters, we barely passed 1000)
  - BPB 1.3477 worse than baseline 1.2244 but not a fair comparison — need 8xH100 or longer wallclock
  - Size is good: 14.04MB with 2MB to spare, int4 quantization working well (3.78x compression ratio)

---

## A/B Tests To Run

| Test | Config A | Config B | Hypothesis |
|------|----------|----------|------------|
| Depth recurrence vs unique layers | NUM_LAYERS=6 RECURRENCE=2 | NUM_LAYERS=11 RECURRENCE=1 | Unique layers win but recurrence is cheaper |

## Best Config So Far
```
# paste winning env vars here once we have runs
```
