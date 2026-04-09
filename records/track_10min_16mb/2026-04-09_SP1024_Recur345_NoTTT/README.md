# Run 010: Track A Baseline — Depth Recurrence (No TTT)

**Date**: 2026-04-09  
**Status**: Pending submission  
**Track**: A (no adaptation)

## Hypothesis

3-layer depth recurrence (layers 3,4,5) beats our previous 2-loop on layers 4-5, even without TTT.

**Expected**: ~1.08-1.09 BPB (architecture gain offsets TTT loss from Run 007/008's 1.07389)

## Configuration Changes vs Run 007/008

| Parameter | Run 007/008 | Run 010 |
|-----------|-------------|---------|
| **Recurrence Type** | 2-loop on L4-5 | **Depth recurrence L3-5** |
| **TTT** | 6ep pre-quant (illegal) | **NONE** (Track A) |
| **QK-Gain** | 5.0 | **5.25** |
| **Weight Decay** | 0.085 | **0.095** |
| **Matrix LR** | 0.04 | **0.022** |
| **Warmdown Frac** | 0.667 | **0.72** |
| **Tokenizer** | SP1024 | SP1024 |

## Architecture

- **Layers**: 11 physical → 14 virtual (via depth recurrence on L3-5)
- **Virtual sequence**: 0,1,2,3,4,5,3,4,5,6,7,8,9,10
- **Parallel residuals**: From layer 7+
- **Skip gates**: Enabled
- **QK-Gain**: 5.25
- **EMA decay**: 0.9965

## Why Depth Recurrence?

PR #1487 uses 3-layer depth recurrence and achieved 1.0600 BPB (with pre-quant TTT). Their architecture (without TTT) should be around 1.08-1.09 BPB based on the TTT contribution (~0.02 BPB).

**Depth Recurrence vs. Looping**:
- **Looping** (our Run 007/008): Iterates over layers 4-5 multiple times (shared weights)
- **Depth Recurrence** (PR #1487): Reuses layers 3-5 inline in forward pass (11→14 virtual layers)
- **Direct comparison**: Unknown — this run tests it

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch tokens | 786,432 |
| Max wallclock | 590s |
| Warmup | 20 steps |
| Warmdown | 72% |
| Weight decay | 0.095 (Muon + Adam) |
| Matrix LR | 0.022 |
| Recurrence start | Step 2000 |

## Quantization & Eval

- GPTQ int6 (matrices) + int8 (embeddings)
- Brotli compression
- Sliding window (stride=64)
- ETLB enabled

## Compliance (Track A)

- ✓ No training on validation data
- ✓ No eval-time adaptation
- ✓ No SLOT, no n-gram cache
- ✓ Fixed predictor at eval time

## Reproduction Command

```bash
export SEED=314 VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512
export DEPTH_RECUR_ENABLED=1 DEPTH_RECUR_LAYERS="3,4,5" DEPTH_RECUR_START_STEP=2000
export PARALLEL_START_LAYER=7 QK_GAIN_INIT=5.25
export ADAM_WD=0.095 MUON_WD=0.095 MATRIX_LR=0.022 WARMDOWN_FRAC=0.72
export EMA_DECAY=0.9965
export EMBED_BITS=8 MATRIX_BITS=6 COMPRESSOR=brotli GPTQ_ENABLED=1
export SLIDING_WINDOW_ENABLED=1 ETLB_ENABLED=1
export TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=590
export TRAIN_BATCH_TOKENS=786432
torchrun --nproc_per_node=8 train_gpt.py
```

## Credits

- **Depth recurrence**: PR #1331, PR #1471
- **Hyperparameters**: PR #1487 (QK=5.25, WD=0.095, warmdown=0.72)
- **SP1024 tokenizer**: Our novel approach

## Run Log

| Seed | Pre-quant BPB | Final BPB (quant+slide+ETLB) | Status |
|------|---------------|------------------------------|--------|
| 314 | TBD | TBD | Pending |
| 42 | TBD | TBD | Pending |
| 999 | TBD | TBD | Pending |
| **Mean** | - | **TBD** | - |
