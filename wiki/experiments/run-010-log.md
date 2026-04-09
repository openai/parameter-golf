# Run 010: Track A Baseline — Depth Recurrence (No TTT)

**Date**: 2026-04-09  
**Status**: Running (resubmitted after git auth failure)  
**Cluster**: 8xH100 (c-8fb7887u9z)  
**Job ID**: j-x2ywohxld9 (previous: j-o1h9ofcyw4 FAILED - git auth)

**Failure Root Cause**: Tried to clone personal GitHub fork without credentials. Fixed by cloning official public repo + overlaying our files.

## Hypothesis

3-layer depth recurrence (L3-5) beats our previous 2-loop on L4-5, even without TTT.

**Expected**: ~1.08-1.09 BPB (architecture gain offsets TTT loss from Run 007/008's 1.07389)

## Configuration

| Parameter | Run 007/008 (Baseline) | Run 010 |
|-----------|------------------------|---------|
| Recurrence Type | 2-loop on L4-5 | **Depth recurrence L3-5** |
| Virtual Layers | ~13 | **14** (11 + 3) |
| TTT | 6ep pre-quant (illegal) | **NONE** (Track A) |
| QK-Gain | 5.0 | **5.25** |
| Weight Decay | 0.085 | **0.095** |
| Matrix LR | 0.04 | **0.022** |
| Warmdown Frac | 0.667 | **0.72** |

**Unchanged**: SP1024 tokenizer, parallel residuals L7+, EMA 0.9965, GPTQ int6 + Brotli

## Why Depth Recurrence?

PR #1487 uses 3-layer depth recurrence and achieved 1.0600 BPB (with pre-quant TTT). Their architecture without TTT should be around 1.08-1.09 BPB.

**Depth Recurrence vs. Looping**:
- **Looping**: Iterates over layers 4-5 multiple times (shared weights)
- **Depth Recurrence**: Reuses layers 3-5 inline in forward pass (11→14 virtual layers)
- **Direct comparison**: Unknown — this run tests it

## Expected Results

| Metric | Run 007/008 | Run 010 Target |
|--------|-------------|----------------|
| val_bpb (3-seed mean) | 1.07389 (with illegal TTT) | **~1.08-1.09** (Track A legal) |
| vs Official SOTA (1.1147) | -0.041 BPB | **-0.025 to -0.035 BPB** |
| Training time | 588s | ~590s |

## Actual Results

*Job running: j-o1h9ofcyw4 on c-8fb7887u9z (8xH100)*
*Expected completion: ~15-20 min from submission*

## Post-Mortem

*To be filled after results*

---

## Run 009: CANCELLED — Pre-Quant TTT Legality Concerns
