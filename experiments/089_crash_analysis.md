# Exp089 Crash Analysis

## Experiment Config
- **Run name**: `089_10L_int5post_ttt_prune`
- **Script**: `pr135_modified.py`
- **Host**: `wnckcaina01` (animal.netravi.net), 8×H100
- **Timestamp**: 2026-03-20 17:35:39

## Training Results (EXCELLENT — all completed successfully)
| Step | val_bpb | train_time |
|------|---------|------------|
| 500  | 1.3933  | 45.1s      |
| 1000 | 1.3226  | 90.6s      |
| 2000 | 1.2699  | 181.6s     |
| 3000 | 1.2493  | 272.3s     |
| 4000 | 1.2352  | 362.9s     |
| 5000 | 1.2082  | 453.4s     |
| 6000 | 1.1790  | 544.0s     |
| 6500 | 1.1620  | 589.2s     |
| **6620** | **1.1600** | **600.0s** |

- **Step avg**: 90.64ms/step
- **Peak memory**: 18741 MiB allocated, 18952 MiB reserved
- Stopped at wallclock cap (600s) after 6620 steps

## Post-Training Phase (partially completed)
1. ✅ **Serialization**: fp32 model = 95,552,805 bytes
2. ✅ **Int8+zlib**: 19,883,874 bytes (too big)
3. ✅ **Magnitude pruning**: zeroed 482,360/24,117,248 params (2.00%)
4. ✅ **Int6 quantization**: mixed_quantize_int6 ran on all 8 ranks (60 outlier tensors, 23640 outliers)
5. ✅ **Compression comparison**:
   - int6+zstd: 17,353,883 bytes → total submission: 17,449,495 bytes ❌ Over 16MB
   - int6+manual+zstd: 16,582,683 bytes → total submission: **16,678,295 bytes** ❌ Over 16MB
   - int6+manual+lzma: total submission: 16,883,572 bytes ❌ Over 16MB  
   - int6+manual+zlib9: total submission: 17,249,156 bytes ❌ Over 16MB
6. ❌ **TTT (Test-Time Training): CRASHED**

## Root Cause: `TypeError` in TTT code

### Error
```
TypeError: GPT.forward() missing 1 required positional argument: 'target_ids'
```

### Location
**File**: `pr135_modified.py`, **line 2048**

### Buggy Code (line 2048)
```python
logits = base_model(x_ttt)
```

### Why It Crashed
The `GPT.forward()` method signature requires TWO positional arguments:
```python
def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
```

But the TTT code at line 2048 only passes `input_ids` (`x_ttt`). It expects to get **logits** back, but `GPT.forward()` computes the full forward pass + loss internally and returns a **scalar loss**, not logits.

The TTT code was written assuming `forward()` returns logits (like a standard model), but this GPT class computes `F.cross_entropy` inside `forward()` and returns the loss.

### The Fix
There are two options:

**Option A (recommended)**: Use `forward_logits()` which already exists (line 1421):
```python
# Line 2048: CHANGE THIS:
logits = base_model(x_ttt)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_ttt.view(-1))

# TO THIS:
logits = base_model.forward_logits(x_ttt)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_ttt.view(-1))
```

**Option B (simpler, fewer changes)**: Use `forward()` directly for the loss:
```python
# Lines 2047-2050: CHANGE THIS:
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = base_model(x_ttt)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_ttt.view(-1))

# TO THIS:
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = base_model(x_ttt, y_ttt)
```

Option B is simpler because `forward()` already computes CE loss internally. But Option A is better because it's consistent with the eval code (line 458 uses `forward_logits`).

## Secondary Issue: Artifact Too Large (16.68MB > 16.00MB)

Even if TTT had worked, the artifact is **678KB over budget**:
- Best compression: int6+manual+zstd = 16,678,295 bytes
- Budget: 16,000,000 bytes
- Overage: **678,295 bytes**

### Fixes for artifact size:
1. **Int5 for MLP weights** (clip_range=15 instead of 31): MLP weights have 3 zero high bits in int8 → zstd compresses ~1.88x vs ~1.51x. Should save ~1-2MB.
2. **Reduce model params**: Drop to 9 layers, or reduce MLP hidden dim
3. **Increase pruning** from 2% to 5%: more zeros = better compression
4. **Remove bigram if present**: saves param budget

## Tertiary Issue: Disk Full (100%)

```
/dev/nvme0n1p2  438G  417G  0  100% /
```

Disk is 100% full. This didn't cause the crash (the TypeError happened first), but would likely cause issues with saving artifacts or logs in future runs. Need to clean up old experiment files.

## Summary of Required Code Changes

| Priority | Fix | Impact |
|----------|-----|--------|
| P0 | Change `base_model(x_ttt)` → `base_model.forward_logits(x_ttt)` on line 2048 | Fixes TTT crash |
| P0 | Reduce artifact size by ~700KB (int5 MLP / fewer params / more pruning) | Fits 16MB budget |
| P1 | Clean disk space on wnckcaina01 | Prevents future failures |

## Impact Assessment
The pre-quant val_bpb of **1.1600** at 6620 steps is excellent. If we fix:
1. The TTT crash → expected ~0.006 BPB improvement from TTT
2. The artifact size → int5 MLP quantization should fit AND possibly allow 10th layer
3. Combined: potential val_bpb ~1.14-1.15 range post-quant

This would be a **record-setting result** if all pieces come together.
