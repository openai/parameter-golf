# Quick Reference: Test-Time Compute for Parameter Golf

## THE SINGLE MOST IMPACTFUL CHANGE

**Replace non-overlapping eval with sliding window eval.**

Current eval: non-overlapping 1024-token windows → ~50% of tokens have < 512 context.
Sliding window (stride=256): every token has ≥ 768 context → lower CE → lower BPB.

**Expected BPB improvement: 0.015–0.060 (vs 0.005 required by competition)**

## Changes needed in train_gpt.py:

### 1. Add to GPT class (~20 lines):
```python
def get_logits(self, input_ids):
    # Same as forward() but returns logits instead of loss
    # (copy the forward pass code, remove the loss computation)
```

### 2. Add eval_val_sliding() function (~60 lines):
- Process overlapping windows with configurable stride
- Only score the last `stride` tokens per window
- Count bytes for scored tokens only

### 3. Call sliding eval instead of standard eval:
```python
# OLD:
val_loss, val_bpb = eval_val(args, model, ...)

# NEW:
val_loss, val_bpb = eval_val_sliding(args, model, ..., stride=256)
```

## Why this works:
- BPB = sum(CE_per_token) / (total_bytes × ln2)
- With more context, each token's CE is LOWER
- Same bytes → BPB decreases
- This is LEGITIMATE better compression, not a trick
- Competition explicitly allows "evaluation at any sequence length"
- All tokens still scored exactly once

## Compute: 
- 4x more forward passes (stride=256 with window=1024)
- ~4 seconds on 8xH100 (baseline: <1s)
- Budget: 600 seconds. We use <1% of it.

## Diminishing returns:
- stride=512: ~70% of max gain, 2x compute
- stride=256: ~90% of max gain, 4x compute  ← SWEET SPOT
- stride=128: ~95% of max gain, 8x compute
- stride=1:   100% of max gain, 1024x compute (too slow for batching)
