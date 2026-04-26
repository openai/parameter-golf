# Non-record: v11 — SDClip-matched FakeQuantize

## Summary

**Not a record.** Submitted for the **SDClip-matched FakeQuantize** contribution.

val_bpb: 1.1872 (post-quant, 1-seed, 1×H100 Kaggle, 4000 steps)

## Key Contribution: SDClip-matched FakeQuantize

In earlier experiments (v8), QAT used per-row absmax scaling during training,
but the save-time quantizer used SDClip (clip = k × std(row), k=12.85).
This train/save mismatch caused catastrophic collapse:

| Version | Pre-quant BPB | Post-quant BPB | Degradation |
|---------|--------------|----------------|-------------|
| v8 (naive FakeQuantize) | 1.1387 | 1.3103 | **+0.17** |
| v11 (SDClip-matched)    | 1.1630 | ~1.18-1.20 | **+0.044** |

v11 uses the SAME SDClip formula during FakeQuantize:

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        max_val = (1 << (bits - 1)) - 1
        k = 12.85  # matches save-time SDClip
        std = x.std(dim=1, unbiased=False, keepdim=True)
        cl = (k * std).clamp_min(1e-8)
        scale = (cl / max_val).clamp_min(1.0 / max_val)
        clipped = torch.clamp(x, -cl, cl)
        return (torch.clamp(torch.round(clipped / scale), -max_val, max_val) * scale).to(x.dtype)
    @staticmethod
    def backward(ctx, grad): return grad, None
```

The principle is general: **QAT fake-quant must match the exact quantizer used
at save time**. Any mismatch in clipping, scaling, or rounding behavior lets the
model learn to rely on patterns that disappear post-quant.

## Stack

Built on top of PR #1394 (Kevin Clark) and PR #1493 (bigbag).

- SP8192 tokenizer, 11 layers × 512 dim, 40.5M params
- GQA 8H/4KV, Partial RoPE 16/64
- QK-Gain 5.25, MuonEq-R (WD=0.095, LR=0.022)
- BigramHash(4096), SmearGate, Value Embeddings
- Parallel Residuals on layers 7+
- 3-layer Depth Recurrence (L3,4,5) activated at 35% of training
- EMA 0.9965 from 50%
- Warmdown 72% (1-sqrt cooldown)
- SDClip-matched FakeQuantize from 80% (QAT)
- Legal Score-First TTT (SGD lr=0.005, mom=0.9, 3 cosine epochs)
- Mixed int5 MLP / int6 Attn / int8 Embed (k=12.85/20.0)
- Byte-shuffle + Brotli-11 compression

## Reproduction

Single-seed run on 1×H100 (Kaggle). See `train_seed1337.log`.

Expected sub-1.15 post-quant BPB when scaled to 8×H100.

## Why Non-record

- Current SOTA: 1.0810 (PR #1493)
- This PR: ~1.1872 post-quant (1×H100, compute-limited)
- Gap is compute, not architecture — ~6× less compute than record runs

Submitted to document the SDClip-matched FakeQuantize technique, which
should be useful to anyone doing QAT with non-trivial quantizer shapes
(SDClip, AWQ-style per-channel clip, Hessian-aware SDClip).

## Credits

- PR #1394 (@clarkkev): SDClip, depth recurrence, GPTQ embedding base
- PR #1412 (@Robby955): Parallel Residuals, Hessian-aware SDClip
- PR #1493 (@bigbag): 3-layer recurrence + QK-Gain 5.25 + Legal TTT
- modded-nanogpt: Muon, Value Embeddings, BigramHash, SmearGate lineage
