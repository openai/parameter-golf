# Non-Record: XSA-all-layers + VRL + bigram3072 + lzma9 sweep — 1.1509 bpb

**Score:** 1.15088552 bpb (single seed, sliding window stride=64)
**Hardware:** 8×H100 SXM, 600s training
**Artifact:** 15,316,405 bytes (15.3MB, under 16MB cap)
**Steps:** 3,796 @ ~158ms/step
**Base:** PR #414 stack (11L d512, XSA4, EMA, QAT, lzma)

## Summary

Systematic sweep of four axes on the 11L d512 architecture, each individually measured, then combined:

| Change from #414 stack | Delta bpb (approx) | Notes |
|---|---|---|
| XSA on all 11 layers (XSA_LAST_N=11) | −0.002 | vs last-4-only |
| Value Residual Learning (VALUE_RESIDUAL=1) | −0.001 | VRL on all XSA layers |
| bigram3072 (3072-vocab bigram head, dim=112) | −0.001 | vs bigram2048 |
| lzma preset=9 (vs preset=6) | 0.0 bpb, −200KB artifact | tighter compression |
| AdamW TTT (lr=0.002, 3ep) | **+0.13** (hurts) | see below |

**Combined result without TTT:** 1.1509 bpb
**Legal TTT eval (AdamW, lr=0.002, 3ep):** 1.2804 bpb — TTT at this LR degrades quality.

## Architecture

```
NUM_LAYERS=11, MODEL_DIM=512
XSA_LAST_N=11       # Cross-attention on ALL 11 layers (vs last 4 in #315/#414)
VALUE_RESIDUAL=1    # V = V + residual_V (value gating)
BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=112
QAT_ENABLED=1       # Full-training fake-quant (STE int6)
SWA_ENABLED=0       # SWA disabled (hurts with XSA-all)
TTT_ENABLED=1, TTT_LR=0.002, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768
```

## Code Changes from Baseline

Two changes vs PR #414's train_gpt.py:

### 1. AdamW TTT optimizer (line ~1136)
```python
# Before:
optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
# After:
optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

### 2. lzma preset=9 (line ~1803)
```python
# Before:
quant_blob = lzma.compress(quant_raw, preset=6)
# After:
quant_blob = lzma.compress(quant_raw, preset=9)
```

The architecture changes (XSA_LAST_N=11, VALUE_RESIDUAL=1, BIGRAM_VOCAB_SIZE=3072) are all env-var controlled and require no code changes.

## Key Findings

### XSA on all layers works well

XSA_LAST_N=11 (cross-attention on every layer) produces ~0.002 bpb improvement vs XSA_LAST_N=4 on this 11L d512 stack. This is consistent with the hypothesis that broader cross-sequence attention helps throughout the depth.

### lzma9 compression: measured ratio ~0.96 (not 0.85)

For 11L d512 int6-quantized weights:
- lzma preset=6: ~15.50MB
- lzma preset=9: ~15.23MB (−270KB)
- zlib: ~15.92MB (baseline for comparison)

The compression ratio for lzma9 on int6 weights is approximately **0.96 of uncompressed int6**, not 0.85 as one might assume from zlib. This is critical for byte budget calculations: the 11L d512 architecture with bigram3072 + lzma9 sits at ~15.3MB with ~700KB headroom.

### AdamW TTT at lr=0.002 hurts significantly

Replacing SGD with AdamW in the TTT adaptation loop at the same learning rate (0.002) degraded the legal TTT score from 1.1509 → **1.2804 bpb** (+0.13 bpb regression). The model without TTT (sliding window eval) scores 1.1509.

Possible causes:
- AdamW's adaptive learning rates may interact poorly with the per-document adaptation pattern (each document sees only a few gradient steps before optimizer state is reset)
- LR=0.002 is appropriate for SGD but may be too high for AdamW in this setting — the effective per-parameter LR is lower but the adaptive moments may amplify unstable directions
- The SOTA TTT approaches use SGD with specific momentum settings tuned for TTT; AdamW is not a drop-in replacement

**Recommendation:** If using AdamW for TTT, use a much lower LR (try 1e-4 to 1e-3) and potentially reset optimizer state per-document rather than carrying it across the validation set.

## Full Training Log (key lines)

```
step:3796/20000 val_loss:1.9680 val_bpb:1.1655 train_time:600157ms
stopping_early: wallclock_cap train_time:600157ms step:3796/20000
Serialized model int6+lzma: 15226312 bytes
Total submission size int6+lzma: 15316405 bytes
final_int6_sliding_window_exact val_loss:1.94321685 val_bpb:1.15088552
legal_ttt_exact val_loss:2.16183080 val_bpb:1.28036136
```

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With env vars:
```
NUM_LAYERS=11 MODEL_DIM=512 XSA_LAST_N=11 VALUE_RESIDUAL=1
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
QAT_ENABLED=1 SWA_ENABLED=0
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
```

Note: score reflects sliding window eval, not legal TTT. To disable TTT: `TTT_ENABLED=0`.
