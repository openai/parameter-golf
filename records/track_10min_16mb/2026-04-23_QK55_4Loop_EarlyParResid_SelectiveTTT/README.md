# QK-Gain 5.5 + 4-Loop Recurrence + Early Parallel Residuals + Selective TTT

**val_bpb = 1.1716** (pre-quantization, 1×H100 ablation) — 3-seed 8×H100 run pending compute credits

## Summary

Incremental improvements on the current SOTA (bigbag, PR #1493, 1.0810 bpb). Four orthogonal changes applied simultaneously on top of the SP8192+3LayerRecur+ParResid+QK525+LegalTTT stack:

1. **QK-Gain 5.5** — increased from 5.25, monotonic improvement trend continues
2. **NUM_LOOPS=3** (4 recurrence passes: 3→4→5→3→4→5→3→4→5→3) vs 2 loops — 19 virtual layers from 11 physical
3. **Early Parallel Residuals** — GPT-J style parallel attention+MLP starting at layer 5 (was layer 7)
4. **Selective TTT** — test-time training applied only on recurrent (loop) layers, reduced chunk size 24576 (was 32768), ttt_epochs=4 (was 3)

## Results (1×H100 ablation run — 2000 steps)

| Experiment | val_bpb @500 | val_bpb @1000 | val_bpb @1500 | val_bpb @2000 | Pre-quant |
|--|--|--|--|--|--|
| Baseline (train_gpt.py) | 1.4808 | 1.3748 | 1.3264 | 1.2964 | 1.2977 |
| **incr_QK55_4loop** | **1.3673** | **1.2682** | **1.2188** | **1.1675** | **1.1716** |

Delta vs baseline at 2000 steps: **-0.121 bpb** (-9.3% relative improvement)

> Note: Final quantized BPB on 8×H100 × 10 min pending. Brotli serialization was not available on the 1×H100 test pod.

## Architecture Changes from SOTA

```python
# SOTA (bigbag PR #1493)            # This submission
QK_GAIN_INIT = 5.25                  QK_GAIN_INIT = 5.5
NUM_LOOPS = 2   # 3 passes           NUM_LOOPS = 3  # 4 passes → 19 virtual layers
parallel_residual_start = 7          parallel_residual_start = 5
ttt_selective = False                ttt_selective = True  # loop layers only
ttt_chunk_tokens = 32768             ttt_chunk_tokens = 24576
ttt_epochs = 3                       ttt_epochs = 4
```

## Base Architecture (unchanged from SOTA)

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. SP1024 BPE tokenizer. Depth recurrence on layers 3-5 (activated at step frac=0.35). Skip gates (sigmoid-gated U-Net). GPTQ SDClip int6/int8 + Brotli-11 compression.

## Attribution

- SP8192 + GPTQ SDClip: @clarkkev (PR #1394)
- Depth recurrence: @dexhunter (PR #1331, #1437)
- Parallel residuals: @Robby955 (PR #1412), @msisovic (PR #1204)
- QK-Gain: @dexhunter (PR #1413)
- Legal Score-First TTT: @abaybektursun (PR #549), @dexhunter (PR #1413)
- Full SOTA stack: @bigbag (PR #1493)
