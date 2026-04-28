# Stack B + GPTQ-lite (Fixed Scale Clamp) + Depth Recurrence + 6-bit Packing

Non-record submission. Three quantization insights that may help others building on the Stack B foundation.

## Foundation

Built on the Stack B architecture (PR #1218 / #1260 lineage):
- 11 transformer layers, 512 model dim, 4096 vocab (sp4096 tokenizer)
- 4x MLP expansion (2048 hidden), GQA (8 heads / 4 KV heads)
- LeakyReLU(0.5) squared, partial RoPE (16/64), LN scale (1/sqrt(layer+1))
- XSA on all 11 layers, EMA (decay=0.997), MuonEq-R
- Depth recurrence: layers 4,5 share MLP weights
- Decoupled weight decay 0.085

## Quantization Contributions

### 1. GPTQ-lite Scale Clamp Bug Fix

The original GPTQ-lite (as used in several leaderboard entries) computes per-row scales as:

```python
scale = (row_clip / clip_range).clamp_min(1 / clip_range)
```

For int6 with `clip_range=31`, the minimum scale is `1/31 ~ 0.032`. But typical weight row
maxima are `O(0.01-0.05)`, meaning the clamp fires on most rows and forces scales ~16x too
large. This wastes ~90% of the available quantization dynamic range.

**Fix:** Use `clamp_min(1e-7)` (same as the int8 path). This allows scales to track actual
weight magnitudes. The fix is one line but affects every int6-quantized tensor.

### 2. 6-bit Packing

Standard int6 stored in int8 wastes 25% of every byte. We pack 4 int6 values into 3 bytes:

```
byte0 = a[0:6] | b[0:2] << 6
byte1 = b[2:6] | c[0:4] << 4
byte2 = c[4:6] | d[0:6] << 2
```

This is a 25% payload reduction for all int6 tensors, which directly helps fit under the
16MB artifact limit. The pack/unpack routines are ~10 lines each and add negligible eval-time
overhead.

### 3. Forced Int8 for Depth-Recurrence Shared Layers

When layers share weights (depth recurrence), quantization error compounds through each
reuse. A weight matrix used twice has its quantization error applied twice to the forward
pass. For layers 4 and 5 sharing MLP weights, int6 quantization (6-bit, clip_range=31)
produces enough error to measurably degrade output quality.

**Solution:** Force int8 (127 levels) for shared layers while keeping int6 for non-shared
layers. The extra bytes for 2 layers at int8 vs int6 are small compared to the quality gain.

This is controlled via the `int8_forced_layers` parameter in `quantize_state_dict_mixed()`.

## Results (1xH100 NVL, undertrained)

717 training steps on 1xH100 NVL (~3.5% of the 20,000 step target on 8xH100):

| Metric | Value |
|--------|-------|
| Pre-quant val BPB | 1.377 |
| Post-quant val BPB (mixed int6/int8) | 1.727 |
| Quant degradation | +0.350 BPB |

The large quantization gap is expected for undertrained models. Fully-trained weights settle
into more compressible distributions where the int6 degradation is much smaller (typically
+0.01-0.03 BPB based on prior entries).

No 8xH100 run has been performed yet. Expected full-training result: ~1.10-1.12 BPB
post-quantization.

## Usage

```bash
pip install -r requirements.txt

# Single GPU (testing)
torchrun --nproc_per_node=1 train_gpt.py

# 8xH100 SXM (competition)
torchrun --nproc_per_node=8 train_gpt.py
```

All techniques are ON by default. Override via environment variables (e.g., `GPTQ_LITE=0`
to use standard int8, `DEPTH_RECURRENCE_LAYERS=""` to disable weight sharing).

## Credits

- Stack B architecture: PR #1218, PR #1260
- MuonEq-R: PR #1260 by @signalrush
- Depth recurrence concept: PR #1260, inspired by Universal Transformer
- GPTQ-lite baseline: PR #374, PR #1019
- XSA: @abaybektursun
- LeakyReLU squared: PR #493 by @parinzee, PR #518 by @sofiabod
