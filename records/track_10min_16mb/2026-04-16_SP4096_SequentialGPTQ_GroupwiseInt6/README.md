# WIP: Sequential GPTQ with Groupwise Int6 Quantization

**Track:** 10min / 8×H100 / 16MB artifact
**Base:** PR #1218 (clarkkev — SP4096, MLP 4×, WD 0.085, brotli, XSA-all) — 1.098 BPB baseline
**Status:** Implementation complete — requesting compute credits for 3-seed validation

## Approach

This submission improves the post-training quantization pipeline while keeping the training procedure identical to PR #1218. The core insight: #1218 loses ~0.012 BPB from quantization (pre-quant 1.1047 → post-quant 1.1162 non-sliding). Recovering even a fraction of that loss is free — zero training-time cost, zero throughput tax.

### Changes from #1218

#### 1. Sequential cross-layer GPTQ propagation (`collect_hessians_sequential`)

Instead of collecting all Hessians in a single pass and quantizing each layer independently, we process layers one at a time: collect Hessian for layer *i*, quantize it with GPTQ, **inject the quantized weights back into the model**, then collect the Hessian for layer *i+1*. This means later layers' Hessians reflect the actual quantized activations they'll see at eval time, capturing cross-layer error accumulation that per-layer GPTQ misses.

Controlled by `GPTQ_SEQUENTIAL=1` (default on).

#### 2. Groupwise int6 scales (`group_size=128`)

Replace per-row scales with per-group scales (128 columns per group). Each group of weights gets its own fp16 scale factor, giving the quantizer finer control over heterogeneous weight distributions within each row. The scale storage overhead is small (~2% of weight bytes) but the reconstruction error reduction is significant for layers with high weight variance.

Controlled by `GPTQ_GROUP_SIZE=128` (default).

#### 3. Hessian-weighted scale selection

For per-row mode, instead of selecting scales by minimizing MSE `(W - Q)^2`, we minimize the Hessian-weighted error `sum(H_diag * (W - Q)^2)`, which directly optimizes for output reconstruction quality. Columns with high Hessian diagonal (high activation variance) get proportionally more weight in the error metric.

### Why this is distinct

- **Not in PR #1218**: #1218 uses independent per-layer GPTQ with per-row scales and MSE-based selection.
- **Not in PR #1019**: #1019 focuses on self-generated calibration data; this improves the quantization algorithm itself.
- **Not in PR #1204 / #1209**: Those PRs focus on architecture changes (parallel residuals, depth recurrence, TTT).
- **No SLOT / TTT**: Pure post-training compression improvement — clean causal submission.

### Expected improvement

| Metric | Estimate |
|---|---|
| dBPB (vs #1218) | −0.004 to −0.008 |
| Step-time cost | 0 ms (post-training only) |
| Quant time overhead | +5–10s (sequential propagation) |
| int6 compatible | Yes (this IS the int6 path) |
| torch.compile compatible | Yes (post-training only) |

## Implementation details

New/modified functions:
- `_compute_groupwise_scales()` — per-group fp16 scale computation
- `_quantize_with_groupwise_scales()` — apply groupwise quantization
- `_dequant_groupwise()` — reconstruct from groupwise int6
- `collect_hessians_sequential()` — layer-by-layer Hessian collection with error propagation
- `gptq_quantize_weight()` — extended with `group_size` param and Hessian-weighted error metric
- `gptq_mixed_quantize_int6()` — passes `group_size`, stores group metadata
- `dequantize_mixed_int6()` — handles 2D groupwise scale tensors

New hyperparameters:
- `GPTQ_GROUP_SIZE=128` — columns per quantization group (0 = per-row fallback)
- `GPTQ_SEQUENTIAL=1` — enable sequential cross-layer propagation
- `GPTQ_RESERVE_SECONDS=12` — increased from 10 to account for sequential overhead

## Ablation Plan

1. **Baseline**: Reproduce #1218 at 3 seeds (1337, 42, 2025)
2. **+Sequential propagation only**: `GPTQ_SEQUENTIAL=1 GPTQ_GROUP_SIZE=0`
3. **+Groupwise scales only**: `GPTQ_SEQUENTIAL=0 GPTQ_GROUP_SIZE=128`
4. **+Hessian-weighted only**: per-row mode with H-weighted error
5. **Full stack**: All three combined (default config)

Acceptance criteria: paired t-test across 3 seeds, p < 0.01, dBPB > 0.003.

## Run Command

```bash
RUN_ID=1337 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Requirements

Same as PR #1218. Flash Attention 3 (Hopper) required.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
pip install -r requirements.txt
```
