# WIP: Sequential GPTQ with Groupwise Int6 Quantization

**Track:** 10min / 8×H100 / 16MB artifact  
**Base:** PR #1218 (clarkkev — SP4096, MLP 4×, WD 0.085, brotli, XSA-all) — 1.098 BPB baseline  
**Status:** Work in progress — requesting compute credits for 3-seed validation

## Approach

This submission improves the post-training quantization pipeline while keeping the training procedure identical to PR #1218. The core insight: PR #1218 loses ~0.012 BPB from quantization (pre-quant 1.1047 → post-quant 1.1162 non-sliding). Recovering even a fraction of that loss through better quantization is free — zero training-time cost, zero throughput tax.

### Changes from #1218

1. **Sequential layer-wise GPTQ propagation**: Instead of quantizing each layer independently, propagate the reconstruction error of earlier layers through to later layers' Hessian estimates. This captures cross-layer error accumulation that vanilla per-layer GPTQ misses.

2. **Groupwise int6 scales** (group_size=128): Replace per-row scales with per-group scales, giving the quantizer finer control over weight distributions within each row. The scale overhead is small (~2% of weight size) but the MSE reduction is significant for layers with heterogeneous weight magnitudes.

3. **Hessian-weighted scale selection**: Instead of searching over percentile-based clip candidates using MSE, select scales that minimize the Hessian-weighted quantization error `(W - Q)^T H (W - Q)`, which directly optimizes for output reconstruction quality.

### Why this is distinct

- **Not in PR #1218**: #1218 uses independent per-layer GPTQ with MSE-based scale search.
- **Not in PR #1019**: #1019 focuses on self-generated calibration data; this improves the GPTQ algorithm itself.
- **Not in PR #1204 / #1209**: Those PRs focus on architecture changes (parallel residuals, depth recurrence, TTT).
- **No SLOT / TTT**: This is a pure post-training compression improvement — clean causal submission.

### Expected improvement

| Metric | Estimate |
|---|---|
| dBPB (vs #1218) | −0.004 to −0.008 |
| Step-time cost | 0 ms (post-training only) |
| Quant time overhead | +5–10s (sequential propagation) |
| int6 compatible | Yes (this IS the int6 path) |
| torch.compile compatible | Yes (post-training only) |

## Ablation Plan

1. **Baseline**: Reproduce #1218 at 3 seeds (1337, 42, 2025)
2. **+Sequential propagation**: Add cross-layer error propagation only
3. **+Groupwise scales**: Add group_size=128 scales on top
4. **+Hessian-weighted selection**: Replace MSE scale search with H-weighted
5. **Full stack**: All three combined

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
