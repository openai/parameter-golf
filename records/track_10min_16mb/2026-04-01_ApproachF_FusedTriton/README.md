# Approach F: Fused Triton MLP Activation Kernel

**val_bpb: TBD** (pending 8xH100 run)
**Artifact: TBD**

## Key Innovation: Fused Triton Activation Kernel

Custom Triton kernels that fuse `relu(x).square()` into a single GPU kernel, eliminating the intermediate hidden-dimension tensor write to HBM. This is a pure systems optimization -- mathematically identical output.

### What's fused

The standard MLP activation path launches two separate elementwise kernels:

```python
# Standard: 2 kernel launches, writes 1792-dim intermediate to HBM
h = torch.relu(self.fc(x))    # elementwise relu, writes to HBM
h = h.square()                  # elementwise square, reads+writes HBM
out = self.proj(h)
```

The fused kernel combines both operations:

```python
# Fused: 1 kernel launch, no intermediate write
h = fused_relu_sq(self.fc(x))  # relu + square in one pass
out = self.proj(h)
```

This saves one full read+write of the hidden dimension tensor (batch * seq_len * 1792 elements) per layer, per forward and backward pass. With 11 layers:
- Forward: 11 fewer HBM roundtrips
- Backward: 11 fewer HBM roundtrips (fused backward kernel too)

### Expected performance improvement

Based on PR #1072 results (87ms -> 70ms/step with a similar fused kernel), we expect ~15-20% step time reduction. Even a conservative 10% improvement yields ~10% more training steps within the 590s budget.

### Triton dependency

Requires Triton (ships with PyTorch on CUDA). Falls back to standard PyTorch ops if Triton is unavailable. The RunPod `runpod/parameter-golf:latest` image includes Triton.

### Also provides: fused LeakyReLU(0.5)^2

A `fused_leaky_relu_sq(x, neg_slope)` kernel is included for future use with LeakyReLU activation variants.

## Architecture (unchanged from Approach B)

| Component | Detail |
|-----------|--------|
| Layers | 11 |
| Dimension | 512 |
| Heads | 8 query / 8 KV |
| MLP | ReLU² with fused Triton kernel, 3.5x expansion (1792 hidden) |
| Attention | XSA on all 11 layers, Partial RoPE (16/64 dims), QK-norm |
| Embeddings | BigramHash 6144 (128-dim), Value Embeddings on layers 9-10 |
| Skip connections | U-Net with learned per-dim scaling |
| Other | SmearGate, LN depth scaling, logit softcap (30.0) |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon + AdamW |
| Batch size | 786,432 tokens |
| Warmdown | 3,500 steps (wallclock-aware) |
| QAT | Late QAT at scale < 0.5 |
| EMA | 0.997 decay |
| SWA | Every 50 steps during warmdown |
| Quantization | Int5 GPTQ + zstd/zlib |
| Pruning | 10% magnitude pruning |

## Rule Compliance

- Fused kernels are a systems-only optimization (legal, no significance test needed)
- No eval changes -- identical scoring methodology
- All assertions preserved (artifact budget, wallclock budget)
- Falls back to standard PyTorch if Triton unavailable

## Credits

Built on Approach B (Int5 GPTQ + larger model). Fused kernel pattern inspired by PR #1072 (Vilhelm Toivonen).
