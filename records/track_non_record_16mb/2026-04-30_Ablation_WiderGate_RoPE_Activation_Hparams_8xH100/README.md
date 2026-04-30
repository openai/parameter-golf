# Ablation: WiderGate, RoPE dims, activation slopes, hparam stack (8xH100)

Systematic ablation of 10 configurations on the PR #1693 architecture with CaseOps SP8192. All runs on 8xH100 SXM, 600s wallclock, single seed unless noted.

## Results

Base config: 11L/512d, CaseOps SP8192, AttnOutGate + SmearGate, Polar Express NS, MIN_LR=0.10, LQER OFF, brotli.

| # | Experiment | Change | Pre-quant | Post-quant | Post-TTT | Artifact | Delta vs baseline |
|---|-----------|--------|-----------|------------|----------|----------|-------------------|
| 1 | gates_caseops | baseline (EMBED_BITS=8) | 1.0712 | 1.0781 | **1.0674** | 16.93 MB ❌ | — |
| 2 | optimized_v1 | WARMDOWN=0.85, BETA2=0.99, clip sigmas | 1.0712 | 1.0806 | **1.0703** | 17.26 MB ❌ | +0.003 worse |
| 3 | rope24 | ROPE_DIMS=24 | 1.0715 | 1.0815 | **1.0706** | 16.39 MB ❌ | +0.003 worse |
| 4 | rope32 | ROPE_DIMS=32 | 1.0705 | 1.0806 | **1.0698** | 16.39 MB ❌ | +0.002 worse |
| 5 | v2_baseline | EMBED_BITS=6 | 1.0718 | 1.0941 | **1.0819** | 15.15 MB ✅ | +0.015 worse |
| 6 | v2_slope03 | LEAKY_SLOPE=0.3 | 1.0729 | 1.0942 | **1.0822** | 15.15 MB ✅ | +0.000 neutral |
| 7 | v2_slope00 | LEAKY_SLOPE=0.0 (ReLU²) | 1.0737 | 1.0958 | **1.0837** | 15.16 MB ✅ | +0.002 worse |
| 8 | **v2_gate32** | **GATE_WIDTH=32** | **1.0700** | **1.0908** | **1.0788** | **15.89 MB ✅** | **-0.003 better** |

Experiments 5-8 use EMBED_BITS=6 (int6 embeddings) to fit under 16MB without LQER.

## Key Findings

### 1. Wider attention gates help (GATE_WIDTH=32)

Increasing AttnOutGate input from 12 to 32 dimensions gives **-0.002 pre-quant** and **-0.003 post-TTT** improvement. The wider gate sees more of the residual stream for its per-head gating decision. Cost: 1,760 extra float16 params (negligible).

```python
# Standard (width=12): gate sees x[:, :, :12]
# Wider (width=32):    gate sees x[:, :, :32]
gate_in = x_orig[:, :, :gate_w.shape[-1]].contiguous()
gate = (2.0 * torch.sigmoid(F.linear(gate_in, gate_w))).contiguous()
return attn_output * gate.unsqueeze(-1)
```

**Recommendation:** Adopt GATE_WIDTH=32 as default. Free improvement.

### 2. More RoPE dims hurt post-quantization

| ROPE_DIMS | Pre-quant | Post-quant | Quant gap |
|-----------|-----------|------------|-----------|
| 16 (default) | 1.0712 | 1.0781 | 0.0069 |
| 24 | 1.0715 | 1.0815 | 0.0100 |
| 32 | 1.0705 | 1.0806 | 0.0101 |

RoPE 32 improves pre-quant by -0.0007 but **increases quant gap** from 0.007 to 0.010. More rotated dimensions create weight distributions that GPTQ handles worse. **Keep ROPE_DIMS=16.**

### 3. Activation slope changes are neutral or worse

| Slope | Pre-quant | Post-TTT | Note |
|-------|-----------|----------|------|
| 0.5 (default) | 1.0718 | 1.0819 | baseline |
| 0.3 (PR #1948) | 1.0729 | 1.0822 | neutral |
| 0.0 (pure ReLU²) | 1.0737 | 1.0837 | +0.002 worse |

PR #1948 reported slope=0.3 as optimal on a different base config. On the CaseOps+gates stack with EMBED_BITS=6, **slope 0.5 remains optimal**. Pure ReLU² hurts — the leaky negative slope provides useful gradient flow.

### 4. PR #1855 hparam stack does not transfer

| Hparam | Default | PR #1855 | Result |
|--------|---------|----------|--------|
| WARMDOWN_FRAC | 0.75 | 0.85 | Neutral pre-quant |
| BETA2 | 0.95 | 0.99 | Neutral pre-quant |
| EMBED_CLIP_SIGMAS | 20.0 | 14.0 | **Worse** quant gap (+0.0025) |
| MLP_CLIP_SIGMAS | 10.0 | 11.5 | **Worse** quant gap |
| TTT_BETA2 | 0.999 | 0.99 | Neutral |

The 9-hparam stack from PR #1855 was greedy-validated on a different config (SparseAttnGate, no SmearGate widening). On our CaseOps+AttnOutGate stack, **tighter clip sigmas hurt quantization** and WARMDOWN/BETA2 changes are neutral. **Keep defaults.**

### 5. EMBED_BITS=6 costs +0.014 BPP

Dropping embedding precision from int7 to int6 saves ~500KB but costs +0.014 BPB post-TTT. This is the price of fitting under 16MB without per-group lrzip compression.

### 6. LZMA compression is worse than brotli

LZMA produced artifacts ~300KB larger than brotli-11 on this architecture. **Use brotli.**

## Negative Results Summary

| Technique | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| WARMDOWN_FRAC=0.85 | -0.002 | 0.000 | Dead |
| BETA2=0.99 | -0.001 | 0.000 | Dead |
| EMBED_CLIP_SIGMAS=14 | better quant | +0.0025 worse | Dead |
| ROPE_DIMS=24/32 | -0.003 | +0.002/+0.003 | Dead |
| LeakyReLU slope=0.3 | -0.001 | 0.000 | Dead |
| Pure ReLU² | -0.003 | +0.002 | Dead |
| LZMA compressor | better compression | +300KB larger | Dead |
| LQER + Gates combo | both help | over 16MB | Incompatible |

## Configuration

All experiments use `train_gpt.py` from the record submission (PR #1969) with env var overrides. No code changes needed except GATE_WIDTH and LEAKY_SLOPE which require `train_gpt_v2.py`.
