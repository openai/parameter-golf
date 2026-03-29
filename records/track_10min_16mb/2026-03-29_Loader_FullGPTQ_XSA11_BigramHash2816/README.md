# Record: Coprime-Stride Loader + Full Hessian GPTQ + XSA-all (val_bpb 1.1123)

**val_bpb: 1.1123** (3-seed mean, std 0.0005) | **~15.99 MB** | 8×H100 SXM, 600s train, ~87s eval

Built on [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun.

## Results (8×H100 SXM, no TTT)

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 1337 | **1.1119** | 15,987,110 |
| 42 | **1.1129** | 15,991,086 |
| 2025 | **1.1121** | 15,997,694 |
| **Mean ± Std** | **1.1123 ± 0.0005** | |

## What's New

This submission combines training-side data diversity with quantization-side Hessian error compensation on the PR #549 scaffold. To our knowledge, this is the first submission to stack both improvements together.

### 1. Coprime-Stride Multi-Shard Data Pipeline
Inspired by [PR #726](https://github.com/openai/parameter-golf/pull/726) by @DeepReinforce. Instead of reading training shards sequentially, the loader:
- Samples blocks from multiple shards per batch using coprime strides
- Each shard is visited with a stride coprime to its block count, ensuring every block is seen before repeating
- Provides significantly more diverse batches than the standard sequential loader
- Zero step time overhead (87ms/step, same regime as baseline)

### 2. Full Hessian GPTQ
Inspired by [PR #634](https://github.com/openai/parameter-golf/pull/634) by @raahilshah and [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun. Replaces GPTQ-lite with Full Hessian GPTQ (Frantar et al., ICLR 2023):
- Cholesky error compensation with column reordering
- 64-batch calibration within 14s reserved training budget
- Reduces quantization error vs GPTQ-lite diagonal approximation

### 3. XSA on All 11 Layers
Extended Exclusive Self-Attention from last 4 layers to all 11 layers. Zero new parameters.

### 4. BigramHash(2816×112)
Enlarged from PR #549's BigramHash(1536×128). Captures more bigram patterns with a slightly narrower projection.

### 5. No TTT
Score-first TTT was tested but found to be neutral or slightly negative on this stack. The sliding window result consistently outperforms the TTT result. This confirms [PR #1019](https://github.com/openai/parameter-golf/pull/1019)'s finding that TTT becomes unnecessary with stronger quantization.

## Architecture

PR #549 stack with modifications:
- 11L, 512d, 8H/4KV (GQA), MLP 3x LeakyReLU(0.5)²
- XSA on all 11 layers, BigramHash(2816×112), SmearGate
- Partial RoPE (16d), LN Scale, EMA(0.997)
- Full Hessian GPTQ int6 + LZMA compression
- Parallel Muon + Parameter Banking, FA3 Hopper

## Timing

| Phase | Time |
|-------|------|
| Training (6,816 steps @ 87ms) | 586s |
| GPTQ calibration + quantization | 14s (reserved from training) |
| Sliding window eval (stride=64) | 87s |
| **Total eval** | **~87s** |

## Rule Compliance

- ✅ Standard F.cross_entropy scoring (softmax, sum=1)
- ✅ No TTT, no mixer, no cache, no unnormalized scoring
- ✅ Artifact < 16,000,000 bytes (all 3 seeds)
- ✅ Training < 600s, eval < 600s
- ✅ Single left-to-right evaluation pass

## Credits

- **Base scaffold**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun (LeakyReLU² + Parallel Muon)
- **Data pipeline ideas**: [PR #726](https://github.com/openai/parameter-golf/pull/726) by @DeepReinforce (coprime-stride loader)
- **Full Hessian GPTQ**: [PR #634](https://github.com/openai/parameter-golf/pull/634) by @raahilshah, [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun
- **XSA**: [PR #287](https://github.com/openai/parameter-golf/pull/287) by @jfprincz
