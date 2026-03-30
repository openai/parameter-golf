# Record: Coprime-Stride Loader + Full Hessian GPTQ + XSA-all + Optimized GPTQ Reserve (val_bpb 1.1136)

**val_bpb: 1.1133** (3-seed mean, std 0.0001) | **~15.89 MB** | 8xH100 SXM, 600s train, ~85s eval

Built on [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun and [PR #1060](https://github.com/openai/parameter-golf/pull/1060) by @dexhunter.

## Results (8xH100 SXM, no TTT)

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 1337 | **1.1133** | 15,899,687 |
| 42 | **1.1132** | 15,881,359 |
| 999 | **1.1133** | 15,892,371 |
| **Mean +/- Std** | **1.1133 +/- 0.0001** | |

## What's New

This submission extends PR #1060's coprime-stride loader + Full Hessian GPTQ stack with two targeted improvements:

### 1. GPTQ Reserve Optimization
Reduced GPTQ calibration reserve from 14s to 9s. PR #1060's GPTQ calibration completes in ~8.4s, so 14s wastes ~4s of training budget. This recovers ~44 additional training steps at 91ms/step, translating to measurable BPB improvement.

### 2. FA3/FA2 Graceful Fallback
Added try/except import for `flash_attn_interface` (FA3) with fallback to `flash_attn` (FA2). Allows the same script to run on pods with or without FA3 Hopper kernels built.

### 3. FP32 SWA Accumulation (experimental, not used in final)
Fixed SWA accumulation to use FP32 instead of model dtype (BF16). A/B testing showed EMA(0.997) still outperforms SWA on this stack by ~0.0006 BPB, so EMA is used for the final submission.

## Architecture

PR #549 + PR #1060 stack:
- 11L, 512d, 8H/4KV (GQA), MLP 3x LeakyReLU(0.5)^2
- Coprime-stride multi-shard data pipeline
- XSA on all 11 layers, BigramHash(2816x112), SmearGate
- Partial RoPE (16d), LN Scale, EMA(0.997)
- Full Hessian GPTQ int6 + LZMA compression (10s reserve)
- Parallel Muon + Parameter Banking, FA3 Hopper

## Timing

| Phase | Time |
|-------|------|
| Training (~6,479 steps @ 91ms) | 590s |
| GPTQ calibration + quantization | 9s (reserved from training) |
| Sliding window eval (stride=64) | ~85s |
| **Total eval** | **~85s** |

## Env Vars (overrides from defaults)

```
BIGRAM_VOCAB_SIZE=2816
BIGRAM_DIM=112
XSA_LAST_N=11
USE_GPTQ=1
GPTQ_RESERVE_MS=9000
WARMDOWN_ITERS=4000
SWA_APPLY=0
TTT_ENABLED=0
```

## Rule Compliance

- Standard F.cross_entropy scoring (softmax, sum=1)
- No TTT, no mixer, no eval-built adaptation, no unnormalized scoring
- Full `fineweb_val_*` split in canonical sorted order with tokenizer-derived byte accounting
- Artifact < 16,000,000 bytes (all 3 seeds)
- Training < 600s, eval < 600s
- Causal sliding-window evaluation on the full validation split (stride=64)

## Reproduction

```bash
# From this records folder (with data symlinked):
SEED=1337 \
BIGRAM_VOCAB_SIZE=2816 \
BIGRAM_DIM=112 \
XSA_LAST_N=11 \
USE_GPTQ=1 \
GPTQ_RESERVE_MS=9000 \
WARMDOWN_ITERS=4000 \
SWA_APPLY=0 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment: PyTorch 2.6+, Flash Attention 3 (`flash_attn_interface`), 8xH100 SXM.

## Credits

- **Base scaffold**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun (LeakyReLU^2 + Parallel Muon)
- **Coprime-stride loader + Full GPTQ + XSA-all**: [PR #1060](https://github.com/openai/parameter-golf/pull/1060) by @dexhunter
- **Data pipeline ideas**: [PR #726](https://github.com/openai/parameter-golf/pull/726) by @DeepReinforce
- **Full Hessian GPTQ**: [PR #634](https://github.com/openai/parameter-golf/pull/634) by @raahilshah
- **XSA**: [PR #287](https://github.com/openai/parameter-golf/pull/287) by @jfprincz
