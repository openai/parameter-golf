# Turbo-Muon + EngramLite + ParamBanking + GPTQ Reserve Opt (val_bpb 1.1126)

**val_bpb: 1.1126** (3-seed mean, std 0.0003) | **~15.98 MB** | 8xH100 SXM, 600s train, ~120s eval

Built on [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @mikeapedia. Fused Triton MLP architecture from [PR #1072](https://github.com/openai/parameter-golf/pull/1072) by @vimeto, forward-only fusion insight from [PR #1105](https://github.com/openai/parameter-golf/pull/1105) by @abaybektursun.

## Results (8xH100 SXM, SWA applied, no TTT)

| Seed | Sliding BPB | val_loss (nats) | Artifact |
|------|-------------|-----------------|----------|
| 1337 | **1.1126** | 1.87857 | 15,981,856 |
| 42 | **1.1123** | 1.87803 | 15,984,349 |
| 999 | **1.1129** | 1.87900 | 15,985,912 |
| **Mean +/- Std** | **1.1126 +/- 0.0003** | **1.87853** | |

vs merged leaderboard SOTA ([PR #549](https://github.com/openai/parameter-golf/pull/549), 1.1194 BPB, 1.89002 nats): **-0.01149 nats** (-0.0068 BPB). Note: open PRs #1089 (1.1091) and #1105 (1.1138) achieve better scores.

## What's New vs PR #1089

### 1. GPTQ Reserve Optimization
Reduced GPTQ calibration reserve from 14s to 9s. Calibration consistently completes in ~8.4s across all runs, so 14s wastes 5+ seconds of training budget. Recovers ~55 extra training steps at ~105ms/step.

### 2. Forward-Only Fused Triton MLP Kernel Architecture
Designed a `torch.library.triton_op`-based fused kernel for `matmul + LeakyReLU(0.3) + square` with standard PyTorch backward (cuBLAS matmuls + elementwise ops). This architecture addresses two known issues:
- PR #1072's `torch.autograd.Function` crashes `torch.compile(fullgraph=True)` due to FakeTensor data pointer access
- PR #1105 showed Triton backward forces eager mode (2.7x slower)

Our solution: `triton_op` + `wrap_triton` for compile-safe forward, `register_autograd` with standard ops for backward. The kernel code is included but **hard-disabled** — it produces NaN on PyTorch 2.9 due to a TTIR analysis bug. The scored runs use the standard MLP path. This is included as experimental code for future work.

### 3. Centralized Activation Parameters
All `negative_slope` references unified via `_NEGATIVE_SLOPE = 0.3` constant with derived `_SLOPE_SQ = _NEGATIVE_SLOPE ** 2`.

## Architecture (from PR #1089)

- 11L, 512d, 8H/4KV (GQA), MLP 3.5x LeakyReLU(0.3)^2
- Turbo-Muon optimizer (AOL preconditioning + Polar Express coefficients + row_col normalization, 4 Newton-Schulz iterations)
- EngramLite hash embeddings (bigram + trigram, 2 heads, 8192 buckets)
- Parameter Banking (3D bank tensors for batched Newton-Schulz via torch.bmm)
- U-Net sigmoid-gated skip connections + ValueEmbedding (layers 9-10)
- SmearGate, Partial RoPE(16), LN Scale
- SWA (threshold=0.2, every 50 steps, 14 snapshots) + EMA(0.997) fallback
- Mixed-precision GPTQ: int5 base + selective int6/int7 promotion by Hessian sensitivity
- Brotli-11 + byte-shuffle compression
- F.scaled_dot_product_attention (auto-selects FA3 backend)

## Timing

| Phase | Time |
|-------|------|
| Training (~5,668 steps @ 104ms) | 591s |
| GPTQ calibration + quantization | 9s (reserved) |
| Sliding window eval (stride=64) | ~120s |

## Reproduction

```bash
# Use official template: runpod/parameter-golf:latest (PyTorch 2.9.1+cu128)
# Or any 8xH100 SXM pod with PyTorch >= 2.6

pip install brotli sentencepiece
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

GPTQ_RESERVE_MS=9000 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Rule Compliance

- [x] Standard F.cross_entropy scoring (softmax, sum=1)
- [x] No eval-time training data access
- [x] Artifact < 16,000,000 bytes (all 3 seeds)
- [x] Training < 600s, eval < 600s
- [x] Causal sliding-window evaluation on full validation split (stride=64)
- [x] 3-seed verification: delta = -0.01149 nats vs SOTA (> 0.005 threshold)
- [x] No n-gram caching, no external downloads during eval

## Credits

- **Turbo-Muon + EngramLite + ParamBanking**: [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @mikeapedia
- **Fused Triton MLP kernel design**: [PR #1072](https://github.com/openai/parameter-golf/pull/1072) by @vimeto
- **Forward-only fusion insight**: [PR #1105](https://github.com/openai/parameter-golf/pull/1105) by @abaybektursun
- **Base scaffold**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun
