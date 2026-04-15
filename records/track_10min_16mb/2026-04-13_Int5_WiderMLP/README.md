# Negative Result: Int5 GPTQ + Wider MLP

**val_bpb = not competitive** | 8xH100 SXM | Non-record submission (negative result)

**Hypothesis:** Switching matrix quantization from int6 to int5 frees enough parameter budget to widen the MLP (4x -> 4.8x). The extra model capacity should outweigh the increased quantization error.

**Result:** Int5 quantization error is ~2x worse than int6, which negates any capacity gains. This approach is a dead end under current GPTQ.

## Experimental results

### Run 1: Int5 + MLP 4.8x (bigger model)

| Metric | SOTA (int6, MLP 4x) | Int5, MLP 4.8x |
|---|---|---|
| model_params | 35,944,536 | 40,551,512 |
| steps in 10 min | 4,550 | 3,509 |
| pre-quant val_bpb | 1.0873 | 1.0887 |
| compressed size | 15.99 MB | 15.70 MB |

The bigger model is slower per step (3,509 vs 4,550 steps), so the extra capacity barely compensates for fewer training steps. Pre-quant BPB is nearly identical.

### Run 2: Int5 + MLP 4.0x (same model, quantization-only ablation)

| Metric | SOTA (int6, MLP 4x) | Int5, MLP 4.0x |
|---|---|---|
| model_params | 35,944,536 | 35,944,536 |
| steps in 10 min | 4,550 | 4,654 |
| pre-quant val_bpb | 1.0873 | **1.0870** |
| quantized val_bpb | 1.0997 | 1.1109 |
| sliding_window val_bpb | 1.0829 | 1.0942 |
| **TTT val_bpb** | **1.0808** | **1.0909** |
| **quantization gap** | **0.0124** | **0.0239** |
| compressed size | 15.99 MB | 13.99 MB |

This is a clean ablation: identical model, only quantization bitwidth differs. Pre-quant BPB is essentially the same (1.0870 vs 1.0873), confirming training is unaffected. But int5 quantization gap is **2x worse** (0.024 vs 0.012 BPP). The penalty persists through sliding window and TTT eval: final BPB is 1.0909 vs SOTA's 1.0808 — a **0.010 BPB regression**.

## Why int5 fails

Int5 has only 16 quantization levels per row (clip_range=15), vs int6's 32 (clip_range=31). Even with GPTQ's Hessian-weighted error correction and SDClip, the rounding error is fundamentally larger. The 2MB of compressed size savings from int5 cannot buy enough extra model capacity to overcome the quantization penalty, because:

1. **Wider MLP costs training steps** — bigger model = slower per step = fewer steps in 10 min
2. **Quantization gap is intrinsic** — 16 levels can't represent the weight distribution as well as 32, regardless of clip_sigmas tuning

The SDClip formula `compressed_size ~ b - log2(k) + const` correctly predicts the size savings, but doesn't account for the non-linear increase in reconstruction error at low bitwidths.

## Configuration

Based on the SOTA stack (PR #1493). Architecture: 11 layers, 512 dim, 8 heads / 4 KV heads, SP8192 vocab, depth recurrence (layers 3-5, 2 loops, 17 virtual layers), parallel residuals from layer 7, QK-gain 5.0, XSA, skip gates, partial RoPE (16/64), LeakyReLU(0.5)^2, EMA 0.9965, MuonEq-R, legal score-first TTT.

Only changes: `MATRIX_BITS=5`, `MATRIX_CLIP_SIGMAS=9.0`, and `MLP_MULT` (4.0 or 4.8).

## How to reproduce

```bash
rm -f data/datasets/manifest.json data/manifest.json
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Int5 + MLP 4.0 (clean ablation)
SEED=42 TTT_ENABLED=0 MATRIX_BITS=5 MATRIX_CLIP_SIGMAS=9.0 MLP_MULT=4.0 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-13_Int5_WiderMLP/train_gpt.py

# Int5 + MLP 4.8 (bigger model)
SEED=42 TTT_ENABLED=0 MATRIX_BITS=5 MATRIX_CLIP_SIGMAS=9.0 MLP_MULT=4.8 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-13_Int5_WiderMLP/train_gpt.py
```

## Credits

Based on the SOTA stack by @bigbag (PR #1493), building on work by @clarkkev, @dexhunter, @abaybektursun, @Robby955, @msisovic, @X-Abhishek-X.

## Included files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
