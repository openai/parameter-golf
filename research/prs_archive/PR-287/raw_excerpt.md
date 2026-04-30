# PR 287 — 11L XSA + EMA + Int6 MLP3x + WD=0.04

**Author:** Jack Princz (jfprincz)
**Branch created:** 2026-03-20
**Claimed BPB:** 1.12707 (sliding window stride=64, seed 1337); mean 1.1280 across 3 seeds
**Artifact size:** 15,534,645 bytes (15.5 MB, int6+zstd-22)
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_10min_16mb__2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271__README.md`
- `records__track_10min_16mb__2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271__submission.json`
- `records__track_10min_16mb__2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271__train_gpt.py`

## Claimed changes (from README, verbatim)

> ### What's new
> 1. Exclusive Self Attention (XSA) on last 4 layers. After the standard attention output, XSA subtracts the component aligned with each token's own value vector using an efficient GQA-aware reshape (no repeat_interleave). This encourages attention to capture only information orthogonal to what the token already knows, improving context modeling. Zero new parameters, ~2ms/step overhead.
> 2. EMA replacing SWA. Instead of collecting periodic SWA checkpoints during warmdown, we maintain an exponential moving average shadow model on GPU that updates every step: `ema = 0.997 * ema + 0.003 * param`. The EMA weights are used for quantization and eval. Smoother averaging than periodic SWA, better generalization and artifact compression.
> ### Carried from PR #198
> - 11 transformer layers with U-Net skip connections
> - Orthogonal + muP-scaled init, 3x MLP (hidden=1536), relu² activation
> - Int6 mixed quantization + zstd-22 (int6 on MLP+attention, int8 on embeddings)
> - Weight decay 0.04 (Muon + AdamW)
> - SmearGate, Bigram Hash Embedding (2048-bucket, dim=128), FlashAttention 3
> - Sequence length 2048 with NTK-aware RoPE
> - Muon optimizer, momentum 0.99 with warmup, warmdown 3000 iters, grad clip 0.3

Pre-quant 1.1427; int6 roundtrip 1.1494; 7103 steps at 84ms/step on 8xH100. XSA_LAST_N=4, EMA_DECAY=0.997.
