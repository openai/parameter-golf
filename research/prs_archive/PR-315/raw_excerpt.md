# PR 315 — 11L Partial RoPE + LN Scale + EMA + XSA4

**Author:** Jack Princz (jfprincz)
**Branch created:** 2026-03-21
**Claimed BPB:** 1.12485 (sliding window stride=64, seed 2025); mean 1.1250 across 3 seeds
**Artifact size:** 15,612,308 bytes (15.6 MB, int6+zstd-22)
**Seeds:** 2025, 42, 1337

## Files retrieved
- `records__track_10min_16mb__2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248__README.md`
- `records__track_10min_16mb__2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248__submission.json`
- `records__track_10min_16mb__2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248__train_gpt.py`

## Claimed changes (from README, verbatim)

> ### What's new
> 1. Partial RoPE (16 of 64 dims). Rotary position embeddings applied to only the first 16 of 64 head dimensions (25%). The remaining 48 dims attend without positional bias, allowing the model to learn position-invariant patterns. Zero new parameters.
> 2. LN Scale. RMSNorm outputs are scaled by 1/sqrt(layer_idx+1), damping deeper layers' contributions. Stabilizes training and improves convergence in deep models. Zero new parameters.
> ### Carried from PR #287
> - 11 transformer layers with U-Net skip connections
> - Exclusive Self Attention (XSA) on last 4 layers
> - EMA weight averaging (decay=0.997, every step)
> - Orthogonal + muP-scaled init, 3x MLP (hidden=1536), relu² activation
> - Int6 mixed quantization + zstd-22 (int6 on MLP+attention, int8 on embeddings)
> - Weight decay 0.04 (Muon + AdamW)
> - SmearGate, Bigram Hash Embedding (2048-bucket, dim=128), FlashAttention 3
> - Muon optimizer, momentum 0.99 with warmup, warmdown 3000 iters, grad clip 0.3
> ### Note on Late QAT
> The submitted code includes a Late QAT flag (LATE_QAT=1) intended to enable STE int6 fake-quantization in the final 4% of training. Post-submission analysis (credit: @152334H) revealed that torch.compile constant-folds the CastedLinear._qat_enabled class attribute at first trace, so the STE branch is dead-code-eliminated and never activates during training. Late QAT had no effect on the results. The score is driven entirely by Partial RoPE and LN Scale.

Pre-quant 1.1418; int6 roundtrip 1.1485; 7051 steps at 85ms/step on 8xH100. Env: ROPE_DIMS=16, LN_SCALE=1.
