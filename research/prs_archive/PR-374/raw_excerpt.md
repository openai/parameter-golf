# PR 374 — Record: 11L + Tight SWA + Shared VE128 + Partial RoPE + LN Scale + XSA4

**Author:** vadim borisov (unnir / tabularis.ai)
**Claimed BPB:** val_bpb 1.12461967 (val_loss 1.89886818), seed 1337
**Artifact size:** 15,706,024 bytes
**Seeds:** 1337 (single seed submitted)

## Files retrieved
- `records__track_10min_16mb__2026-03-21_v38_TightSWA__README.md`
- `records__track_10min_16mb__2026-03-21_v38_TightSWA__submission.json`
- `records__track_10min_16mb__2026-03-21_v38_TightSWA__train_gpt.py`

## Claimed changes (from README, verbatim)

> NEW SOTA — beats previous record of 1.1248
>
> Key Innovation: Tight SWA
> SWA checkpoint collection restricted to scale<0.2 (last ~600 steps), every 50 steps. This eliminates the SWA quality penalty (post-SWA BPB = pre-SWA BPB) while maintaining quantization-friendly weight averaging. Standard SWA (scale<0.5) averages stale checkpoints that hurt final quality.
>
> Architecture: 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA); 3x MLP expansion with relu-squared; Efficient Partial XSA on last 4 layers (GQA-aware, zero-alloc); Partial RoPE (16/64 dims) + NTK-aware scaling; LN Scale Factor 1/sqrt(layer_idx+1); U-Net skip connections (5 encoder, 6 decoder); SmearGate + BigramHash (2048 buckets, dim=128); Shared Value Embedding (dim=128, layers 9,10) — 1 table, per-layer learned scales; FlashAttention 3 (Hopper); Orthogonal init with proj scaling by 1/sqrt(2*num_layers); Logit softcap 30.0, tied embeddings.
>
> Training: Muon (matrices) lr=0.025 momentum=0.99 (warmup 0.92→0.99 over 1500 steps) WD=0.04; AdamW (embeddings) lr=0.035, (scalars) lr=0.025, WD=0.04; grad clip 0.3; batch 786,432 tokens/step seq_len=2048; Warmdown 3000 iters (wallclock-based); Tight SWA every 50 steps when scale<0.2 (12 checkpoints from last 600 steps); Late QAT STE int6 fake-quantization when LR scale<0.1.
>
> Quantization: Int6 per-row MLP+attention; Int8 per-row embeddings; control tensors fp32; zstd level 22.
>
> Results: 6942 steps in 600s at 86.4ms/step; Pre-quant val_bpb 1.1407; Post-SWA val_bpb 1.1407 (zero SWA penalty); Quant gap 0.008.
