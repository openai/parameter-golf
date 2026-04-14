# PR 1395 — SP4096 + Linear LR Decay + Depth Recurrence + MuonEq-R

**Author:** dttdrv
**Claimed BPB:** 1.09244346 (3-seed mean, std 0.00043499); pre-quant 1.0974
**Artifact size:** ~15.99 MB (mean 15,988,225; max 15,989,311)
**Seeds:** [42, 314, 999] — [1.09269834, 1.09269085, 1.09194120]
**Track:** 10min_16mb
**Hardware:** 8xH100 80GB SXM, 600s, PyTorch 2.9.1+cu128
**Date:** 2026-04-06
**Baseline:** PR #1019 (1.1147), delta -0.0223, Welch t=-68.85

## Files retrieved
- `records__track_10min_16mb__2026-04-06_SP4096_LinearLR_DepthRecurrence_1.0924__README.md`
- `records__track_10min_16mb__2026-04-06_SP4096_LinearLR_DepthRecurrence_1.0924__submission.json`
- `records__track_10min_16mb__2026-04-06_SP4096_LinearLR_DepthRecurrence_1.0924__train_gpt.py`

## Environment variables (from run command)
`DATA_DIR=./data`, `VOCAB_SIZE=4096`, `SEED=42`. torchrun standalone nproc_per_node=8.

## Claimed changes (from README, verbatim)
"The single critical change: Linear LR Decay to Zero. Prior cosine warmdown floored at 5% of peak LR; replaced with linear warmdown to zero. Measured impact: Quantization gap (roundtrip) 0.038 -> 0.014 BPB (-61%); Values pruned to fit 16MB 1,860,936 -> 340,142 (-82%); Unpruned artifact 16.23 -> 16.09 MB; Post-quant sliding BPB 1.1124 -> 1.0924. Also: Reduced Selective Pruning Factor from excess*8 to excess*4. Architecture: 11L 512d, 8 GQA heads, 4 KV heads; MLP 4x (2048) LeakyReLU(0.5)^2; SP4096 tokenizer; Depth Recurrence layers 4,5 from step 3000; Parallel Residuals from layer 7; XSA all 11 layers; QK-Gain 5.0; Partial RoPE 16/64; LN Scale 1/sqrt(layer+1); U-Net gated skips; SmearGate; MuonEq-R; Selective +/-1 pruning; Full Hessian GPTQ; EMA decay 0.997; FA3. Training: Muon matrix LR 0.02, Muon WD 0.09, Adam WD 0.02, momentum 0.99 (warmup from 0.92 over 1500 steps), Warmdown fraction 0.667 linear to 0, grad clip 0.3, batch 786,432 tokens/step, recurrence activated at step 3000. Quantization: int6 per-row matrix, int8 per-row embeddings, Brotli quality=10. No TTT, no SLOT, no n-gram."
