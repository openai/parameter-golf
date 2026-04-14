# PR 1218 — 4096-Vocab + Larger Model + High WD + Simplifications

**Author:** Kevin Clark (clarkkev)
**Claimed BPB:** 1.09785 (3-seed mean, std=0.0004) — sliding BPB
**Artifact size:** 15,916,170 bytes (mean)
**Seeds:** 42, 1337, 2025
**Hardware:** 8xH100 80GB SXM, PyTorch 2.11.0+cu130

## Files retrieved
- `records__track_10min_16mb__2026-04-01_Vocab4096_MLPMult4_WD085__README.md`
- `records__track_10min_16mb__2026-04-01_Vocab4096_MLPMult4_WD085__submission.json`
- `records__track_10min_16mb__2026-04-01_Vocab4096_MLPMult4_WD085__train_gpt.py`

## Run command (from README)
```
RUN_ID=1337 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)
Fixes:
- Fixed a small bug in the sliding window evaluation causing it to score tokens at the end of the val dataset multiple times.

Simplifications:
- Use XSA in all layers instead of only the last 4.
- Removed parameter banking and distributed muon implementation and instead just used Muon + DDP.
- Removed test time training.
- Removed quantization-aware training.
- Removed gated attention.
- Removed value residuals.
- Removed hash embeddings.
- Removed the smear gate.

Additions:
- Increased the vocabulary size from 1024 to 4096 (sentencepiece).
- Bigger but more strongly regularized model: muon WD 0.04 -> 0.085, embeddings WD added at 0.085, adam WD 0.04 -> 0.02, mlp_mult 3 -> 4, LR 0.025 -> 0.02.
- Added the coprime-stride data loader from #726.
- Added GPTQ Hessian-aware quantization (based on #1060).
- Byte shuffle + brotli compression from #1089.
- Sigmoid-gated skip connections to the unet from #1089.
- Increased qk_gain_init 1.5 -> 4 following #1125.
