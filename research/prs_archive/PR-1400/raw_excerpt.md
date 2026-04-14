# PR 1400 — Hadamard-Rotated GPTQ + Discriminative TTT + Depth Recurrence

**Author:** Tommy Mancino (tmancino)
**Claimed BPB:** 1.10352022 (3-seed exact mean, std 0.00037859)
**Artifact size:** ~15,878,628 bytes mean
**Seeds:** 271, 503, 999

## Files retrieved
- `records__track_10min_16mb__2026-04-05_Hadamard_dTTT_Recur2__README.md`
- `records__track_10min_16mb__2026-04-05_Hadamard_dTTT_Recur2__train_gpt.py`
- `records__track_10min_16mb__2026-04-05_Hadamard_dTTT_Recur2__submission.json`
- `records__track_non_record_16mb__2026-04-01_WhirlpoolV5b_LorentzianTransformer__README.md`
- `records__track_non_record_16mb__2026-04-01_WhirlpoolV5b_LorentzianTransformer__submission.json`

## Environment variables (from run command)
SEED=271, MAX_WALLCLOCK_SECONDS=600

## Claimed changes (from README, verbatim)
> Builds directly on PR #1019 codebase (11L GQA, BigramHash, XSA-all, AR self-gen GPTQ, sliding window eval) and adds three techniques:
>
> 1. MR-GPTQ: Hadamard Rotation Before GPTQ. Apply a block-diagonal Walsh-Hadamard transform to the weight matrix columns. Spreads outlier energy uniformly. 68× reduction in reconstruction MSE vs bare GPTQ at int6. −0.015 BPB improvement from rotation alone. Zero artifact overhead. Block-diagonal design handles non-power-of-2 dimensions. Inspired by MR-GPTQ (ICLR 2026) and PolarQuant.
>
> 2. Discriminative TTT (dTTT): per-block adaptive LR, later blocks 1.0×, earlier blocks 0.3×, cosine decay over 10 epochs. −0.037 BPB pre-quant improvement. Adapted from PR #1351.
>
> 3. Depth Recurrence (2 Layers): Re-runs last 2 transformer layers giving 13 effective layers from 11 stored parameters. Reduced from 3 to 2 layers for ~10% faster step time.
>
> 4. Selective ±2 Pruning with LZMA: Extended selective pruning from ±1 to ±1/±2, weighted by magnitude × scale². Binary search over LZMA-compressed artifact size.
>
> Other: WD 0.03 (from 0.04), Warmdown 3500, ASQU per-channel scaling.
