# Experiment: Int5 GPTQ + Wider MLP

**Status:** Not yet run — awaiting GPU time

**Hypothesis:** Switching matrix quantization from int6 to int5 frees enough parameter budget to widen the MLP from 4x to 5x. The extra model capacity should outweigh the increased quantization error, yielding lower BPB than the current SOTA (1.0810).

## Changes from SOTA (2026-04-09 SP8192 3LayerRecur ParResid QK525 LegalTTT)

| Parameter | SOTA | This experiment | Rationale |
|---|---|---|---|
| `matrix_bits` | 6 | **5** | Reduces per-weight storage from 6 to 5 bits, freeing ~16% of matrix budget |
| `mlp_mult` | 4.0 | **5.0** | Uses freed budget: MLP hidden dim 2048 -> 2560 (+25% capacity) |
| `matrix_clip_sigmas` | 12.85 | **9.0** | int5 has clip_range=15 vs int6's 31; lower sigma reduces rounding error at the cost of more clipping. Starting guess, needs tuning. |

Everything else is identical to SOTA: 11 layers, 512 dim, 8 heads / 4 KV heads, SP8192 vocab, depth recurrence (layers 3-5, 2 loops giving 17 virtual layers), parallel residuals from layer 7, QK-gain 5.0, XSA on all layers, skip gates, partial RoPE (16/64 dims), LeakyReLU(0.5)^2, EMA 0.9965, MuonEq-R optimizer, legal score-first TTT.

## Parameter budget math

**SOTA (int6, MLP 4x):**
- Per-layer matrix params: Q(512x512) + K(512x256) + V(512x256) + O(512x512) + MLP_fc(512x2048) + MLP_proj(2048x512) = 2,883,584
- 11 layers: 31,719,424 matrix params at 6 bits = ~23.8MB raw
- Embedding: 8192x512 = 4,194,304 at 8 bits = ~4.2MB raw
- After GPTQ SDClip + brotli: ~15.99MB

**This experiment (int5, MLP 5x):**
- Per-layer matrix params: same attention + MLP_fc(512x2560) + MLP_proj(2560x512) = 3,407,872
- 11 layers: 37,486,592 matrix params at 5 bits = ~23.4MB raw
- Embedding: unchanged at ~4.2MB raw
- Expected compressed size: similar or slightly under 16MB (needs verification)

Net: ~18% more matrix parameters, ~2% less raw storage. The compression ratio with brotli may shift — int5 quantized values have lower entropy (16 levels vs 32), which could compress slightly better.

## Key risk

Int5 has only 16 quantization levels per row-scaled value (±15), vs int6's 32 (±31). GPTQ's Hessian-weighted error correction helps, but there is a floor to how well it can compensate. If the quantization noise is too large, the extra MLP capacity won't help.

## Tuning plan

Priority order for sweeps (if first run shows promise):

1. **`MATRIX_CLIP_SIGMAS`** — Most critical. Try: 7.0, 8.0, 9.0, 10.0, 11.0. This controls the clipping-vs-rounding tradeoff.
2. **`MLP_MULT`** — If model exceeds 16MB, reduce to 4.5. If headroom, try 5.5 or 6.0.
3. **`QK_GAIN_INIT`** — May need re-tuning for wider MLP. Try 4.5, 5.0, 5.25, 5.5.
4. **`MUON_WD`** — Wider model may benefit from different regularization.

## How to run

```bash
# Download data (delete stale manifest first — default repo doesn't have sp8192)
rm -f data/datasets/manifest.json data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Single seed run
SEED=42 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-13_Int5_WiderMLP/train_gpt.py

# Quick iteration without TTT (faster eval)
SEED=42 TTT_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-13_Int5_WiderMLP/train_gpt.py

# Sweep clip sigmas
for CS in 7.0 8.0 9.0 10.0 11.0; do
  SEED=42 TTT_ENABLED=0 MATRIX_CLIP_SIGMAS=$CS RUN_ID="int5_cs${CS}" \
    torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-13_Int5_WiderMLP/train_gpt.py
done
```

## What to look for in logs

1. **`model_params:`** — Should be ~41.7M total (vs SOTA's ~35.9M)
2. **`Total submission size`** — Must be under 16,000,000 bytes. If over, reduce `MLP_MULT`.
3. **`pre-quantization post-ema val_bpb`** — Float model quality before quantization. Should be lower than SOTA's pre-quant BPB due to wider MLP.
4. **`quantized val_bpb`** — The quantization gap (pre-quant minus post-quant). If this gap is large, clip_sigmas needs tuning.
5. **`quantized_sliding_window val_bpb`** — The score that matters for comparison.
6. **`quantized_ttt val_bpb`** — Final score with TTT. Target: < 1.0810.

## Credits

Based on the SOTA stack by @bigbag (PR #1493), building on work by @clarkkev, @dexhunter, @abaybektursun, @Robby955, @msisovic, @X-Abhishek-X.
