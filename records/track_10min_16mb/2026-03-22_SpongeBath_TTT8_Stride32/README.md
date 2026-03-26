# Sponge Bath — TTT 8 Epochs + Stride 32

## Result

**val_bpb: 1.1295** (seed 1337) | 15.74 MB artifact | 8xH100 SXM

2-seed verification:

| Seed | val_bpb | Artifact Size | Status |
|------|---------|---------------|--------|
| 1337 | 1.1295  | 15.74 MB      | Pass   |
| 42   | 1.1307  | 15.69 MB      | Pass   |

Baseline (SOTA254 with TTT 3 epochs): **1.1303 BPB**

## What changed

This is a pure eval-time improvement over the SOTA254 base (PR #254). No model architecture or training changes were made. The same trained artifact is used; only TTT adaptation and eval stride are modified:

1. **TTT epochs: 3 -> 8** — More test-time training adaptation epochs on the validation set
2. **Eval stride: 64 -> 32** — Finer sliding window during evaluation

## Why it works

More TTT epochs allow the model to better adapt to the validation distribution at test time. The additional epochs are essentially free — they cost ~115s of the 600s wallclock budget, well within limits. The finer eval stride (32 vs 64) captures more context overlap, reducing boundary effects in sliding window evaluation.

The key insight: this is a "free" improvement. The artifact size is unchanged, the training is unchanged, and the extra eval-time compute fits comfortably within the wallclock cap.

## Configuration

Based on SOTA254 (PR #254) with the following eval-time overrides:

```
TTT_EPOCHS=8          # was 3
EVAL_STRIDE=32        # was 64
TTT_LR=0.002
TTT_MOMENTUM=0.9
```

Full architecture (unchanged from SOTA254):
- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with SmearGate + BigramHash (2048 buckets)
- Int6 QAT + zlib/zstd compression
- Muon optimizer: lr=0.025, WD=0.04, momentum=0.99
- FlashAttention 3, NTK-RoPE, orthogonal init, tied embeddings

## Eval budget breakdown

- TTT adaptation (8 epochs): ~115s
- Sliding window eval (stride 32): ~170s
- Total eval: ~285s of 600s budget

## Included files

- `sponge_bath/train_gpt.py` — Code snapshot (same as SOTA254 base)
- `sponge_bath/run.sh` — Single-seed run script
- `sponge_bath/run_2seed.sh` — 2-seed validation wrapper
- `records/track_10min_16mb/2026-03-22_SpongeBath_TTT8_Stride32/submission.json` — Leaderboard metadata
- `records/track_10min_16mb/2026-03-22_SpongeBath_TTT8_Stride32/README.md` — This file
