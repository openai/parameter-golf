# Leapfrog Experiment Results — 2026-03-22

Target: Beat PR #414 (1.1233 BPB, 15.55 MB)

## Results Summary

| Variant | Description | Sliding BPB (s64) | Size | Verdict |
|---------|-------------|-------------------|------|---------|
| v1 seed 1337 | TTT burst (2ep, 10% LR, before EMA) | **1.12319** | 15.68 MB | WINNER |
| v1 seed 42 | Same as above, different seed | 1.12397 | 16.37 MB | Over size |
| v1b seed 1337 | EMA-first, then burst (1ep, 5% LR, QAT) | 1.12624 | 15.97 MB | Worse BPB |
| v1c seed 1337 | Burst+QAT before EMA + 15 GPTQ percentiles | 1.12319 | 15.68 MB | Same as v1 |
| v2 seed 1337 | Self-distillation (50 steps, KL+CE) | 1.12328 | 15.62 MB | ~Tied with v1 |
| v4 seed 1337 | Burst + distill + train_seq_len=1024 | 1.22243 | 15.53 MB | BUST |

## Key Findings

1. **TTT burst before EMA works** — replaying 100 recent batches for 2 epochs at 10% LR, with EMA updates, then applying EMA. Gives ~0.0001 over baseline.

2. **Self-distillation matches burst** — using EMA as teacher with KL+CE loss lands in the same spot. Both approaches hit the same ceiling.

3. **Stacking burst + distill doesn't help** — the two techniques capture the same signal.

4. **EMA-first then burst is worse** — the burst needs to happen before EMA so EMA can smooth the sharpened weights.

5. **15 GPTQ percentiles = no gain over 5** — the original 5 percentiles already find near-optimal clips.

6. **train_seq_len=1024 is catastrophic** — only 6% more steps but massive quality loss. Partial RoPE extrapolation from 1024→2048 is not good enough.

7. **zlib vs zstd matters for size, not BPB** — same quantization, different compression. zstd-22 saves ~1.3MB.

| v5 seed 1337 | QAT percentile fix + TrigramHash + EMA-SWA blend | 1.12439 | 15.43 MB | Worse — all 3 changes hurt |
| v6 seed 1337 | Fractal 6L×2 loops, 512d/16H/8KV/4xMLP | 1.17566 | 10.65 MB | BUST — too few params, too slow |

## Key Findings (continued)

8. **QAT percentile clip mismatch fix = no gain** — changing QAT STE from row_max to 0.9995 percentile didn't improve quant tax.

9. **TrigramHash = marginal at best** — 3-token n-gram embeddings from PR #440 added params and overhead without measurable BPB gain on our stronger baseline.

10. **EMA-SWA blend (80/20) = slightly worse than pure EMA** — SWA dilutes EMA signal.

11. **Fractal weight sharing is a dead end at this scale** — 6L×2 loops (12 effective) at 512d/16H/4xMLP: 18.3M params (vs 27M for 11L), 126ms/step (vs 86ms), only 4757 steps. The double forward pass costs more compute than it saves in params. Final sliding window 1.1757 — nowhere near 1.1232.

12. **12L/480d/16H/4xMLP is strong on DGX Spark** — 2% relative improvement over baseline in local test (3.005 vs 3.071). But 29.5M params and 480d gives head_dim=30 (invalid for FA3). 512d/16H works (head_dim=32) but different tradeoffs.

## Submitted

PR #445: v1 seed 1337, 1.12319 BPB, 15.68 MB

## v7 TTT Results

| Config | BPB | Notes |
|--------|-----|-------|
| Full TTT (lr=0.002, 3ep, freeze=2, 1893 chunks) | 1.13599 | Degraded — overfitting past chunk 51 |
| Early stop 60 (lr=0.002, 3ep, freeze=2, 60 chunks) | **1.12312** | Best TTT result |
| Gentle TTT (lr=0.0005, 1ep, freeze=4, 1893 chunks) | 1.12328 | Same as early stop |

| Higher LR (lr=0.030, 3ep, freeze=2, 60 chunks) | 1.12467 | 15.89 MB | Worse — higher LR hurt base model |
| MTP (2 heads, 0.2 weight, early stop 60) | ~1.16+ | 15.63 MB | BUST — MTP needs more steps than 7000 |

Peak at chunk 51: **1.1119** — unachievable over full val set with current approach.
PR #473 gets 1.1218 with same recipe — their parameter banking likely helps TTT stability.

## SwiGLU Fork Results (2026-03-23)

| Config | BPB | Size | Notes |
|--------|-----|------|-------|
| SwiGLU + GPTQ + OptRot + AdamW TTT | **1.0763** | 19.6 MB ❌ | Over 16MB limit — OptRot hurts compression |
| v7 GPTQ + TTT EMA (seed 1337) | **1.1206** | 15.56 MB ✅ | PR #508 submitted |
| v7 GPTQ + TTT EMA (seed 42) | **1.1218** | 15.57 MB ✅ | |
| v7 GPTQ + TTT EMA (seed 7) | **1.1221** | 15.56 MB ✅ | |
| v7 GPTQ + TTT EMA (3-seed mean) | **1.1215** | — | Beats old SOTA 1.1218 |
| v7 GPTQ + AdamW TTT (seed 1337) | 1.1498 | 17.1 MB ❌ | AdamW worse on relu² arch |

## Key Insight
SwiGLU + AdamW TTT = 1.0763 BPB. Architecture is the multiplier for AdamW TTT.
Size problem: GPTQ+OptRot inflates artifact 19.6MB vs PR #462's 15.7MB with naive int6.
Next: solve size (disable OptRot? int5 MLP?) to submit competitive score.
