# Negative Results Compendium

**Author:** nprime06 | **Date:** 2026-04-30 | **Track:** Non-record (negative results)

50+ compute runs on 8xH100 documenting what didn't work on the path from PR-1493 (1.0810) to PR-1787 (1.06335). Full details with code snippets and tables in [DETAILED_RESULTS.md](DETAILED_RESULTS.md). Weight entropy deep-dive in [WEIGHT_ENTROPY_ANALYSIS.md](WEIGHT_ENTROPY_ANALYSIS.md).

## Summary

| # | Direction | Verdict | Key number | Why it fails |
|---|-----------|---------|------------|--------------|
| 1 | **FP8 MLP training** | Dead | +13–18% slower per step | Arithmetic intensity (340–407 FLOP/byte) below H100 FP8 ridge (591). amax+cast overhead (~17 ms) exceeds GEMM savings ceiling (~3–6 ms). |
| 2 | **Multi-token prediction** | Dead | 3 implementations, all negative | ~4–8% step-rate overhead at 600s budget costs ~200–400 training steps. Future-token signal too weak to compensate. |
| 3 | **Batch-size ramp** | Marginal | -0.0012 BPB (PR-1493), +1.94 mBPB (PR-1736) | Small isolated win on older stack. Loses on newer stack. Compile transitions eat 15–30s of budget. |
| 4 | **WD scheduling** | Wash | Artifact unchanged across all WD configs | L2 WD is scale-invariant under SDClip: `Q(γW) = Q(W)`. Proportional shrinkage is absorbed by per-row scale. Binary WD-off early = +0.0058 BPB — early WD is load-bearing. |
| 5 | **L1 sparsity** | Dead | Crashed at step 4500 (λ=1e-4) | Cumulative proximal L1 overwhelms 36M-param model as LR decays. Stable λ too weak to create sparsity. |
| 6 | **Loop curriculum 1→2→3** | Negative | +1.15 mBPB vs stock 1→3 | Final depth-3 phase gets only 35% of tokens — undertrained. Stock binary 1→3 gives depth-3 65% of tokens. |
| 7 | **SparseGPT** | Dominated | ~20% sparsity, 0.11 MB saved | Brotli compresses near-zero ints almost as well as exact zeros. k-inflation (k=19.3) saves 6x more bytes at comparable BPB cost. |
| 8 | **Quant grid design** | Near-optimal already | NF5: worse on both axes | Uniform int6 at k=12.85 is tuned for Brotli, not MSE. Non-uniform grids maximize entropy by construction — exactly what Brotli hates. GPTQ absorbs most per-tensor allocation changes. |
| 9 | **Tokenizer sweep** | Exhausted | No transforms found | CaseOps was the last favorable byte-savings-per-information-loss transform. PR-1271 cautionary: 93% of Scylla's claimed gap was a byte-accounting error. |
| 10 | **Dataset substitution** | Negative | 100% FineWeb-Edu: +0.068 BPB | Val set distribution is closer to original train data than to Edu. Moving toward original improves monotonically. |
| 11 | **max-autotune** | Dead | -0.8% steady-state, +34 min compile | Default Inductor kernel choices already near-optimal for these GEMM shapes. Not viable without persistent compile cache. |
| 12 | **DeepSeek NS10** | Dominated | -1.0 mBPB (1-seed), within noise at scale | Slightly slower per step. Polar Express 5-step gives comparable quality at zero throughput cost. |
| 13 | **Weight entropy shaping** | Fundamental limit | Best: 105 KB saved for +0.185 mBPB | Gaussian = max entropy at fixed variance = max expressiveness. Training-time regularizers resolve this poorly — the model games bucket penalties. See [WEIGHT_ENTROPY_ANALYSIS.md](WEIGHT_ENTROPY_ANALYSIS.md). |
| 14 | **Bigram training** | Superseded | Dropped from all SP8192 frontier PRs | Larger vocab captures bigram information natively. n-gram tilt (PR-1420) more effective than separate BigramHash embedding. |

## The Three Big Lessons

**1. L2 weight decay is scale-invariant under SDClip quantization.** `Q(γW) = Q(W)` because per-row scale absorbs proportional shrinkage. This kills all WD-tuning, L2-based compression, and most "train for compressibility" ideas in one theorem. Only methods that change distribution *shape* (not scale) can help — and those are hard because Gaussian shape is what makes the model expressive.

**2. GPTQ absorbs most quantization improvements.** Per-tensor bit allocation, sensitivity-based reallocation, alternative grids, and SparseGPT all get absorbed by GPTQ's Hessian error compensation. The strongest quant lever is the simplest: k-inflation (widen clipping to make the integer distribution peakier for Brotli). The best *reliable* lever is loop-aware k allocation (spend precision where sensitivity is high).

**3. The 600s budget is brutally tight.** Any overhead — FP8 cast kernels, MTP auxiliary losses, batch-size recompilation, autotuning — directly costs training steps. At ~5000 total steps, even 4% overhead means ~200 fewer steps, which is hard to recover through improved gradient signal. The winning strategy is to maximize raw tokens/second, not to add auxiliary objectives.

## Included Files

| File | Description |
|------|-------------|
| [DETAILED_RESULTS.md](DETAILED_RESULTS.md) | Full writeup with code snippets, tables, and per-experiment analysis |
| [WEIGHT_ENTROPY_ANALYSIS.md](WEIGHT_ENTROPY_ANALYSIS.md) | Standalone: Gaussian expressiveness vs. compression ceiling |
| `analyze_entropy.py` | Shannon entropy + bucket distribution analysis on saved weight bundles |
| `analyze_buckets.py` | GPTQ q-stream bucket analysis across quant configs |

## Lineage

Built on PR #1493 (bigbag, 1.0810) and PR #1736 (dexhunter, 1.06549). Our positive contribution is PR #1787 (1.06335, merged).
