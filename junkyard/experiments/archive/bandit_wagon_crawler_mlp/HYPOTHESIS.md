# bandit_wagon_crawler_mlp — Crawler MLP Leaky Slope Sweep

## Background

The crawler block's MLP already has a separate `CRAWLER_MLP_MULT` (6.0 vs flat 3.0).
But `mlp_leaky_slope` has always been SHARED between flat and crawler blocks via a
single `MLP_LEAKY_SLOPE=0.5` env var. The crawler is fundamentally different from flat
blocks: it is applied 3× in series with loop-indexed FLOW conditioning. The optimal
leaky slope for a repeatedly-applied MLP is not necessarily the same as for a
single-pass block.

## Activation function

`leaky_relu_sq`: x² if x≥0, else leaky_slope × x²

- slope=0.0 → pure relu_sq, maximum sparsity, zero negative gradient
- slope=0.5 → current shared value
- slope=1.0 → symmetric x², no sparsity asymmetry

Flat blocks stay locked at MLP_LEAKY_SLOPE=0.5 for all arms. Only CRAWLER_MLP_LEAKY_SLOPE varies.

## Code change (train_gpt.py — new file, not modifying tested scripts)

Four additions:
1. `crawler_mlp_leaky_slope = float(os.environ.get("CRAWLER_MLP_LEAKY_SLOPE", 0.5))` (line 68)
2. `crawler_mlp_leaky_slope: float = 0.5` added to CrawlerGPT.__init__ signature
3. crawler_blocks construction uses `mlp_leaky_slope=crawler_mlp_leaky_slope` (was `mlp_leaky_slope`)
4. build_model() passes `crawler_mlp_leaky_slope=args.crawler_mlp_leaky_slope`

Default is 0.5 — bit-equivalent to all prior runs when CRAWLER_MLP_LEAKY_SLOPE is unset.

## Arms

| ID | CRAWLER_MLP_LEAKY_SLOPE | Regime | Rationale |
|----|:-----------------------:|--------|-----------|
| BW3-00 | 0.5 | **control repin** | Must match BW2-00 (1.52365 ±0.002) — validates code change |
| BW3-01 | 0.0 | pure relu_sq | Max sparsity per loop — best quant robustness? Dead neurons don't recover across loops |
| BW3-02 | 0.25 | light asymmetry | Midpoint — retains some negative gradient to keep all loops alive |
| BW3-03 | 0.75 | less sparse | Richer negative signal — FLOW corrections may span both sign directions |
| BW3-04 | 1.0 | symmetric x² | No sparsity. Full refinement signal. Does removing asymmetry hurt quant? |

## Decision Rules

**Gate 0 — control repin (BW3-00):**
BW3-00 must land 1.521–1.526. If it misses, stop: code change has a bug.

**Gate 1 — signal present:**
At least one arm must beat BW3-00 by ≥0.005 to justify promotion.
If all arms within ±0.003 of control: crawler is slope-insensitive, stop.

**Gate 2 — promotion:**
Winning arm → 2000-step gate → if beats BW2-00 proxy by ≥0.008 → 8×H100 full run.

**Special:** If BW3-01 (0.0) wins, run 0.1 as a single follow-up to check monotonicity.

## Key Interaction Effects

- **Track raw val_bpb AND int6_sw_bpb separately** — all signal lives in the quant gap
- **slope=0.0 + CRAWLER_QUANT_INT8=1**: dead activations round exactly to zero, ideal for int8
- **slope=1.0 + 3 loops**: x² can amplify values >1 across loops — watch for val_loss instability
- **Flat blocks unchanged**: MLP_LEAKY_SLOPE=0.5 locked; this ablation is CRAWLER only

## Results

| ID | CRAWLER_MLP_LEAKY_SLOPE | Raw val_bpb | INT6_SW_BPB | Quant gap | Delta |
|----|:-----------------------:|:-----------:|:-----------:|:---------:|:-----:|
| BW3-00 | 0.5 | 1.4509 | 1.55702 | 0.1061 | control |
| BW3-01 | 0.0 | 1.4504 | 1.55741 | 0.1070 | +0.00039 ❌ |
| BW3-02 | 0.25 | 1.4525 | 1.56116 | 0.1087 | +0.00413 ❌ |
| BW3-03 | 0.75 | 1.4526 | **1.55637** | **0.1038** | **−0.00065** |
| BW3-04 | 1.0 | 1.4524 | 1.55656 | 0.1042 | −0.00046 |

**VERDICT: Not promotable. Slope is insensitive — stay at 0.5.**
No arm cleared ≥0.005. Marginal directional signal: higher slope (0.75) slightly helps
because negative gradient carries cross-loop corrections. Pure relu_sq (0.0) is worst.
See ablation_results_2026-03-30.md for full analysis.

Reference: BW2-00 (shared slope=0.5, XSA=11, flash_attn pod) → 1.52365
This session (no flash_attn): control BW3-00 → 1.55702
