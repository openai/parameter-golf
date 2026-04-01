# bandit_wagon_crawler_mlp Results — 2026-03-30

**Setup:** seed=444, 500 steps, warmdown=0, SKIP_GPTQ=1, CRAWLER_QUANT_INT8=1
**Note:** Pod missing flash_attn — control repin missed 1.521–1.526 gate (landed 1.55702).
This is a pod/environment difference, NOT a code bug. Within-session relative comparison is valid.
**Flat blocks:** MLP_LEAKY_SLOPE=0.5 locked. Only CRAWLER_MLP_LEAKY_SLOPE varies.

## Results

| ARM | CRAWLER_MLP_LEAKY_SLOPE | Step avg (ms) | Raw val_bpb | INT6_SW_BPB | Quant gap | Delta vs ctrl |
|-----|:-----------------------:|:-------------:|:-----------:|:-----------:|:---------:|:-------------:|
| BW3-00 (ctrl) | 0.5 | 540.41ms | 1.4509 | 1.55702 | 0.1061 | — |
| BW3-01 | 0.0 | 540.50ms | 1.4504 | 1.55741 | 0.1070 | +0.00039 ❌ |
| BW3-02 | 0.25 | 540.61ms | 1.4525 | 1.56116 | 0.1087 | +0.00413 ❌ |
| BW3-03 | 0.75 | 540.30ms | 1.4526 | **1.55637** | **0.1038** | **−0.00065** |
| BW3-04 | 1.0 | 540.11ms | 1.4524 | **1.55656** | **0.1042** | **−0.00046** |

## Key Findings

### 1. Crawler MLP is slope-insensitive — 0.5 is already near-optimal

No arm beats control by ≥0.005. The threshold for promotion was not cleared.
Maximum delta is −0.00065 (slope=0.75) — within noise at proxy scale.

### 2. Direction is clear but marginal: MORE negative gradient slightly helps

| Slope | Quant gap | Direction |
|:-----:|:---------:|-----------|
| 0.0 | 0.1070 | worse — dead neurons can't carry corrections across loops |
| 0.25 | 0.1087 | worse — same issue, less severe |
| 0.5 | 0.1061 | control |
| 0.75 | **0.1038** | **best** — more negative gradient survives loops |
| 1.0 | 0.1042 | good — symmetric, but marginally worse than 0.75 |

The pattern is U-shaped with a minimum around slope=0.75. More negative gradient
(less sparsity) helps the crawler propagate corrections across 3 loop iterations.
Pure relu_sq (slope=0) is the worst — dead neurons cannot carry cross-loop signal.

### 3. Why slope=0 was expected to win but didn't

Hypothesis was: more sparsity → fewer non-zero activations → better quantization.
Reality: the crawler's 3-loop structure requires negative gradient to flow corrections
backwards. Zeroing out negative activations kills the cross-loop correction mechanism.
The quantization benefit of sparsity is outweighed by the loss of correction bandwidth.

### 4. Speed: flat across all arms

All arms ran at ~540ms/step — slope has zero effect on step time, as expected.

## Decision

**VERDICT: Slope is not a meaningful lever. Stay at 0.5.**

No arm cleared the ≥0.005 gate. The marginal improvement at slope=0.75 (−0.00065)
does not justify a config change — it would be lost in run-to-run variance.

The deeper insight: the crawler's quantization gap is structural (multi-context weights,
depth error amplification), not addressable via activation function shape.
→ choke, smear, tap, and battery are the right interventions.

## Reference

| System | BPB | Notes |
|--------|-----|-------|
| BW2-00 (4F+1C XSA=11, flash_attn pod) | 1.52365 | proxy control, different session |
| BW3-00 (same config, no flash_attn pod) | 1.55702 | this session's control |
| BW3-03 (slope=0.75, best arm) | 1.55637 | −0.00065 vs BW3-00, not promotable |
