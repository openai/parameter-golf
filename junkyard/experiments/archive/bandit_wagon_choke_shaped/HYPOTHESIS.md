# bandit_wagon_choke_shaped — Shaped Bottleneck MLP Sweep

## Background

Flat choke sweep (bandit_wagon_battery mega run, 2026-03-30) produced:

| ARM | Choke dim | INT6_SW_BPB | Delta vs ctrl | Note |
|-----|:---------:|:-----------:|:-------------:|------|
| CTRL-00 | 0 | 1.44185 | — | control |
| BWC-01 | 32 | 1.45004 | +0.00819 | too tight, worse |
| BWC-02 | 128 | 1.43674 | **−0.00511** | clears threshold |
| BWC-03 | 256 | 1.44071 | −0.00114 | below threshold |
| BWC-04 | 512 | **1.42887** | **−0.01298** | strong winner |
| BWS-01 | smear | 1.44628 | +0.00443 | smear doesn't help |

**Key finding:** Improvements come from raw val_bpb, not quant gap reduction. BWC-04
(choke_dim=model_dim=512) is qualitatively different — it replaces the shared proj with
3 per-loop learned projections at full model width. BWC-03 (256) being worse than BWC-02
(128) suggests a non-monotonic regime where 512 crosses a threshold into full per-loop
routing rather than compression.

## Shaped Choke Hypothesis

With flat choke validated, the question becomes: can **structure inside the bottleneck**
improve over flat, either by matching BWC-04 at lower parameter cost, or by exceeding it?

### Shape 1 — Pyramid (BWCS-01, BWCS-02)
```
fc:     512 → 3072 (shared)
stage1: 3072 → 512 (shared, new — expensive matrix shared rather than replicated)
choke:  512 → [C per-loop] → 512
```
Shares the expensive 3072→512 compression. Per-loop routing happens at model-dim level
with cheap matrices. BWCS-01 is pure pyramid (no bypass). BWCS-02 adds the free residual:
stage1 output IS the bypass, delta is learned on top. Zero extra params for the residual.

**Why interesting:** Flat BWC-04 has 3 copies of 3072×512 (per-loop). Pyramid has 1 shared
copy + 3 cheap 512×512. Same total routing but cheaper and stage1 learns a universal
compressed representation that each loop refines.

### Shape 2 — Pyramid + residual (BWCS-03)
```
out = stage1_output + choke_up[loop](act(choke_down[loop](stage1_output)))
```
The stage1 output serves as bypass. At step 0, delta=0, so MLP starts with stage1's signal
(non-zero warm start, unlike flat which starts at zero). Matches LoRA-style learning: shared
base + per-loop corrections.

### Shape 3 — Grouped (BWCS-04, BWCS-05)
```
fc: 512 → 3072 (shared)
group_down[loop]: block-diagonal 3072 → 512, G independent groups of (3072/G)→(512/G)
choke_up[loop]:   512 → 512
```
Block-diagonal down-projection: each group of 384 input features compresses to 64 output
features independently. G groups, balanced surface area, equal representation budget.

**Why interesting:** The quantization surface per group is (3072/G)×(512/G) instead of
3072×512. Balanced contribution from all regions. "Local communication gradating toward
final solution." Same dimension as BWC-04 but fundamentally different routing structure.
Fewer effective parameters (block-diagonal vs dense).

### Shape 4 — Residual (BWCS-06)
```
bypass = proj(act(fc(x)))      # shared 3072→512 (original MLP structure)
delta  = choke_up[loop](act(choke_down[loop](act(fc(x)))))  # per-loop 3072→128→512
out = bypass + delta
```
Shared bypass carries the "universal" signal; per-loop choke learns the loop-specific
correction. Both zero-initialized — clean gradient flow from step 0. Flat choke at 128 dim
but bypass ensures no capacity loss at narrow bottleneck.

## Arms

| ID | Shape | Choke dim | Groups | Purpose |
|----|-------|:---------:|:------:|---------|
| BWCS-00 | control | 0 | — | Repin — must match CTRL-00 ≈ 1.44185 |
| BWCS-01 | pyramid | 128 | — | Cheap shared stage + per-loop 512→128→512 |
| BWCS-02 | pyramid | 512 | — | Pyramid at model dim — cheaper than flat-512? |
| BWCS-03 | pyramid_res | 128 | — | Pyramid + free residual bypass |
| BWCS-04 | grouped | 512 | 8 | 8 balanced groups, block-diagonal |
| BWCS-05 | grouped | 512 | 4 | 4 coarser groups |
| BWCS-06 | residual | 128 | — | Shared bypass + per-loop delta at 128 |

## Key Comparisons

- BWCS-01 vs BWC-02 (mega, flat-128): does shared stage1 add value at same dim?
- BWCS-02 vs BWC-04 (mega, flat-512): can pyramid-512 match flat-512 cheaper?
- BWCS-03 vs BWCS-01: does the free residual help pyramid?
- BWCS-04 vs BWC-04 (mega): can grouped-8 match flat-512 with block-diagonal routing?
- BWCS-05 vs BWCS-04: does group granularity (8 vs 4) matter?
- BWCS-06 vs BWC-02 (mega, flat-128): does the bypass rescue the narrow bottleneck?

## Decision Rules

**Gate 0 — repin:** BWCS-00 must land ≈ 1.44185 ± 0.002.

**Gate 1 — beats BWC-04:** Any arm with int6_sw_bpb < 1.42887 is a new winner.
Promote to 2000-step gate → 8×H100 if confirmed.

**Gate 2 — efficiency win:** Any arm that matches BWC-04 (within ±0.002) at lower
parameter count is worth promoting — smaller model quantizes better at full scale.

## Results

| ID | Shape | Dim | Groups | Step avg | Raw BPB | INT6_SW_BPB | Quant gap | Delta |
|----|-------|:---:|:------:|:--------:|:-------:|:-----------:|:---------:|:-----:|
| BWCS-00 | control | 0 | — | TBD | TBD | TBD | TBD | control |
| BWCS-01 | pyramid | 128 | — | TBD | TBD | TBD | TBD | TBD |
| BWCS-02 | pyramid | 512 | — | TBD | TBD | TBD | TBD | TBD |
| BWCS-03 | pyramid_res | 128 | — | TBD | TBD | TBD | TBD | TBD |
| BWCS-04 | grouped | 512 | 8 | TBD | TBD | TBD | TBD | TBD |
| BWCS-05 | grouped | 512 | 4 | TBD | TBD | TBD | TBD | TBD |
| BWCS-06 | residual | 128 | — | TBD | TBD | TBD | TBD | TBD |

Reference: BWC-04 (flat-512) = 1.42887, BWC-02 (flat-128) = 1.43674, CTRL-00 = 1.44185
