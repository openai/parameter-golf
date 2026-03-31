# bandit_wagon_XSA — XSA Coverage Sweep on 4F+1C

## Background

BW5F ablations (2026-03-30) established:
- 4F+1C is the optimal F-count at this parameter budget
- Raw learning rate is identical across all tested configs (~1.424 raw val_bpb at 500 steps)
- The entire performance gap between configs lives in quantization robustness
- XSA coverage is a quantization robustness lever:
  - 5F+1C XSA=11 (61% coverage): quant gap = 0.119
  - 5F+1C XSA=14 (78% coverage): quant gap = 0.106 → recovered 0.015 BPB

## Hypothesis

**Wider XSA on the confirmed-optimal 4F+1C model will reduce the quantization gap
and improve final BPB**, because XSA attention provides cross-block bandwidth that
smooths the perturbation introduced by int6 quantization.

Current 4F+1C: XSA=11 out of 15 blocks (73% coverage), quant gap ~0.099 at proxy.
At full run the quant gap is ~0.24 BPB (raw ~0.95 → final 1.186). Real headroom exists.

Risk: wider XSA costs compute. On 8×H100 at 600s, slower steps = fewer total steps.
The ablation measures this tradeoff directly — step time is recorded for each arm.

## Arms

| ID | Config | XSA_LAST_N | Coverage | Purpose |
|----|--------|:----------:|:--------:|---------|
| Control | 4F+1C, dim=512 | 11 | 73% | BW2-00 result: **1.52365** (carried) |
| BWXSA-01 | 4F+1C, dim=512 | 13 | 87% | partial coverage increase |
| BWXSA-02 | 4F+1C, dim=512 | 15 | 100% | full coverage — ceiling |

XSA=15 is the ceiling for the 15-block model. If full coverage doesn't beat XSA=11,
nothing will. XSA=13 isolates whether partial coverage recovers most of the gain cheaply.

## Decision Rules

| Outcome | Action |
|---------|--------|
| Either arm improves proxy BPB AND step overhead <8% | Gate winner at 2000 steps, then full 8×H100 run |
| Improvement exists but step overhead >8% | Evaluate net at full-run step count before committing |
| No improvement | XSA=11 is already optimal. Stop. |

**8% overhead threshold rationale:** At 546ms/step baseline on 1×H100, +44ms/step.
On 8×H100 at 600s, 8% slower ≈ 640 fewer steps out of ~8000 (~8%). Needs to return
>8% BPB improvement to net positive — unlikely at the scale of quant robustness gains.

## Locked Base Config

| Setting | Value | Source |
|---------|-------|--------|
| `NUM_FLAT_LAYERS` | 4 | BW5F confirmed |
| `MODEL_DIM` | 512 | BW anchor |
| `CRAWLER_LOOPS` | 3 | CL1 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 |
| `CRAWLER_QUANT_INT8` | 1 | CL1 |
| `SKIP_GPTQ` | 1 | CL3 |
| `SKIP_EMA` | 1 | Ablations_v1 |
| `COMPILE_FULLGRAPH` | 0 | CL3 |
| `SEED` | 444 | BW ablation |

## Results

| ID | XSA_LAST_N | Step avg (ms) | INT6_SW_BPB | Quant gap | Delta vs control |
|----|:----------:|:-------------:|:-----------:|:---------:|:----------------:|
| Control (BW2-00) | 11 | 546ms* | 1.52365 | 0.099 | — |
| BWXSA-01 | 13 | **530ms** | 1.51982 | 0.095 | −0.00383 |
| BWXSA-02 | 15 | **514ms** | **1.51431** | **0.090** | **−0.00934 ✅ PROMOTED** |

\* different pod session — cross-session timing unreliable. BWXSA-01 vs BWXSA-02 (same session) is reliable.

**Verdict: XSA=15 wins on BOTH metrics. Faster AND better BPB. Full coverage is the config.**
See ablation_results_2026-03-30.md for full analysis.

## Reference

| System | BPB | Notes |
|--------|-----|-------|
| CL3 / BW-00 (full run, 4F+1C XSA=11) | 1.18616 | current Crawler SOTA, seed 444 |
| BW2-00 (proxy, 4F+1C XSA=11, 500 steps) | 1.52365 | proxy control for this sweep |
