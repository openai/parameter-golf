# Bandit_wagon_5f_ablations — 4F vs 5F Direct Comparison

## Background

BW ablation (2026-03-30) ran four proxy arms (BW-01 through BW-04) but **never ran BW-00
(4F+1C) as a 500-step proxy arm**. The anchor 1.18616 is from a full 600s run. BW-03
(5F+1C, XSA=11) scored 1.54404 but was never compared to 4F+1C at equal compute.

CL1-09 (the only prior direct 4F vs 5F test) used loops=4, mlp=4.0, no XSA, no FLOW,
no relu_sq — an architecturally different system. That data is not reliable for the
current config.

## Hypothesis

**5F+1C at loops=3, mlp=6.0 beats 4F+1C in the current architecture**, and the BW-03
proxy result (1.54404) will hold up against a proper 4F+1C control run at equal steps.

Secondary: XSA_LAST_N=11 was tuned for a 15-block model (4F+1C×3). At 5F+1C×3=18
blocks, coverage drops from 73% → 61%. Adjusting to XSA_LAST_N=14 restores proportional
coverage and may further improve 5F+1C.

## Arms

| ID | Config | XSA_LAST_N | Blocks | XSA Coverage | Purpose |
|----|--------|:---------:|:------:|:------------:|---------|
| BW2-00 | 4F+1C, dim=512 | 11 | 15 | 73% | **THE CONTROL — missing from BW** |
| BW2-01 | 5F+1C, dim=512 | 14 | 18 | 78% | Proportional XSA for 18-block model |
| BW-03\* | 5F+1C, dim=512 | 11 | 18 | 61% | Reference (already run) → 1.54404 |

\* Not re-run. Result carried forward from BW ablation (seed=444, 500 steps).

## Decision Rules

| Outcome | Action |
|---------|--------|
| BW2-00 < BW-03 (4F worse at proxy) | 5F+1C confirmed → gate BW2-01 winner at 2000 steps |
| BW2-00 > BW-03 (4F still wins) | Stop. 4F+1C is optimal. Don't book 8×H100. |
| BW2-01 < BW-03 (XSA adjustment helps) | 5F+1C + XSA=14 is the full-run candidate |
| BW2-01 ≥ BW-03 (XSA adjustment neutral) | 5F+1C + XSA=11 (BW-03 config) is the candidate |

## Locked Base Config (from CL3 / BW)

| Setting | Value | Source |
|---------|-------|--------|
| `CRAWLER_LOOPS` | 3 | CL1 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 |
| `CRAWLER_QUANT_INT8` | 1 | CL1 |
| `SKIP_GPTQ` | 1 | CL3 |
| `SKIP_EMA` | 1 | Ablations_v1 |
| `COMPILE_FULLGRAPH` | 0 | CL3 |
| `MODEL_DIM` | 512 | BW anchor |
| `SEED` | 444 | BW ablation |

## Results

| ID | XSA_LAST_N | INT6_SW_BPB | Delta vs BW-03 | Notes |
|----|:----------:|:-----------:|:--------------:|-------|
| BW-03 (ref) | 11 | 1.54404 | — | carried from BW |
| BW2-00 | 11 | TBD | TBD | **4F control** |
| BW2-01 | 14 | TBD | TBD | 5F proportional XSA |

## Reference

| System | BPB | Notes |
|--------|-----|-------|
| BW-00 anchor (4F+1C, full run) | 1.18616 | seed 444, 600s, 8×H100 |
| CL3 mean (4F+1C, 3-seed) | 1.18742 | current Crawler SOTA |
