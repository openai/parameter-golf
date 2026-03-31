# bandit_wagon_smear — Loop SmearGate: Depth Error Damping

## Background

The crawler applies the same quantized weights 3× in series. Quantization error doesn't
just accumulate — it **amplifies**: each loop reprocesses the previous loop's error through
the same error-prone weights. This is fundamentally different from a standard transformer
where each layer has independent weights and errors accumulate additively.

```
Loop 0: quantized(x)            → x + ε₀
Loop 1: quantized(x + ε₀)      → x + ε₁  (ε₀ gets re-amplified)
Loop 2: quantized(x + ε₀ + ε₁) → x + ε₂  (compound amplification)
```

**Hypothesis:** A learnable blend between consecutive loop outputs (LoopSmearGate) will
damp error propagation across loop depth by mixing the current loop's noisy output with
the previous loop's less-corrupted output before feeding into the next iteration.

## Architecture

```python
x_prev_loop = x_encoder          # stable anchor (no quantization loops yet)
for loop in 0..2:
    x_loop = run_blocks(x + flow[loop])
    x_loop = loop_smear(x_loop, x_prev_loop)   # blend current with previous
    x_prev_loop = x_loop
    x = x_loop
```

**LoopSmearGate:**
```python
g = sigmoid(gate)                          # learned per-dimension blend weight
return (1-g) * x_current + g * x_previous  # soft interpolation
```

- ~512 learned scalars, **zero matmuls** — essentially free
- gate init=zeros → sigmoid(0)=0.5 start (model learns direction)
- Loop 0 smears with encoder output: creates a soft skip from encoder to loop 0 output
- No causality violation: blending across loop depth, not token positions

## Key difference from FLOW

FLOW conditions the **input** to each loop (additive correction before the block runs).
LoopSmearGate acts on the **output** of each loop before feeding the next — it's a
low-pass filter across loop depth, not a content-aware correction.

These are orthogonal and can be combined.

## Connection to the tap idea

The loop 0 smear with encoder output is a degenerate form of the encoder tap concept:
it gives the crawler a direct connection back to the pre-loop signal at each depth.
A full encoder tap would generalize this to per-layer projections per loop.

## Arms

| ID | CRAWLER_LOOP_SMEAR | Purpose |
|----|:------------------:|---------|
| BWS-00 | 0 | **Control repin** — must match BW2-00 (1.52365 ±0.002) |
| BWS-01 | 1 | Loop smeargate active — gate=zeros, learned per-dimension |

## Decision Rules

**Gate 0:** BWS-00 must land 1.521–1.526. If it misses: code bug. Stop.

**Gate 1:** BWS-01 must beat BWS-00 by ≥0.005 to justify promotion.

**If BWS-01 wins:** 2000-step gate → combine with XSA=15 and winning choke_dim
before 8×H100.

**If BWS-01 doesn't win:** Smeargate is not a meaningful depth-error lever at 500
steps. Encoder tap (per-layer, per-loop projections) is the richer version to probe next.

## Locked Base Config

| Setting | Value | Source |
|---------|-------|--------|
| `NUM_FLAT_LAYERS` | 4 | BW5F confirmed |
| `XSA_LAST_N` | 11 | baseline |
| `MODEL_DIM` | 512 | BW anchor |
| `CRAWLER_LOOPS` | 3 | CL1 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 |
| `CRAWLER_MLP_CHOKE_DIM` | 0 | isolate smear variable |
| `CRAWLER_MLP_LEAKY_SLOPE` | 0.5 | control value |
| `SEED` | 444 | BW ablation |

## Results

| ID | SMEAR | Step avg (ms) | Raw val_bpb | INT6_SW_BPB | Quant gap | Delta |
|----|:-----:|:-------------:|:-----------:|:-----------:|:---------:|:-----:|
| BWS-00 | 0 | TBD | TBD | TBD | TBD | control |
| BWS-01 | 1 | TBD | TBD | TBD | TBD | TBD |

Reference: BW2-00 (XSA=11, no smear) → 1.52365
