# bandit_wagon_cannon (BWE) — Per-Loop Output Calibration

## Background

BWCD-02 established pyramid-512 + 9,1,1 battery as the current best config:
- Loop 0: RoPE scale=9 (wide, global context on cleanest residual)
- Loop 1: RoPE scale=1 (local refinement)
- Loop 2: RoPE scale=1 (local refinement, identical to loop 1)
- Quant gap: +0.0001 (near-zero)
- vs pyramid alone: -0.01193

The battery aligned the **attention side** — what each loop reads.
The cannon addresses the **output side** — what each loop fires into the residual stream.

## The Problem

Loop 0 reads at 9× wider context than loops 1+2. Wide attention aggregates
more signal per token → loop 0's MLP output may arrive at the residual at a
different amplitude than what the shared weights of loop 1 expect to receive.

Loops 1+2 are calibrated to each other (identical scale = near-identical
distributions), but loop 0 is the structural outlier. The residual quant_gap
of +0.0001 is likely this amplitude mismatch.

## Mechanism

Applied to the **delta** (loop_out − loop_in), not the full residual.
At initialization, cannon=1.0 is an exact no-op — BWE-00 and BWE-01 with
fresh weights produce identical output to BWCD-02. The model only moves the
cannon away from 1.0 if it finds a better amplitude for each loop's contribution.

```
delta = x_after_loop - x_before_loop   # what this loop added
x = x_before_loop + cannon[loop] * delta
```

Expected behavior: cannon[0] (loop 0, wide) learns to dampen or scale its
contribution. cannon[1] and cannon[2] (loops 1+2, local) stay near 1.0.

## Arms

| ID | Type | Params | Description |
|----|------|:------:|-------------|
| BWE-00 | none | 0 | Control — must match BWCD-02 (1.43531) |
| BWE-01 | scalar | 3 | 1 learnable gain per loop |
| BWE-02 | channel | 1,536 | Per-channel gain vector per loop (512×3) |
| BWE-03 | rmsnorm | 1,536 | RMSNorm on delta per loop |

All arms: pyramid-512 + CRAWLER_LOOP_ROPE_SCALES=9,1,1

## References

| Run | Config | INT6_SW_BPB | Quant Gap |
|-----|--------|-------------|-----------|
| BWCS-02 | pyramid-512 (1 shard) | 1.44724 | -0.0001 |
| BWCD-02 | pyramid + 9,1,1 (1 shard) | **1.43531** | +0.0001 |

## Results

| ID | Type | Step avg | Raw BPB | INT6_SW_BPB | Quant Gap | vs BWCD-02 |
|----|------|:--------:|:-------:|:-----------:|:---------:|:----------:|
| BWE-00 | none | 663.15ms | 1.4359 | 1.44165584 | +0.0058 | +0.00635 |
| BWE-01 | scalar | 745.65ms | 1.4414 | 1.44336814 | +0.0020 | +0.00806 |
| BWE-02 | channel | 608.84ms | 1.4366 | 1.43589764 | -0.0007 | +0.00059 |
| BWE-03 | rmsnorm | 554.28ms | 1.4531 | 1.46352025 | +0.0104 | +0.02821 |

### Readout (seed 444, 500-step proxy, nproc=1)

- No cannon arm beat BWCD-02 (1.43531057).
- Best cannon arm was BWE-02 (channel), but still +0.00059 behind BWCD-02.
- Scalar cannon regressed quality and slowed throughput substantially.
- RMSNorm cannon was the worst quality arm (+0.02821 vs BWCD-02).
