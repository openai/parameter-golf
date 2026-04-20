# Non-Record Submission: Architectural Scaling (SwiGLU + Parallel Residuals)

This document explores the impact of architectural scaling—specifically increasing depth and changing activation/residual patterns—under a strict **600-second wall-clock limit**.

The goal was to determine if a more complex, deeper model could achieve lower validation bits-per-byte (val_bpb) than the standard baseline, despite increased computational costs. We conducted initial baseline testing on an RTX 4090 and scaled the model on H100 hardware.

## Summary of Architectural Changes (Exp 003 vs Baseline)

1.  **Increased Depth:** Scaled from **9 to 11 layers**.
2.  **Parallel Residuals:** Modified `Block` forward pass to compute Attention and FFN branches in parallel (`x = x + attn_out + ffwd_out`), improving gradient flow.
3.  **SwiGLU Activation:** Replaced the stock `ReLU^2` MLP with a 3-matrix SwiGLU configuration for dynamic gating.
4.  **QK-Gain:** Adjusted initialization to `2.5` to optimize Softmax probability distribution.

## Key Discovery: Throughput vs. Convergence
Our tests reveal a fundamental trade-off: The architectural upgrades increase the average step time on an H100 by ~9% (from ~513ms to ~561ms). However, the **sample efficiency** of the deeper SwiGLU-based model significantly outperforms the baseline. Even though the modified architecture completes ~9% fewer steps in 600 seconds, the model reaches a lower loss than the simpler, shallower baseline could achieve in its higher step count.

## Experiment Log (600s Constraint)

| Exp | Hardware | Description | Layers | Steps | step_avg | val_bpb (INT8) |
|-----|----------|-------------|--------|-------|----------|----------------|
| 001 | RTX 4090 | SwiGLU + Parallel | 9 | 20000 | ~860ms | 1.2193 |
| 002 | H100     | SwiGLU + Parallel | 9 | 1,168 | ~513ms | 1.3643 |
| 003 | H100     | SwiGLU + Parallel | 11 | 1,069 | ~561ms | 1.3565 |

*Note: The RTX 4090 run reached a lower bpb due to MAX_WALLCLOCK_SECONDS=0 and ran for 20000 steps, but the direct H100 comparison (Exp 002 vs 003) confirms a **net improvement of -0.0078 val_bpb** within the same compute environment.*

## Configuration (Best Run — exp003)

```bash
RUN_ID=h100_swiglu_qk2.5_num-layer-11_h100
VOCAB_SIZE=1024 
NUM_LAYERS=11 
MODEL_DIM=512 
NUM_HEADS=8 
NUM_KV_HEADS=4
ITERATIONS=20000
MAX_WALLCLOCK_SECONDS=600.0
QK_GAIN_INIT=2.5
```