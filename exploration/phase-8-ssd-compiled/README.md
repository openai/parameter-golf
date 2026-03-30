# Phase 8: SSD / Mamba Compiled Track

**Dates:** Mar 19-21, 2026 (parallel track)
**Goal:** Build a competitive model using Mamba-2 SSD sequence mixing with depth recurrence and 4-bit QAT.
**Outcome:** Best single result: **1.3196 BPB** on 8xH100 (iter-004 v2 champion). This track ran in parallel with the transformer-based phases.

## Sub-tracks

### SSDGolf (iter-001 through iter-003)
- SpectralGolf v1 and v2: FFT-based spectral mixing (abandoned — cuFFT bf16 issues)
- SSDGolf: Mamba-2 SSD chunk algorithm, depth recurrence, QAT
- LR sweeps, architecture scaling, smoke tests

### MultilateralSSD (iter-004)
- d=1536, 31.7M params, 4 iterations
- **Champion run: 1.3196 BPB** (8xH100, 600s, 944M tokens seen)
- x0-only variant: 1.689 BPB on single GPU
- Wave 3 variants explored from this champion

### Compiled SSD (iter-005)
- torch.compile optimizations for throughput
- Triton kernels
- Ablation studies (ablation-1 through ablation-8)
- Config sweeps (cfg_a through cfg_d)
- Best: ~1.89 BPB on single GPU configs

### Wave Experiments
- wave2: Variants from ablation-3 peak (dstate32, seq2048, triple fusion)
- wave3: Variants from iter-004 champion (clean, seq2048 push, throughput max)

## Key Insight

Under fixed wall-clock training, the optimal architecture minimizes loss per second, not loss per parameter. The SSD track's throughput advantage (~1.6M tok/s on 8 GPUs) allowed it to see 944M tokens in 10 minutes, compensating for lower per-token efficiency vs attention.
