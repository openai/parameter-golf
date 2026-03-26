# SSM LRU Baseline (WIP)

**Status**: Work in progress — seeking compute credits to validate on H100.

## Approach

First state space model (SSM) submission to parameter golf. Replaces transformer attention with a **Linear Recurrent Unit (LRU)** — a minimal SSM with complex diagonal recurrence.

### Architecture
- **LRU**: Complex diagonal state matrix (log-magnitude + phase parameterization)
- **Parallel scan**: Cumulative sum trick in log-space (torch.compile friendly, no custom CUDA kernels)
- **Gated projection**: Sigmoid gate on SSM output
- **MLP**: ReLU^2 activation (same as transformer baseline)
- **Optimizer**: MuonAdamW with SSM-aware param groups (Adam for A/B/C/D, Muon for projections)

### Why SSMs for Parameter Golf
- SSM blocks are **36% smaller** than attention blocks at equivalent dimension
- SSM can **absorb the MLP** — no separate MLP needed, halving block size
- **No KV cache** — native sliding window eval without recomputation
- SSM-specific params (A, B, C, D) are <0.2% of total — projections dominate
- Can fit **12-15 SSM layers** where transformers fit 9-10 in 16MB

### Current Results (RTX 3090, 5 min budget)
- val_bpb: 1.848 (not competitive yet — bottlenecked by training speed without CUDA kernels)
- MFU: 2.8% (vs 40% for flash attention transformers)
- Pure PyTorch scan is ~15x slower than fused CUDA kernels

### What H100 Compute Would Unlock
- `mamba_ssm` package with fused CUDA kernels → 10-50x faster training
- Proper batch sizes and model dimensions
- Expected to close the quality gap entirely

### Research Backing
- 6 deep-dive research documents covering LinOSS, Mamba-3, SSM taxonomy, compression frontiers
- Autonomous experiment loop (autoresearch) with brainstorming, paper reading, self-reflection
- 5 experiments completed, clear path to competitive results with proper compute
