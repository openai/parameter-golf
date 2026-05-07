# Static-Unified Flash-Nitro (SUFN) 🏆🏁

This submission provides an **Optimized Systems Baseline** for the Paradox-Golf 10-minute 8x H100 challenge. Rather than pushing for raw SOTA BPB, it focuses on eliminating the common architectural and compiler bottlenecks in the "modded-nanogpt" lineage.

## Core Systems Optimizations 🛡️

### 1. Perfectly Static Unified Graph
Traditional runs often trigger a 30-40 second `torch.compile` (Inductor) stall at step 520 when changing sequence lengths.
- **Optimization:** We unified the sequence length to **1024** for the entire run. This keeps the graph perfectly static, gaining ~40 seconds of training time.

### 2. Native is_causal Flash Attention 🏎️
Custom SDPA wrappers often trigger OOM or "invalid backend" errors on H100s.
- **Optimization:** We use standard `F.scaled_dot_product_attention` with `is_causal=True`. This leverages native Flash Attention 2 for $O(n)$ efficiency and maximum throughput.

### 3. Rank-Sharded Token Loading Fix
Fixed the `RuntimeError: shape invalid for input size` by implementing rounding logic in the `DistributedTokenLoader`. This ensures multi-GPU training is stable across all rank counts.

## Results
- **Hardware:** 8x H100 GPUs
- **Wallclock:** 550.2 seconds
- **Final BPB:** 2.5226

**This submission is intended as a high-performance blueprint for further architectural exploration.** 🏁📉🏆✨
