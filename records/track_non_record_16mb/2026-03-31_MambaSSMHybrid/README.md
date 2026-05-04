# Non-record: Mamba-Inspired SSM Hybrid (3:1 SSM:Attention)

**val_bpb: 3.3168** | 1×RTX 5090, 180s

Pure PyTorch SSM implementation (no custom CUDA kernels). 3:1 ratio of SSM to attention blocks following Qwen3-Next/Kimi Linear pattern. Selective gating with input-dependent state transitions, causal Conv1d, and SiLU-gated output.

Implements OpenAI's requested 'State-space models' direction.
