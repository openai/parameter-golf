# GatedDeltaNet SSM — 12L 384d (Non-Record, Unlimited Compute Track)

A Gated DeltaNet selective state space model submission using production Triton kernels from the `flash-linear-attention` (fla) library. This replaces the standard attention mechanism with a delta-rule recurrence that enables O(1) memory per token at inference time.

## Motivation

SSMs are on the OpenAI wishlist for parameter golf submissions. This submission demonstrates a clean, practical integration of a modern SSM (Gated DeltaNet) with the existing parameter golf training infrastructure (Muon optimizer, U-Net skips, bigram embeddings, z-loss).

## Architecture

- **SSM layer**: Gated DeltaNet (`fla.layers.GatedDeltaNet`)
  - State update: `S_t = α_t · S_{t-1} · (I − β_t · k_t kᵀ_t) + β_t · v_t · kᵀ_t`
  - Selective memory erasure via delta rule (more expressive than pure decay)
  - Chunk-parallel scan via fused Triton kernels (`mode='chunk'`, chunk_size=64)
  - Causal conv1d + gated output (SiLU gate)
- **Layers**: 12 (U-Net encoder/decoder split with learned skip weights)
- **Model dim**: 384, head dim: 64, 6 heads per layer
- **MLP**: LeakyReLU(0.5)² activation, 2× expansion
- **Embeddings**: Tied token embeddings + BigramHash embedding (vocab 1536, dim 128)
- **Logit cap**: Polynomial softcap (degree 5, cap=30)
- **Z-loss**: 1e-4 × logsumexp(logits)².mean()
- **Total params**: ~13.7M
- **Artifact size**: 15.79 MB int8+zlib (under 16MB limit)

## Optimizer

- **Muon** (Newton-Schulz, momentum 0.95) for 2D weight matrices
- **Adam** for scalars, embeddings, and GDN-specific delta-rule params (a_proj, b_proj, A_log, dt_bias, o_norm)
- GDN delta-rule parameters explicitly routed to Adam to prevent Newton-Schulz orthogonalization from corrupting the recurrence dynamics

## Training

- **Hardware**: 8× H100 80GB
- **Batch**: 524,288 tokens/step (data parallel across 8 GPUs)
- **Sequence length**: 1,024
- **Steps**: 4,962 in 10 minutes (~121ms/step)
- **Wallclock**: 600s (non-record unlimited compute track)
- **Warmup**: 20 steps

## Results

| Metric | Value |
|--------|-------|
| val_loss | 2.1793 |
| val_bpb (pre-quant) | 1.2781 |
| val_bpb (int8+zlib roundtrip) | **1.2907** |
| Artifact size | **15.79 MB** |
| Step time (8×H100) | ~121 ms |

## Setup

```bash
pip install flash-linear-attention einops sentencepiece
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

On H100s, optionally set `FLA_USE_TMA=1` to enable Tensor Memory Accelerator for fla kernels.

## Key Findings

- GDN weights compress less efficiently than transformer weights (~2.8× vs ~3.7× for SOTA submissions), limiting effective model size to ~384d at 16MB
- Routing delta-rule params (a_proj, b_proj) to Adam instead of Muon is critical — Muon's Newton-Schulz orthogonalization destabilizes the recurrence
- Pure SSM without attention is not competitive at 10 min / 16MB vs hybrid approaches, but demonstrates a clean baseline for future SSM work in this setting
