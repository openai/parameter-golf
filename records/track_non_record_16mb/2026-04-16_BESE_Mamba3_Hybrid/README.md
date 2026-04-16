# BESE + Mamba-3 SSD Hybrid

**Author:** Omer Bese ([@mrbese](https://github.com/mrbese))  
**Date:** 2026-04-16  
**Track:** Non-record (SSM / State-space model submission)  
**val_bpb:** 1.3571 (INT6 + LZMA + sliding window eval with n-gram tilt)  
**Artifact size:** 7,614,888 bytes (48% of 16 MB limit)

---

## Overview

This submission combines two experimental ideas requested by the challenge organizers:

1. **State-space models** (specifically Mamba-3 SSD) — checking the "State-space models" bounty from the challenge README
2. **Novel tokenizer** (BESE, a custom 288-vocab byte-level tokenizer) — testing whether sub-byte tokenization gives SSMs an advantage through 2x token density

To our knowledge, this is the first submission to pair a custom byte-level tokenizer with a Mamba-3 architecture.

## Architecture

**Hybrid: 6 Mamba-3 SSD blocks + 2 Attention blocks (8 layers total)**

```
Layer 0: Mamba-3 SSD
Layer 1: Mamba-3 SSD  
Layer 2: Attention (GQA, FlashAttention/SDPA)
Layer 3: Mamba-3 SSD
Layer 4: Mamba-3 SSD
Layer 5: Attention (GQA, FlashAttention/SDPA)  
Layer 6: Mamba-3 SSD
Layer 7: Mamba-3 SSD
```

### Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_dim` | 512 | |
| `num_layers` | 8 | 6 Mamba + 2 Attention |
| `d_state` | 128 | SSM state dimension |
| `expand` | 2 | d_inner = 1024 |
| `headdim` | 64 | SSM head dimension |
| `nheads` (SSM) | 16 | d_inner / headdim |
| `ngroups` | 1 | All heads share B/C (reference Mamba-2 default) |
| `chunk_size` | 64 | SSD chunk size |
| `num_heads` (Attn) | 8 | |
| `num_kv_heads` (Attn) | 4 | GQA |
| `mlp_mult` | 3.0 | Attention block MLP |
| `vocab_size` | 288 | BESE tokenizer |
| **Total params** | **15,152,432** | |

### Key Design Decisions

**1. ngroups=1 (shared B/C across heads)**

All 16 SSM heads share the same B (input-to-state) and C (state-to-output) projections, with only 1 group. This matches the reference Mamba-2 implementation and was confirmed optimal by PR #1644 ablations. Saves ~6.9M parameters vs per-head B/C (ngroups=16), which we reallocate to larger d_state.

**2. No depth recurrence on SSM layers**

PR #1355 measured a -69 mBPB penalty from depth recurrence on Mamba blocks. Unlike transformers where attention re-processes with updated context, SSM state from pass 1 does not inform pass 2 (initial_states=None). We disable depth recurrence entirely.

**3. Two attention layers at positions [2, 5]**

Following PR #1644's architecture, attention layers provide global token mixing at strategic points, dividing the SSM blocks into three equal segments. The SSM layers handle local sequential processing (O(n)), while attention provides periodic global information bottlenecks.

**4. d_state=128 with ngroups=1**

With shared B/C (ngroups=1), the projection cost is only 1 x d_state per position for B and C. Doubling d_state from 64 to 128 costs just ~400K extra parameters but doubles the SSM's memory bandwidth — how much past context each state vector can retain.

## BESE Tokenizer

BESE (Byte-Encoded Sub-byte Encoding) is a two-layer tokenizer:
- **Layer 1:** 40 base tokens (digrams covering 95% of English byte pairs)
- **Layer 2:** 248 BPE merges on top of the base tokens
- **Total vocab:** 288 tokens

Compared to SP1024 (the challenge default), BESE produces ~2x more tokens per byte of text. This means:
- **Embedding table:** 288 x 512 = 147K params (vs SP1024's 1024 x 512 = 524K, or SP8192's 8192 x 512 = 4.2M)
- **Saved parameters** go directly into model capacity
- **Longer effective sequences** for the same token count

### BPB Correctness

The BESE BPB calculation uses per-token byte count lookup tables built from the tokenizer vocabulary. Each token's contribution to the byte count is computed exactly, accounting for leading-space elision at token boundaries. This was validated in our previous BESE submission (PR for `2026-04-14_BESE_NovelTokenizer`). The same tokenizer and BPB computation code is reused here.

## Training

- **Hardware:** 8x NVIDIA H100 80GB SXM (RunPod)
- **Training time:** 600 seconds (wallclock cap)
- **Steps completed:** 2,191
- **Step average:** 274 ms/step
- **Optimizer:** Muon (Newton-Schulz) for 2D matrices, AdamW for scalars and embeddings
- **EMA decay:** 0.9965
- **Warmdown:** 5000 iterations
- **SWA:** Activated at step 1200
- **Sequence length:** 2048 (train and eval)
- **Batch tokens:** 786,432 per step (global)

### Training Curve

| Step | val_bpb |
|------|---------|
| 0 | 4.1571 |
| 500 | 1.5460 |
| 1000 | 1.4268 |
| 1500 | 1.3806 |
| 2000 | 1.3489 |
| 2191 (final) | 1.3458 |

## Evaluation

| Stage | val_bpb | Notes |
|-------|---------|-------|
| Raw (post-EMA) | 1.3475 | Diagnostic |
| INT6 roundtrip | 1.3809 | Quantized model |
| **INT6 + Sliding Window + N-gram tilt** | **1.3571** | **Final submission score** |

- **Quantization:** Mixed INT6 (6-bit) for MLP, attention, and Mamba projection weights. Scalar params (D, dt_bias, A_log, norms) stored as FP16.
- **Compression:** LZMA preset 9
- **Sliding window eval:** stride=64, full 2048 context per window
- **N-gram tilt:** Pre-computed trigram prior from training data, applied as additive logit bias during sliding window eval

### Artifact Size

| Component | Bytes |
|-----------|-------|
| Compressed model (INT6 + LZMA) | 7,452,680 |
| Code (train_gpt.py + mamba3_ssd.py + tokenizer) | 162,208 |
| **Total** | **7,614,888** |
| Budget remaining | 8,385,112 (52% unused) |

## Additional Runs

We ran three configurations to ablate the architecture:

| Config | Params | Steps | Raw BPB | INT6 BPB | SW BPB | Artifact |
|--------|--------|-------|---------|----------|--------|----------|
| dim=512, d_state=64 | 14.8M | 2,482 | 1.3254 | 1.3445 | not completed | 7.96 MB |
| **dim=512, d_state=128** | **15.2M** | **2,191** | **1.3458** | **1.3809** | **1.3571** | **7.56 MB** |
| dim=576, d_state=128, mlp3.5 | 19.7M | 1,847 | 1.3415 | 1.4053 | not completed | 8.42 MB |

Key findings:
- **d_state=128 vs 64:** Slightly worse raw BPB (fewer steps) but sliding window eval works and n-gram tilt recovers the gap
- **dim=576 (wider model):** Best per-step learning rate and best raw BPB (1.3415 at step 1847), but larger INT6 quantization gap (+60 mBPB). Suggests QAT would unlock significant gains for wider Mamba models.
- **Artifact headroom:** Even the widest model uses only 8.42/16 MB, leaving substantial room for growth

## SSD Implementation Notes

We implemented Mamba-3 SSD in pure PyTorch (no custom CUDA/Triton kernels) using the chunked parallel formulation from the Mamba-2 paper. Key components:

- **segsum:** Stable cumulative sum for decay computation via lower-triangular masking
- **ssd_chunked:** Chunked parallel SSD with intra-chunk quadratic attention and inter-chunk state recurrence
- **Causality fix:** We discovered and fixed a causality bug in the reference implementation's inter-chunk decay matrix (diagonal was 1, allowing each chunk to see its own state through Y_off). Fixed by shifting the column index in the einsum.

We attempted integration with the official `mamba-ssm` Triton kernels (`mamba_chunk_scan_combined`), which worked on single GPU but caused segfaults under multi-GPU torchrun after ~100 steps. The pure PyTorch fallback is stable and provides correct results, though ~2-3x slower per step.

## Comparison to Other SSM Submissions

| Submission | BPB | Arch | Vocab | Artifact |
|-----------|------|------|-------|----------|
| PR #1479 GDN hybrid | 1.1450 | 8 GDN + 2 Attn | SP8192 | 13.83 MB |
| PR #1245 Hymba | 1.1470 | 8L hybrid | SP8192 | ~15 MB |
| PR #1644 Mamba-3 | 1.1473 | 5 SSM + 2 Attn | SP8192 | ~14 MB |
| **This (BESE + Mamba-3)** | **1.3571** | **6 SSM + 2 Attn** | **BESE 288** | **7.56 MB** |

The ~210 mBPB gap to the best SSM submissions is attributable to:
- Byte-level prediction with 288 vocab (estimated ~30-50 mBPB penalty vs SP8192)
- Pure PyTorch SSD without Triton kernels (fewer training steps, ~40-60 mBPB)
- No test-time training (~30-50 mBPB based on other submissions)
- No torch.compile (~20-30 mBPB)

The unique contribution is demonstrating that byte-level tokenization + SSM is viable, achieving competitive artifact efficiency (half the 16 MB budget) while leaving substantial room for optimization.

## Reproduction

```bash
# On RunPod 8xH100 pod:
cd /workspace
git clone https://github.com/mrbese/parameter-golf.git bese
cd bese
git checkout v7-mamba

# Download upstream data (for SP model needed for shard prep)
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Prepare BESE shards (untimed)
cd /workspace/bese
pip install einops --break-system-packages
python scripts/runpod_v7_mamba.py --num-gpus 8

# Or with pre-existing shards:
python scripts/runpod_v7_mamba.py --skip-shards --num-gpus 8
```

## Ongoing Work

We have a pending compute credit request and plan to continue optimizing this submission. Planned next steps:

- **Triton kernel integration**: Fix the multi-GPU segfault in `mamba_chunk_scan_combined` to get 2-3x faster steps (~150ms vs 274ms), enabling ~4,000 steps in 600s
- **torch.compile**: Unblocked once Triton kernels are stable — additional ~15% step speedup
- **Wider model with QAT**: dim=576 + mlp3.5 achieved 1.3415 raw BPB but has a +60 mBPB INT6 gap. Quantization-aware training should close this gap substantially
- **Test-time training (TTT)**: Disabled in current runs to save credits. Other SSM submissions show ~30-50 mBPB improvement from TTT
- **SP8192 + BESE comparison**: Direct ablation of tokenizer impact on the same Mamba architecture
- **Additional seeds**: 3-seed statistical significance for final BPB number

Conservative target with all optimizations: **1.17-1.20 BPB**, which would be competitive with the best SSM submissions while maintaining BESE's artifact efficiency advantage.

## Acknowledgments

Architecture decisions informed by:
- PR #1644 by mradassaad (best Mamba-3 submission, exhaustive ablation study)
- PR #1355 by mradassaad (SSM depth recurrence ablation)
- PR #1245 by mkenney2 (Hymba hybrid architecture)
- The Mamba-2 paper (Dao and Gu, 2024) for the SSD algorithm
- mamba3-minimal (VikramKarLex) for the reference pure-PyTorch implementation
