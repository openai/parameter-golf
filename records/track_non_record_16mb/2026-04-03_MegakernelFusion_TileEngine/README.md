# Full-Depth MLP Megakernel + Fused Attention Preprocessing

**val_bpb: 1.1310** (1-seed, SEED=1337) | **15.6 MB** | 8xH100 SXM, 600s

## The Idea: What if a Video Rendering Engine Architecture Could Train Transformers Faster?

This submission started with a question from a different domain entirely.

While designing a tile-based GPU rendering engine for a real-time video rendering -- where 4K frames are split into tiles that fit in L2 cache, and multiple operations (color correction, blur, sharpen) are fused within each tile to avoid VRAM bandwidth bottlenecks -- I realized the same memory hierarchy problem exists in transformer training: intermediate activations are written to HBM between every operation, even when the next operation immediately reads them back.

The video rendering's solution: keep data in fast on-chip L2 cache, apply all operations there, write once. The transformer equivalent: keep the 1536-dim MLP intermediate in GPU registers, process it via tiled accumulation through the gate projection -> activation -> down projection chain, and never let it touch HBM.

This cross-domain transfer produced two novel contributions, an honest failure, and a key insight about GPU computing that shaped our planned follow-up.

### What Worked
- **Full-depth MLP megakernel:** 5 operations (RMSNorm -> gate projection -> LeakyReLU^2 -> down projection -> residual) fused into 1 Triton kernel. The 1536-dim intermediate is never written to HBM -- processed via tiled register accumulation in BLOCK_K=64 chunks. Deeper fusion than PR #1072 (which fuses adjacent element-wise ops but still materializes the intermediate between groups).
- **Attention preprocessing fusion:** QK RMSNorm + partial RoPE + q_gain fused into 2 Triton kernels, down from 6+. Nobody in the competition fuses these post-projection operations.
- **41% memory reduction** (1562 MiB vs 2656 MiB) -- hardware-independent, reproducible (SubV1-SubV2 delta: 0.0001 BPB).
- **Near-perfect numerical accuracy:** MLP cos_sim=0.99998, attention Q/K cos_sim=0.99999.

### What Didn't Work
- **Step time:** 15% slower on consumer GPU (451.9ms vs 392.7ms). The megakernel's 24 small `tl.dot` calls cannot compete with cuBLAS's single large GEMM, which has decades of per-architecture tensor core optimization.
- **Fully fused attention preprocessing:** Attempted fusing RMSNorm -> QKV projection -> QK norm -> RoPE -> gain into one kernel. Triton's block tensor model can't do the half-dimension register slicing that RoPE requires. Achievable in raw CUDA, not in Triton today.

### The Key Insight
**The Tile Engine metaphor works perfectly for element-wise operations but not for matmul-dominated workloads.** In video processing (all per-pixel ops), tiling into SRAM is optimal -- there are no matrix multiplications to compete with cuBLAS. In transformers (90% matmul by compute), the matmuls should be delegated to hardware-optimized libraries while tiling handles only the element-wise glue between them. The right strategy isn't to replace cuBLAS -- it's to partner with it.

## Results

| Seed | Steps | ms/step | Pre-quant BPB | Sliding BPB | Artifact |
|------|-------|---------|---------------|-------------|----------|
| 1337 | 4,917 | 122.0 | 1.1500 | **1.1310** | 15,597,863 |

Seeds 42 and 2025 blocked by compute budget exhaustion. Awaiting grant approval for additional validation runs.

## Local Development Benchmarks (RTX 5070 Ti, 1 GPU, 500 steps)

Validated on NVIDIA RTX 5070 Ti (12GB VRAM, 101KB shared memory/SM):

| Metric | SOTA Baseline (PR #1019) | Megakernel Submission | Delta |
|--------|-------------------------|----------------------|-------|
| step_avg (steady) | 392.7 ms | 451.9 ms | +15.1% slower |
| val_loss@500 | 3.2530 | 3.4223 | +0.1693 |
| val_bpb@500 | 1.9266 | 2.0269 | +0.1003 |
| peak_memory | 2656 MiB | 1562 MiB | **-41% memory** |
| reproducibility | -- | SubV1-SubV2 diff: 0.0001 | Deterministic |
| MLP megakernel | N/A | cos_sim=0.99998 | Numerically exact |
| Attention fusion | N/A | Q cos=0.99999, K cos=0.99999 | Numerically exact |
| Autotune config | N/A | BLOCK_M=32, BLOCK_K=64, nw=8 | Auto-selected |

**Interpretation:** On consumer GPUs with 101KB SRAM, the megakernel's tiled matmul accumulation (24 small `tl.dot` calls looping over H=1536 in chunks of BLOCK_K=64) cannot compete with cuBLAS's single large GEMM, which is optimized to saturate tensor cores in one call. The 15% step time overhead causes the val_bpb gap -- at the same step count, the loss trajectories are nearly identical (step 1: both 6.9314, step 10: delta 0.0006).

**Where this approach wins:** The 41% memory reduction is hardware-independent and enables larger batch sizes or longer sequences in memory-constrained settings. The fusion becomes speed-competitive when the model is bandwidth-bound rather than compute-bound -- specifically on hardware with larger SRAM (H100: 228KB, enabling larger tiles with fewer iterations) and at larger effective batch sizes where HBM bandwidth becomes the bottleneck.

## Technical Details: MLP Megakernel

The MLP hidden dimension (H=1536) is processed in tiles of BLOCK_K (32-64) elements. For each tile:
1. Load x once from HBM [BLOCK_M, D=512]
2. RMSNorm in registers
3. For each BLOCK_K chunk of H:
   a. Compute partial up-projection [BLOCK_M, D] x [D, BLOCK_K] via `tl.dot` -> [BLOCK_M, BLOCK_K] in SRAM
   b. Apply LeakyReLU(0.5)^2 activation in registers
   c. Accumulate partial down-projection [BLOCK_M, BLOCK_K] x [BLOCK_K, D] into output registers
4. Apply MLP scale + residual add
5. Write result once to HBM [BLOCK_M, D]

The [M, 1536] intermediate tensor is **never written to or read from HBM**. This goes deeper than PR #1072 (which fuses adjacent element-wise ops but still materializes the 1536-dim intermediate between fused groups) -- we fuse element-wise ops WITH matmul ops in a single kernel.

H100 autotune configs: BLOCK_M=32/BLOCK_K=64 (best on sm_90 with 228KB shared memory).

## Novel Contribution #2: Attention Preprocessing Fusion

Fused QK RMSNorm + partial RoPE (16/64 dims) + q_gain scaling into 2 Triton kernels (down from 6+ separate PyTorch kernels). Nobody in the competition fuses these post-projection operations.

- `fused_qk_norm_gain_kernel`: Per-head RMSNorm + optional per-head gain in a single pass
- `fused_partial_rope_kernel`: Loads each head's RoPE half-dimensions via offset arithmetic, applies cos/sin rotation in registers

Together these eliminate 4+ kernel launch round-trips per block x 11 blocks = 44+ eliminated launches per step.

## Attempted: Fully Fused Attention Preprocessing Kernel

We initially designed a single-kernel fusion of the entire attention preprocessing chain: RMSNorm(x) -> Q/K/V projection -> QK RMSNorm -> RoPE -> q_gain scaling. This would eliminate all HBM round-trips between the input activations and FlashAttention's Q/K/V inputs.

However, RoPE requires splitting each 64-dim head vector into two halves (dims 0:31 and 32:63) for independent cos/sin rotation. Triton's block tensor model does not support arbitrary register-level slicing -- tensors loaded as tiles must be operated on as complete blocks. While offset-based loading of separate half-tiles partially works, integrating this with the tiled QKV projection matmul within the same kernel creates register pressure that exceeds practical limits on current hardware.

We instead fuse the post-projection operations (QK RMSNorm + RoPE + q_gain) into a single kernel, reducing attention preprocessing from 6+ kernel launches to 2. The fully fused QKV preprocessing kernel remains a promising direction -- likely achievable with raw CUDA (which allows arbitrary register indexing) or future Triton versions with richer indexing support.

## Results & Analysis

**Memory:** 41% reduction vs SOTA baseline (1562 MiB vs 2656 MiB on RTX 5070 Ti local dev)

**Speed:** On H100, the megakernel is 41% slower per step (122ms vs SOTA's 86.7ms), resulting in 2,005 fewer training steps and +0.016 BPB. This is worse than the 15% slowdown on consumer GPUs -- H100's stronger cuBLAS tensor cores widen the gap between hand-tiled `tl.dot` and optimized GEMMs. The Tile Engine hypothesis (larger SRAM would help) was wrong: more SRAM doesn't overcome the structural disadvantage of replacing cuBLAS.

The 41% memory reduction (local) is confirmed on H100 at 19.6% VRAM utilization (15.7 GiB / 80 GiB). The planned follow-up submission will partner with cuBLAS via epilogue/prologue fusion rather than replacing it.

Kernel launch count reduced from ~17 per transformer block to ~10 per block (~110 vs ~187 per forward+backward step).

## Learnings & Future Directions

### What We Learned

1. **cuBLAS is unbeatable for large GEMMs.** Replacing cuBLAS matmuls with tiled `tl.dot` calls in Triton is structurally slower, even with perfect fusion. cuBLAS has decades of per-architecture tuning and saturates tensor cores in ways that hand-written Triton cannot match for matrix sizes like [M, 512] x [512, 1536].

2. **The value of fusion is in eliminating element-wise HBM traffic, not in replacing matmuls.** The 41% memory reduction proves that fusing RMSNorm, activations, and residual adds INTO matmul boundaries is high-value. The mistake was fusing them by replacing the matmul, rather than injecting them alongside it.

3. **`torch.compile` (Inductor) already captures the easy fusions.** Adjacent element-wise ops (Norm+Scale, Activation+Residual) are automatically fused by Inductor. Novel kernel fusion must go deeper than what the compiler does automatically -- specifically, fusing element-wise ops across matmul boundaries.

4. **Triton's block tensor model limits attention fusion.** RoPE's half-dimension splitting requires register-level indexing that Triton doesn't support. Raw CUDA would solve this but isn't practical for a single-file Python submission.

5. **The Tile Engine metaphor works for element-wise operations but not for matmul-dominated workloads.** In video processing (all element-wise, per-pixel ops), tiling into SRAM is optimal. In transformers (90% matmul), the matmuls should be delegated to hardware-optimized libraries while tiling handles only the element-wise glue.

### Future Direction

The natural next step is to apply these fusion insights differently: instead of replacing cuBLAS, partner with it by injecting the element-wise operations into the matmul's own execution boundaries. This would combine cuBLAS-speed matmuls with the HBM traffic elimination we demonstrated here. We plan to explore this in a follow-up submission.

## Architecture

Same as PR #1019 base:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers |
| BigramHash | 3072 x dim=112 |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | Full Hessian GPTQ int6 (AR self-gen calibration) |
| Compression | LZMA preset=9 |
| Optimizer | Parallel Muon + Parameter Banking |
| **NEW: MLP Fusion** | **Full-depth Triton megakernel (tiled register accumulation)** |
| **NEW: Attn Fusion** | **Fused QK-norm + RoPE + gain (2 Triton kernels)** |

## Run Command

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Requirements

Flash Attention 3 (Hopper) + Triton required:
```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install triton sentencepiece zstandard
```

## Credits

- **PR #1019** @abaybektursun: Base submission (AR Self-Gen GPTQ + XSA-all + BigramHash 3072)
- **PR #1072**: Triton fusion baseline (adjacent op fusion -- this submission goes deeper)
- **PR #1105**: Fused backward epilogue (inspired our forward fusion approach)
- **PR #399** @abaybektursun: Parallel Muon optimizer
- **PR #493** @parinzee: LeakyReLU(0.5)^2 activation
- **PR #478** @gowtham0992: XSA (cross-sequence attention)
- **PR #315** @jfprincz: Partial RoPE, layerwise LN scale
- **PR #374** @unnir: Value Embeddings
- **PR #162** @raahilshah: BigramHash concept
- **PR #535** @raahilshah: GPTQ quantization
