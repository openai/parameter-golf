# Record: Embedding Decoupling + Extended Parallel Residuals + Tuned Quantization

**val_bpb = 1.0850** (TTT) | **~16.44 MB** (over budget) | 8xH100 SXM, 600s

> **Status: Over budget — experimental record only.** This experiment confirmed the tied-embedding bottleneck hypothesis and informed the design of follow-up approaches. Published for community learning.

## Motivation

Starting from the ImprovedParallelResiduals baseline (val_bpb=1.0744), we set out to answer: **why are the first and last transformer blocks so much weaker than the middle blocks?**

## Discovery: The Tied Embedding Bottleneck

### What we found

We analyzed the trained weight matrices of the baseline model and discovered a consistent pattern across all metrics:

**The model's effective contribution follows a symmetric V-shape** — blocks at both boundaries (first and last) contribute 2-4x less than the middle blocks. We measured this through the product of each block's output projection norm and its learned scale factor.

The first block barely transforms its input (low scales, high Q-gain variance across attention heads suggesting the heads can't agree on useful patterns). The last block is even weaker — its MLP scale is the lowest of all blocks at 0.15, and in the parallel residual routing, its attention lane residual is essentially zero (-0.01).

### Why it happens

With tied embeddings, the same weight matrix `tok_emb.weight` serves as both:
- **Input lookup**: `x = tok_emb(input_ids)` — token identity representation
- **Output classifier**: `logits = F.linear(hidden, tok_emb.weight)` — vocabulary projection

This creates a fundamental tension. The embedding must be a compromise between a good lookup table and a good linear classifier. The first block receives this compromise representation and can't freely transform it. The last block must produce output that aligns with this compromise for the logit projection to work.

**Evidence from the weight analysis:**

- **Raw embedding dependency (resid_mix x0_weight)**: Only the first (+0.11) and last (+0.08) blocks significantly mix in the raw embedding signal. All middle blocks have x0_weight ≈ 0 — they've fully escaped the embedding space.

- **Skip connection suppression**: The deepest U-net skip (connecting the first block's output to the last block's input) has the lowest effective weight (0.06) — the model actively gates off the raw embedding signal from reaching the output end.

- **Attention head confusion**: The first block has the highest Q-gain variance (std=1.17) — heads disagree on what's useful in the raw embedding space. Middle blocks have much lower variance (0.29-0.50) — heads are specialized and confident.

- **Scale decay**: The scales form a clear V-pattern: first block attn=0.38/mlp=0.33, middle peak attn=0.60/mlp=0.86, last block attn=0.23/mlp=0.15.

### The rate-distortion context

We also analyzed the quantization bottleneck. At the 16MB budget with int6 GPTQ + brotli compression, the optimal clip_sigmas is approximately 12.3 (confirmed by sweeping the full rate-distortion curve). The quantization error is ~12% relative and is uniform across all layers — there's no per-layer fix possible. The 16MB budget determines the error floor.

## Experiment Lineage

### Stage 1: Dimension-based decoupling (embed_dim ≠ model_dim)

**Approach**: Set `embedding_dim=384` while keeping `model_dim=512`. This activates learned projection layers (`embed_proj: 384→512`, `head_proj: 512→384`) that decouple the internal representation from the tied embedding space.

**Results**: The last block's effective contribution jumped +42.5%. The parallel routing on the penultimate block normalized from 99% cross-routing to 45% (balanced). The capacity distribution flattened dramatically — the min/max ratio went from 3.7x to 2.3x.

**But**: The smaller embedding cost 655K parameters (the tok_emb shrank from 8192×512 to 8192×384). This 7.4% total capacity loss caused the pre-quantization BPB to regress. The raw model quality degraded even though the architecture was healthier.

**Insight**: The projections learned near-perfect orthogonal rotations (all singular values ≈ 1.000 ±0.002). The model doesn't need to compress or expand — it just needs a different basis. A full dimension change is overkill.

### Stage 2: Finding the sweet spot (embed_dim=448)

**Approach**: Reduce the decoupling gap to minimize parameter loss while retaining the structural benefit.

**Results**: With embed_dim=448 (64-dimensional gap instead of 128), the last block still activated (+51% vs parent). The min/max ratio reached 2.2x (best uniformity). Only 65K parameters lost.

**But**: The projection layers stored as fp16 passthrough added 917 KB to the artifact, pushing total size to 16.28 MB. The pre-quant BPB was still worse than the parent by 0.006.

### Stage 3: This experiment — aggressive combo (embed_dim=416 + parallel_start=7 + clip_sigmas=12.0 + TTT freeze)

**Approach**: Combine multiple improvements:
- `embed_dim=416`: moderate decoupling (96-dimensional gap)
- `PARALLEL_RESIDUAL_START=7`: extend two-lane routing to 4 blocks (7-10) since the last block is now active
- `MATRIX_CLIP_SIGMAS=12.0`: tighter quantization clipping to reduce error (using saved headroom from smaller tok_emb)
- `TTT_FREEZE_BLOCKS=2`: freeze the two weakest blocks during test-time training to focus gradient on activated blocks

**Results**:
- Pre-quant BPB: 1.0915 (worse than parent's 1.0821)
- TTT BPB: 1.0850 (worse than parent's 1.0744)
- Artifact: 16.44 MB (over the 16 MB budget)
- The capacity distribution was the most uniform ever (min/max=2.1x)
- But total effective capacity dropped to 297 (parent: 337) — the model spread itself too thin

**Why it failed**:
1. Lower clip_sigmas is counterproductive with decoupled embeddings — tighter clipping increases the entropy of quantized values, causing larger compressed size. The artifact bloated by 0.5 MB instead of shrinking.
2. Starting parallel residuals at block 7 caused blocks 5-6 (previously the strongest) to lose significant capacity. Block 6 dropped from 41.5 to 27.1 effective contribution.
3. TTT freeze on blocks 0-1 actually reduced TTT effectiveness (TTT gain was only -0.0005 vs parent's -0.0012). The frozen blocks still needed adaptation.
4. The fp16 passthrough projections (852 KB) consumed the headroom that was supposed to fund better quantization.

## Key Takeaways for the Community

### What we learned about tied embeddings
1. **The V-shaped capacity profile is real and measurable** — first and last blocks are structurally constrained by the dual-purpose embedding
2. **Decoupling activates boundary blocks** — even a 64-dimensional gap is enough to free the last block (+50% effective contribution)
3. **But parameter loss matters more than architectural elegance** — losing 65K-655K params from the embedding hurts BPB more than activating weak blocks helps
4. **The embedding projections learn near-identity rotations** — the model needs a change of basis, not a dimensionality change

### What we learned about the 16MB budget
1. **clip_sigmas and artifact size are tightly coupled** — lower clip = less quantization error but WORSE compression. You can't have both.
2. **fp16 passthrough is expensive** — even 200K params stored as fp16 adds ~400 KB pre-compression. Stay under the 65K param threshold for passthrough tensors.
3. **The rate-distortion optimal point is clip_sigmas ≈ 12.3** with the standard architecture. Decoupling shifts this because the projection matrices are uncompressible fp16 overhead.

### What we learned about TTT
1. **Freezing early blocks during TTT doesn't help at the frontier** — the model needs all blocks to adapt, even weak ones
2. **More uniform capacity doesn't automatically mean better TTT** — the pre-quant gap dominates. TTT can't recover a 0.009 BPB pre-quant regression.

### The design tenets (confirmed by experiments)
1. **Steps per second is everything** — any architectural addition that slows training by even 5% needs to deliver >0.005 BPB to justify itself
2. **The 16MB budget is a rate-distortion problem** — you can't reduce quantization error without growing the artifact. The architecture must produce compressible weights.
3. **Eval-time adaptation recovers what training and quantization lose** — but it can't compensate for a fundamentally worse pre-quant model

### The promising direction we didn't complete
A **residual low-rank projection** (`x → x + B(Ax)`, rank-32) adds only 65K parameters at 128 KB fp16, keeps tok_emb at full 8192×512, keeps model_dim=512 everywhere, and initializes as identity (A=small random, B=zeros). This avoids all the problems of the dimension-change approach while still providing the basis rotation that decouples boundary blocks. This is the follow-up experiment.

## Architecture

11L × 512d × 8H / 4KV, MLP 4x, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings (embed_dim=416), logit softcap=30.0. Depth recurrence: layers 3-5 (num_loops=2, 17 virtual layers). Parallel residuals from layer 7 (4 blocks with two-lane routing). Skip gates (sigmoid-gated U-Net connections). CUTLASS EVT fusion for throughput.

## Results

| Eval Stage | BPB |
|---|---|
| Pre-quantization post-EMA | 1.0915 |
| Quantized (basic) | 1.1013 |
| Quantized sliding window | 1.0845 |
| Quantized TTT (legal, score-first) | 1.0850 |

| Metric | Value |
|---|---|
| Training steps | 4,727 |
| Training time | 588s |
| Peak memory | 40,177 MiB |
| Artifact size | 16,439,738 bytes (over 16MB budget) |

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ENABLED=1 HASH_EMBED_ENABLED=1 TTT_LR=0.01 TTT_FREEZE_BLOCKS=2 \
  MUON_MOMENTUM=0.97 PARALLEL_RESIDUAL_START=7 EMBEDDING_DIM=416 MATRIX_CLIP_SIGMAS=12.0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The `cutlass_evt_fusion/` directory should live alongside `train_gpt.py` in the directory you run from.

## Credits

- **@msisovic** — Improved Parallel Residuals baseline (PR parent)
- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R + depth recurrence
- **@dexhunter** — 3-layer depth recurrence, legal TTT on SP8192
- **@abaybektursun** — Score-first TTT framework
- Analysis and embedding decoupling experiments by this submission

## Included Files

- `README.md` (this file)
- `run.sh`
- `train_gpt.py`
- `cutlass_evt_fusion/`
- Training logs
