# S4D-Lin SSM Hybrid: Fixing Why Mamba Failed in Parameter Golf

**PR #1013 | Non-Record Submission (Research Contribution)**
**Author:** Himanshu Dongre ([@himanshudongre](https://github.com/himanshudongre))
**Base:** Merged leader PR #549 (1.1194 bpb)
**Companion:** [JEPA-LM PR #1012](https://github.com/openai/parameter-golf/pull/1012) (the research that preceded this work)
**Duration:** 3 days of research, ~$47 in compute across local MPS + 8xH100 RunPod
**Best result:** 1.1499 bpb pre-quantization | 1.1682 bpb post-GPTQ-int5 | **Gap:** +0.049 bpb

---

## The Short Version

I built the first SSM hybrid in Parameter Golf that actually trains, exports, and evaluates without throughput penalty. Previous SSM attempts (Hymba, PR #599 at 1.1828 bpb) used Mamba's selective scan, which requires custom CUDA kernels that `torch.compile` can't fuse -- resulting in a 3.4x throughput penalty that kills any quality advantage. My approach replaces Mamba's selective scan with **S4D-Lin**: a simplified diagonal state-space model that computes as a standard `F.conv1d` -- pure PyTorch, fully compilable, zero custom kernels.

The throughput problem is solved. But a new insight emerged: **attention is more valuable than SSM in the lower layers at this scale**. The 0.049 bpb gap isn't from throughput (we match baseline at 116ms/step) -- it's from the quality that self-attention provides in the first 2 layers. This finding contradicts our local small-scale tests, which showed SSM as dramatically better (-18% cross-entropy at dim=192). The gap between small-scale validation and full-scale GPU results is the most important takeaway from this work.

If you're considering SSMs for Parameter Golf, this document will save you time and money: it covers what works, what doesn't, and exactly why.

---

## Table of Contents

1. [How I Got Here](#how-i-got-here)
2. [The Architecture: S4D-Lin SSM](#the-architecture-s4d-lin-ssm)
3. [Local Experiments (Promising)](#local-experiments-promising)
4. [GPU Results (Disappointing)](#gpu-results-disappointing)
5. [Analysis: Why the Gap?](#analysis-why-the-gap)
6. [What I Tried That Didn't Work](#what-i-tried-that-didnt-work)
7. [Lessons for Future SSM Work](#lessons-for-future-ssm-work)
8. [Reproducing These Results](#reproducing-these-results)

---

## How I Got Here

### Chapter 1: The Enforcement Sweep

My first contribution to Parameter Golf was PR #846 -- a two-pass n-gram rescoring technique that achieved 0.1434 bpb. It was a significant improvement over the baseline, and I was proud of it. Then the enforcement sweep happened (Issue #677). Thirty-three PRs were closed for rule violations, including mine. The two-pass approach was deemed illegal because it used oracle selection between scoring passes.

I was disappointed but understood the ruling. More importantly, it forced me to rethink my approach entirely. Instead of trying to game the evaluation, I decided to pursue **pure architectural innovation** -- changes that live entirely in the model architecture, with zero eval-time tricks. No n-grams, no TTT, no rescoring. Just a different way of building the neural network.

### Chapter 2: JEPA-LM (Dead End)

My first idea was JEPA-LM -- applying Joint Embedding Predictive Architecture to language modeling. The concept: instead of predicting exact token distributions, predict *representations* of future tokens in a learned latent space. The predictor learns compressed patterns while the standard LM head handles the actual token prediction.

Local tests on synthetic Markov chain data showed a dramatic **19.5% cross-entropy improvement**. I was excited. Then I tested on real English text: **-0.24% improvement with +40% throughput overhead**. The synthetic data result was completely misleading -- Markov chains have simple repetitive patterns that JEPA's representation learning could exploit, but real language doesn't work that way.

**Lesson learned:** Always validate on real text. Synthetic benchmarks can be wildly misleading.

### Chapter 3: Why SSM?

After JEPA failed, I surveyed the architecture landscape: Mixture of Experts, Mixture of Depth, Monarch matrices (tried locally, inconclusive), and State Space Models. SSMs stood out because:

1. **O(n) complexity** vs attention's O(n^2) -- at seq_len=2048, this is a real throughput advantage
2. **Multi-scale temporal modeling** -- exponential kernels capture patterns at different timescales
3. **Previous failure was implementation, not theory** -- Hymba (PR #599) failed at 1.1828 because of Mamba's CUDA kernels, not because SSMs can't model language

The key insight from PR #831 is critical: at 16MB/600s, **every millisecond of overhead costs ~7 training steps**. Mamba's 3.4x overhead meant losing ~2000 training steps. No quality advantage can survive that. But what if we could build an SSM with *zero* overhead?

That's what S4D-Lin is.

---

## The Architecture: S4D-Lin SSM

### Why S4D-Lin, Not Mamba

Mamba (S6) uses **input-dependent state transitions** (selective scan), which means:
- Custom CUDA kernels (`mamba-ssm` package)
- Can't be fused by `torch.compile`
- High kernel launch overhead at dim=512
- 3.4x throughput penalty (measured by PR #831)

S4D-Lin uses **fixed diagonal transitions** that unfold into a causal conv1d:

```
kernel[d, t] = C[d] * exp(-rate[d] * t)    for t in [0, kernel_size)
```

This is just `F.conv1d(input, kernel, groups=dim)` -- a standard PyTorch op that `torch.compile` handles perfectly. No custom kernels, no extra dependencies.

### Multi-Scale Temporal Receptive Fields

Different channels get different decay rates, log-spaced from 0.05 to 2.0:
- **Slow-decaying channels** (rate ~ 0.05): remember 50+ tokens back, capture phrase-level patterns
- **Fast-decaying channels** (rate ~ 2.0): focus on last 2-3 tokens, capture character/subword-level patterns

This is inspired by the HiPPO framework from the original S4 paper -- the key idea is that different time scales need different memory horizons.

### Block Design

```
Input x --> RMSNorm --> in_proj (dim -> 2*inner_dim) --> [gate | input]
                                                          |       |
                                                       sigmoid   causal conv1d (groups=inner_dim, kernel=64)
                                                          |       |
                                                        gate * conv_out --> out_proj (inner_dim -> dim)
                                                                              |
                                                                         + residual --> MLP (standard)
```

The gate mechanism is crucial -- it allows the model to learn when to rely on the SSM's temporal pattern vs. passing through the residual.

### Integration with SOTA Stack

The SSM block is a drop-in replacement for TransformerBlock in the lower layers:

- Uses `CastedLinear` for all projections (GPTQ-compatible)
- `resid_mix` for x0-mixing (same as upper transformer blocks)
- `ln_scale_factor` for layer-indexed normalization scaling
- SSM kernel and scale parameters routed to scalar optimizer (not Muon)
- Classify tensor updated: `ssm_proj` added to `INT6_TENSOR_CLASSES` for proper quantization

Activated via environment variable: `SSM_LAYERS=2` replaces the first 2 layers with SSM blocks.

### Parameter Budget

| Component | Params per block | Note |
|-----------|-----------------|------|
| SSM in_proj | 512 x 1024 = 524K | Gate + input projection |
| SSM out_proj | 512 x 512 = 262K | Output projection |
| SSM kernel | 512 x 64 = 32K | Learned exponential kernels |
| MLP (same as Transformer) | ~1,575K | Standard LeakyReLU^2 MLP |
| Control (norms, scales, mix) | ~2K | Small overhead |
| **Total** | **~2,395K** | vs ~2,361K for TransformerBlock |

Net parameter change from replacing 2 TransformerBlocks: **+68K params** (negligible, 0.25%).

---

## Local Experiments (Promising)

All local experiments run on Mac Mini M4 (MPS backend), dim=192, 6 layers.

### Throughput Scaling

The key experiment: SSM overhead decreases (and reverses) as sequence length grows.

| seq_len | SSM-Light (2 SSM + 4 attn) | SSM-First (3 SSM + 3 attn) |
|---------|---------------------------|---------------------------|
| 256     | +5.6% slower              | +9.5% slower              |
| 512     | **-6.1% faster**          | **-8.6% faster**          |
| 1024    | **-16.1% faster**         | **-23.5% faster**         |

This confirmed the O(n) vs O(n^2) advantage. At competition seq_len=2048, SSM layers should be ~25% faster than attention layers.

### Quality at seq_len=512

Trained 2000 steps on real English text (from FineWeb subset):

| Config | Final CE | vs Baseline | Speed vs Baseline |
|--------|----------|-------------|-------------------|
| Pure Transformer | 1.6192 | -- | -- |
| **SSM-Light (2 SSM + 4 attn)** | **1.3252** | **-18.2%** | **-5.7% faster** |
| SSM-First (3 SSM + 3 attn) | 1.3309 | -17.8% | -8.2% faster |

These results were incredibly promising: **both faster AND dramatically better quality**. The SSM hybrid converged to the pure Transformer's final quality at half the training steps.

### torch.compile Verification

- Forward pass: compiles cleanly with `fullgraph=True`
- Backward pass: has MPS-specific stride issue in `convolution_backward` (expected to work on CUDA)
- All operations are standard PyTorch -- no graph breaks expected on CUDA

At this point, I was confident enough to spend GPU credits.

---

## GPU Results (Disappointing)

### Setup
- 8x H100 SXM5 80GB (RunPod)
- Merged leader config: 11 layers (2 SSM + 9 Transformer), dim=512
- `full_gptq_int5` quantizer + lzma compression (required to fit in 16MB)
- Single seed (1337), 600s wallclock

### Training Trajectory

| Step | train_loss | val_bpb | train_time |
|------|-----------|---------|------------|
| 0 | -- | 4.1031 | 0ms |
| 500 | 2.3836 | -- | 58s |
| 1000 | 2.2563 | -- | 116s |
| 2000 | 2.0437 | -- | 233s |
| 3000 | 2.1218 | -- | 349s |
| 4000 | 1.9131 | 1.1853 | 465s |
| **5146** | -- | **1.1501** | **600s** |

**Post-EMA diagnostic:** val_bpb = 1.1499

Training ran smoothly: no NaN, no OOM, clean convergence. Step time averaged 116ms (matching baseline ~117ms). The throughput promise was delivered.

### Export and Quantization

| Metric | Value |
|--------|-------|
| GPTQ int5 layers | 62 |
| Total int5 params | 25,952,256 |
| Artifact size | 13,007,924 bytes (13.0 MB) |
| GPTQ MSE | 0.031 |
| **Post-quantization val_bpb** | **1.1682** |

### The Three Failed Attempts

Before the successful export, I hit two export failures that consumed GPU time:

1. **Attempt 1:** Default `rowclip_int6` quantizer. SSM proj weights classified as `ssm_proj` (not in `INT6_TENSOR_CLASSES`), fell to int8. Model 17.5MB -- exceeded 16MB cap.
2. **Attempt 2:** Added `ssm_proj` to `INT6_TENSOR_CLASSES`. Still used `rowclip_int6`. Model 17.16MB -- still over.
3. **Attempt 3:** Switched to `full_gptq_int5` + lzma compression. Model 13.0MB -- success. But int5 quantization degraded SSM weights more than expected (0.018 bpb quantization loss).

**Lesson:** If you're adding novel weight types, think about quantization from the start. Know your quantizer. The merged_leader preset defaults to `rowclip_int6` but the actual competition submissions use `full_gptq_int5`.

### Final Comparison

| Metric | SSM Hybrid | SOTA Baseline (PR #549) |
|--------|-----------|------------------------|
| Pre-quant val_bpb | 1.1499 | ~1.10 (estimated) |
| Post-quant val_bpb | **1.1682** | **1.1194** |
| Steps in 600s | 5,146 | ~5,000 |
| Step time | 116ms | ~117ms |
| Artifact size | 13.0 MB | ~15 MB |
| Custom CUDA kernels | 0 | 0 |

**Gap: +0.049 bpb. Not competitive.**

---

## Analysis: Why the Gap?

### The Scale Discrepancy

The most important finding: **local tests at dim=192 are unreliable predictors of full-scale performance (dim=512).**

| Scale | SSM vs Transformer |
|-------|--------------------|
| dim=192, 6L, seq=512 | SSM **-18.2% better** |
| dim=512, 11L, seq=2048 | SSM **+2.7% worse** |

Why the reversal? Several hypotheses:

1. **Attention capacity scales better with dimension.** At dim=192, attention heads have 48 dimensions each (4 heads x 48). At dim=512, they have 64 dimensions (8 heads x 64). Larger head dimensions mean richer attention patterns that SSM can't replicate.

2. **Lower layers need global context.** The SOTA architecture uses attention even in the lowest layers. SSM with kernel_size=64 has a fixed receptive field -- it literally cannot look at tokens beyond position 64. Attention has no such limit.

3. **The throughput advantage was smaller than expected.** SSM layers are O(n) but `F.conv1d` at dim=512 is memory-bandwidth bound, not compute bound. The practical speedup was only ~1ms/step (116ms vs 117ms), yielding ~150 extra training steps -- not enough to compensate for the quality gap.

### Quantization Sensitivity

SSM weights appear more sensitive to aggressive quantization:
- GPTQ MSE for SSM layers: ~0.031
- Typical GPTQ MSE for attention layers: ~0.005-0.01
- Quantization added 0.018 bpb to the SSM model vs typical ~0.005-0.01 for pure transformers

The exponential kernel structure may not compress well to int5 -- the smooth decay patterns might require higher precision to preserve.

---

## What I Tried That Didn't Work

### JEPA-LM (3 days of research, $0 compute)
- **Idea:** Predict future token representations in latent space alongside standard LM loss
- **Local result:** -19.5% CE improvement on synthetic Markov chain data
- **Real text result:** -0.24% CE with +40% throughput overhead
- **Verdict:** Synthetic data is a trap. JEPA's representation learning exploits repetitive Markov patterns that don't exist in natural language.

### Monarch Matrices (1 day, $0 compute)
- **Idea:** Replace dense linear layers with structured butterfly matrices for parameter efficiency
- **Result:** torch.compile works, but quality results were inconclusive at small scale
- **Verdict:** Abandoned in favor of SSM after throughput scaling results

### Two-Pass N-gram Rescoring (PR #846, CLOSED)
- **Idea:** First pass scores with neural model, second pass rescores with n-gram cache
- **Result:** 0.1434 bpb (dramatic improvement)
- **Verdict:** Closed in enforcement sweep. Oracle selection between passes violates Issue #677 rules. This failure motivated the shift to pure architectural innovation.

---

## Lessons for Future SSM Work

1. **The throughput problem IS solvable.** S4D-Lin proves that SSMs can run at transformer speed in this competition. The key is avoiding custom CUDA kernels.

2. **The quality problem needs a different approach.** Replacing attention with SSM hurts quality. Instead, consider:
   - SSM as an **auxiliary module alongside attention** (SSM + Attention in parallel, not SSM instead of Attention)
   - SSM for **very long-range patterns** only (larger kernel, acting as a global context module)
   - **Learnable kernel mixtures** instead of fixed exponential decay (more expressiveness without custom kernels)

3. **Quantization-aware design is essential.** Any novel weights must be quantizable to int5/int6. Design the quantization strategy before writing the architecture code.

4. **Don't trust small-scale validation.** A -18% improvement at dim=192 meant nothing at dim=512. The minimum reliable test requires at least dim=384+ and 8+ layers. This costs real compute, but it's cheaper than a failed GPU run.

5. **Know your export pipeline.** I wasted 2 out of 3 GPU runs on export failures that could have been avoided by understanding the quantizer configuration upfront. `merged_leader` defaults to `rowclip_int6` but actual submissions need `full_gptq_int5`.

---

## Reproducing These Results

### Environment
- 8x H100 SXM5 80GB (RunPod, parameter-golf template)
- Network volume with FineWeb 10B SP1024 dataset pre-loaded

### Training Command
```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MODEL_PRESET=merged_leader \
RUN_PROFILE=full_8gpu_600s \
SSM_LAYERS=2 \
SSM_KERNEL_SIZE=64 \
TTT_ENABLED=0 \
NGRAM_EVAL_ENABLED=0 \
INT6_TENSOR_CLASSES=attention,mlp,local_mixer,ssm_proj \
EXPORT_QUANTIZER=full_gptq_int5 \
EXPORT_COMPRESSOR=lzma \
SEED=1337 \
torchrun --nproc_per_node=8 train_gpt.py
```

### Expected Output
- ~5100 training steps in 600s
- val_bpb ~1.15 pre-quantization, ~1.17 post-GPTQ-int5
- Artifact ~13MB (well under 16MB)
- No NaN, no OOM, clean convergence

### Cost
- ~$8-12 per seed on RunPod 8xH100
- ~15 min total per seed (10 min train + 5 min GPTQ calibration + eval)

---

## Acknowledgments

- The merged leader stack (PR #549) provided the foundation
- PR #831 by the GatedDeltaNet team for the critical analysis of why novel architectures fail at 16MB
- The Hymba team (PR #599) for the first SSM attempt -- their failure motivated this work
- PR #363 (Depth Recurrence) for the template on how to write a good negative-result PR
- The S4/S4D papers (Gu et al.) for the diagonal state-space model framework

---

*Total compute spent on this research: ~$47 across all approaches (JEPA + Monarch + SSM, local + GPU). Not a lot of money, but every dollar had to count.*
