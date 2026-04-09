# XNOR-Net LLM for OpenAI Parameter Golf Challenge

**Author:** Ciprian-Florin Ifrim -- April 2026

A full XNOR-Net language model that binarizes both weights and activations, trained for the [OpenAI Parameter Golf Challenge](https://openai.com/parameter-golf). The challenge requires training the best possible LLM that fits within a 16MB compressed artifact, evaluated on bits-per-byte (bpb) on the FineWeb validation set.

This work extends the Binary-Weight-Network (BWN) and ternary submissions with a true XNOR-Net implementation -- the first known application of full activation binarization to transformer language models at this scale.

**Best results:**

| Track | Run | Config | Roundtrip bpb | Sliding bpb | Size |
|-------|-----|--------|---------------|-------------|------|
| 10-minute (8xH100) | R40 | 1024d 10L embed=384 BF16 scales | 1.578 | -- | 15.96MB |
| 10-minute (3 seeds) | P3/P4/P5 | Same as R40 | 1.602 +/- 0.012 | 1.567 +/- 0.012 | 15.96MB |
| Notable (100k steps) | N2 | R40 + scale QAT | 1.575 | 1.539 | 15.91MB |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Technical Contributions](#key-technical-contributions)
3. [Development Timeline](#development-timeline)
4. [Activation Binarization Modes](#activation-binarization-modes)
5. [Activation Function Analysis](#activation-function-analysis)
6. [Triton XNOR Kernel](#triton-xnor-kernel)
7. [Compression Pipeline](#compression-pipeline)
8. [Optimizer Exploration](#optimizer-exploration)
9. [Learning Rate and Schedule Analysis](#learning-rate-and-schedule-analysis)
10. [Sequence Length Scheduling](#sequence-length-scheduling)
11. [Batch Size Sweep](#batch-size-sweep)
12. [Architecture Ablations](#architecture-ablations)
13. [Scale QAT and Roundtrip Gap](#scale-qat-and-roundtrip-gap)
14. [Attention Residuals](#attention-residuals)
15. [Complete Run Log](#complete-run-log)
16. [EGGROLL Exploration](#eggroll-exploration)
17. [Multi-Seed Variance](#multi-seed-variance)
18. [Final Configuration](#final-configuration)
19. [Reproduction](#reproduction)
20. [Key Insights](#key-insights)

---

## Architecture Overview

The model is a U-Net transformer with skip connections between encoder and decoder halves. All large weight matrices (QKV projections, attention output, MLP up/down) are binarized using the XNOR-Net approach from Rastegari et al. (2016). Small parameters (RMSNorm scales, skip weights, residual mixing, QK gains) remain in full precision.

### Model Configuration (Best: R40 / N2)

| Component | Value |
|-----------|-------|
| Model dimension | 1024 |
| Layers | 10 |
| Attention heads | 8 |
| KV heads | 4 (GQA) |
| MLP multiplier | 4x |
| Embedding dimension | 384 (R40) / 256 (R34) |
| BPE vocabulary | 1024 tokens |
| Total parameters | 117.6M |
| Binary parameters | 115.3M (98.0%) |
| FP parameters | 2.3M (2.0%) |
| Activation function | signsq (x * abs(x)) |
| Activation binarization | Mode 2 (XNOR except MLP down proj) |
| Group size | 256 |
| RoPE | YaRN (base=5000, max_len=2048) |
| Logit softcap | 10.0 (polynomial) |
| Tied embeddings | Yes |
| FP param storage | FP8 (e4m3fn) |
| Scale storage | BF16 (with FP8 STE for scale QAT) |

### U-Net Structure

The transformer is split into encoder (first N/2 layers) and decoder (remaining layers). Skip connections with learnable weights connect corresponding encoder-decoder pairs, initialized to ones. This provides error correction for the information loss inherent in binary quantization -- early features bypass the deepest (most lossy) layers.

### Weight Binarization (STE)

Each weight matrix W is binarized per group:
```
W_binary = sign(W) * alpha,  where alpha = mean(|W|) per group of 256 elements
```
During training, the Straight-Through Estimator (STE) passes gradients through sign() as if it were the identity function. The real-valued weights are maintained in float32 and updated normally; only the forward pass uses binary weights.

---

## Key Technical Contributions

### 1. Activation Binarization Mode 2

Full XNOR (binarizing all activations) plateaus at ~2.0 bpb regardless of training duration. The root cause: the MLP down projection receives all-positive inputs from activation functions, so sign() always returns +1 -- carrying zero information. Mode 2 skips activation binarization on the MLP down projection only, breaking through to 1.575 bpb while keeping all other projections binary.

### 2. signsq Activation Function

`signsq(x) = x * |x|` replaces relu^2 for Mode 2. Unlike relu^2 (which outputs only positive values), signsq produces negative outputs, so subsequent sign() operations in the attention path carry real information. This is critical for quality when activations are binarized.

### 3. Scale QAT (Quantization-Aware Training)

Binary weight group scales (alpha = mean(|W|) per group) are stored in FP8 at save time. Without scale QAT, the model trains with float32 scales but encounters FP8 quantization error at roundtrip, causing catastrophic degradation at long training runs (0.87 bpb gap at 200k steps). Scale QAT simulates FP8 quantization via STE during training, so the model learns to compensate for precision loss. Result: gap drops from 0.87 to 0.006 bpb.

### 4. Triton XNOR+POPCOUNT Kernel

A custom Triton kernel performs true 1-bit matrix multiplication using XOR and population count instructions. The kernel operates on packed int32 words (32 binary weights per word) with per-group scaling factors.

### 5. Cosine LR Schedule

Binary STE training with a flat learning rate followed by warmdown wastes ~70% of training in a divergent plateau. Cosine decay from the start keeps every step productive and enables 4x higher peak LR (0.008 vs 0.002).

### 6. Sequence Length Scheduling

Training starts at seq_len=128 and ramps through 256->512->1024 over four equal time phases. Short sequences give 8x more gradient updates per second during the early phase where the model just needs to learn token frequencies. All torch.compile graphs are cached during warmup by running one forward pass at each sequence length.

### 7. Low Momentum for Binary STE

Standard momentum (0.95) amplifies noisy STE gradients, causing destructive sign oscillations. Reducing momentum to 0.80 dampens this noise, giving a 0.027 bpb improvement.

---

## Development Timeline

### Phase 1: RTX 5090 Development (T-series runs)

Initial development on a single RTX 5090 32GB (Blackwell SM120) on Vast.ai. Established the architecture, debugged FlashAttention3 compatibility, developed and debugged the Triton XNOR kernel, and tested basic training dynamics. Key discovery: per-group alpha scaling is essential (per-row loses 0.62 bpb).

### Phase 2: 8xH100 Scaling (S-series runs)

Moved to 8xH100 SMX 80GB in a Docker container (driver 565.57, CUDA 12.7). Discovered that batch size dramatically affects binary training -- 65536 tokens outperforms 524288 by 0.07 bpb because binary networks benefit from frequent, small updates rather than rare, large ones.

### Phase 3: Record Attempts (R-series runs)

Systematic hyperparameter optimization covering 42 runs. Discovered cosine LR schedule (R25-R29), momentum reduction (R31-R35), gradient clipping (R30), sequence length scheduling (R33), and scale storage optimization (R37-R40). Cumulative improvement from R1 to R40: 2.074 -> 1.574 bpb (0.500 bpb gain).

### Phase 4: Notable Track (N-series runs)

Extended training at 100k-200k steps. N1 revealed the roundtrip gap problem from FP8 scale accumulation over long training. Scale QAT (N2) fixed this, achieving 1.575 roundtrip bpb.

### Phase 5: EGGROLL Exploration (E-series runs)

Attempted gradient-free evolution strategies using the EGGROLL algorithm (Sarkar et al. 2026). Tested full perturbation, layer-limited perturbation, and LoRA-based perturbation across 11 runs. Found that STE+Muon finds a basin too precise for zeroth-order methods to improve upon at 115M parameters.

### Phase 6: Attention Residuals (R41-R42, N3)

Implemented Attention Residuals from the Kimi Team (2026) paper as an alternative to U-Net skip connections. Each layer attends over all prior outputs via learned depth-wise attention. The 33% overhead from the depth-wise softmax reduced the number of training steps achievable in 10 minutes, resulting in worse final quality than the simpler U-Net skips.

---

## Activation Binarization Modes

`BINARIZE_ACTIVATIONS` controls which layers have their input activations binarized:

| Mode | Description | Best bpb | Notes |
|------|-------------|----------|-------|
| 0 | BWN -- weights only, float activations | 1.16* | Separate BWN submission |
| 1 | Full XNOR -- all activations binarized | 2.00 | Information bottleneck |
| 2 | XNOR except MLP down projection | **1.575** | Best quality |

*BWN result from separate Binary BitNet submission, not this XNOR codebase.

### Why Full XNOR Plateaus at 2.0 bpb

With relu^2 or signsq activation, MLP hidden states passed through the activation are either all-positive (relu^2) or mixed-sign (signsq). When the down projection's input goes through sign(), the quality depends entirely on whether these signs carry information:

With relu^2: every hidden element is positive, so sign() returns all +1. The binary dot product `sign(x) * sign(w)` degenerates to just `sum(sign(w))` -- the activation signs carry no information. This bottleneck limits quality to ~2.0 bpb regardless of model size or training duration.

With signsq in Mode 2: the down projection receives un-binarized signsq outputs (mixed signs with magnitude information), bypassing the bottleneck. All other projections (QKV, attention out, MLP up) still use full XNOR with binarized activations.

---

## Activation Function Analysis

| Activation | Formula | Pros | Cons |
|-----------|---------|------|------|
| relu^2 | relu(x)^2 | Excellent LZMA compression (structured signs) | Quality ceiling at ~2.0 bpb (all-positive) |
| signsq | x * abs(x) | Produces negative outputs, best quality | Poor compression (random signs) |
| swiglu | silu(gate) * up | Standard for LLMs | Higher param count |

### relu^2 Compression Phenomenon

relu^2 makes all MLP hidden activations positive. The down projection's weight signs evolve to be highly structured (correlated within groups) because the gradient signal only comes through the positive activation channel. LZMA compresses these structured signs extremely well -- a 196M param model (16L) compresses to 15.5MB.

However, this compression comes at the cost of quality. The model trades information capacity for compressibility. With signsq, the signs are high-entropy (incompressible) but carry genuine information, yielding much better bpb.

---

## Triton XNOR Kernel

### Architecture

The kernel performs binary matrix multiplication using XOR + population count:
```
dot(sign(x), sign(w)) = group_size - 2 * popcount(x_packed XOR w_packed)
```

Each group of 256 weights is packed into 8 int32 words. The kernel accumulates per-group dot products, scales by per-group alpha, and sums across groups.

### Per-Group Alpha Scaling

The kernel supports per-group weight scaling factors (alpha = mean(|w|) per group), matching the STE reference path exactly. Initial versions used per-row alpha which lost 0.62 bpb of quality.

### Bug Fix: int64 Promotion

Triton promotes int32 to int64 during 2D broadcast operations. When `xv[:, None] ^ wv[None, :]` creates a [BLOCK_M, BLOCK_N] tensor, the result is int64. `popc()` then dispatches to `__nv_popcll` (64-bit popcount) instead of `__nv_popc` (32-bit), counting 32 extra zero-bits for every positive int32 and 32 extra one-bits for every negative int32.

Fix: cast the XOR result back to int32 before calling popc:
```python
diff = (xv[:, None] ^ wv[None, :]).to(tl.int32)
group_acc += tl.extra.cuda.libdevice.popc(diff)
```

### bfloat16 Accumulation

The kernel accumulates in bfloat16 (not float32) to match the precision of the STE reference path and the roundtrip reconstruction. This reduced the quantization gap from 0.008 to 0.003 bpb.

### Performance

At the current model size (1024d, 65536 batch tokens, 8 GPUs), the Triton kernel shows no speed improvement over the BF16 STE path (~38ms/step for both). The matrices are too small for the kernel launch overhead to be amortized. The kernel's value is correctness verification and future larger models.

---

## Compression Pipeline

### Storage Formats

| Component | Format | Size (R40) |
|-----------|--------|------------|
| Binary weights | Packed bits (1 bit/param) | 14.87MB (pre-compression) |
| Group scales (g=256) | BF16 | 0.90MB |
| Embeddings, head, projections | FP8 (e4m3fn) | 0.86MB |
| Code | UTF-8 | 0.08MB |

### Compression Comparison

| Algorithm | Compressed Size (R34) | Compressed Size (R40) |
|-----------|-----------------------|-----------------------|
| LZMA preset 9 | 15.37MB | 15.95MB |
| **Brotli quality 11** | **15.30MB** | **15.89MB** |
| zstd level 22 | 17.08MB | 17.73MB |

Brotli consistently wins by ~50KB. zstd is worst for binary data -- it's optimized for structured text, not near-random bit patterns. The save process tries all three and picks the smallest, with a 1-byte header indicating the method for the decompressor.

### FP8 Scale Storage vs BF16

| Scale Storage | Extra Size | Roundtrip Gap | Notes |
|--------------|-----------|---------------|-------|
| FP8 | -0.45MB | 0.013 (R34) | Smaller artifact, minor precision loss |
| BF16 | baseline | 0.005 (R40) | Better roundtrip, essential for long training |

FP8 scale storage saves ~0.45MB but introduces quantization error on per-group scales. For 10-minute runs (15k steps), the gap is tolerable. For 100k+ steps, the error compounds and BF16 scales are essential (or scale QAT is needed).

### Sign-Sort Permutation

Post-training, MLP hidden dimensions are permuted so same-sign weight columns are adjacent. The corresponding rows of the paired projection are permuted identically, preserving model output. Intended to create long runs of identical bits for LZMA compression. Result: did not help for signsq (signs are high-entropy), only useful for relu^2 which has structured signs.

### Compression Regularizer

A differentiable penalty using `tanh(10*w)` that pushes weight signs within each group toward uniformity. Controlled by `SIGN_COMPRESS_REG`. Result: hurt quality, not worth it. The regularizer fights against the STE gradient signal, reducing model capacity without sufficient compression gain.

---

## Optimizer Exploration

### Muon (Momentum + NS Orthogonalization)

The primary optimizer for binary weight matrices. Uses Newton-Schulz orthogonalization on the gradient before applying the update. Muon was chosen because it produces well-conditioned updates that help binary STE training converge faster than Adam.

| NS Variant | Steps | Precision | val_bpb | ms/step |
|-----------|-------|-----------|---------|---------|
| **Original ns_orth** | **3** | **bf16** | **1.671** | **38.5** |
| Our Gram NS | 5 | bf16 | 1.713 | 38.8 |
| Library Gram NS | 5 | fp16 | 1.740 | 39.1 |

Original 3-step NS wins. Binary STE gradients are inherently noisy because sign() is a discontinuous function. More precise orthogonalization (Gram NS with 5 steps) doesn't help because the gradient itself is approximate. The library's float16 precision actively hurts because bfloat16's larger dynamic range matters more than mantissa precision for binary training.

### NS Step Count Ablation

| Steps | val_bpb | roundtrip | gap | ms/step |
|-------|---------|-----------|-----|---------|
| 2 | 1.684 | 1.733 | 0.049 | 37.8 |
| **3** | **1.671** | **1.672** | **0.001** | 38.5 |
| 5 | 1.713 | 1.719 | 0.006 | 38.9 |

3 steps is the sweet spot. 2 steps under-orthogonalizes, producing updates that are poorly conditioned and create a huge roundtrip gap (0.049). 5 steps over-orthogonalizes noisy STE gradients, wasting compute on precision that doesn't exist in the signal.

### Momentum

Momentum controls how much of the previous gradient update is carried forward. In standard float training, high momentum (0.95) smooths out mini-batch noise. But for binary STE training, each gradient is fundamentally approximate because sign() is not differentiable. High momentum amplifies these approximation errors, causing weights to oscillate across zero (flipping their sign back and forth unproductively).

| Momentum | val_bpb | Roundtrip | Gap | Notes |
|----------|---------|-----------|-----|-------|
| 0.95 | 1.636 | 1.639 | 0.003 | Standard, too noisy |
| 0.85 | 1.616 | 1.627 | 0.011 | Better |
| **0.80** | **1.589** | **1.602** | **0.013** | Best balance |
| 0.75 | 1.591 | 1.613 | 0.022 | Under-smoothed, worse gap |

At 0.80, the noise from STE gradient errors is dampened enough that the model trains stably, but there is still enough momentum to escape shallow local optima. At 0.75, gradients become too noisy (not enough smoothing), and the roundtrip gap doubles -- weights jitter more and quantize poorly.

### EMA (Exponential Moving Average)

| EMA Config | val_bpb | roundtrip | gap |
|-----------|---------|-----------|-----|
| **Off** | **1.589** | **1.602** | **0.013** |
| Start at 60% | 1.590 | 1.612 | 0.022 |
| Start at 0% | 1.674 | 1.909 | 0.235 |

EMA averages weights over recent training history. For float models this smooths out noise, but for binary models it's catastrophic. The averaged weights have less decisive signs -- they sit closer to zero where sign() is maximally sensitive to perturbation. During roundtrip (load from compressed artifact), these near-zero weights flip unpredictably, destroying quality. EMA is harmful for binary models.

---

## Learning Rate and Schedule Analysis

### LR Schedule: Linear Warmdown vs Cosine

The training loss curve for binary STE networks shows a distinctive "wandering" pattern. After an initial drop (steps 0-3000), loss increases and oscillates for thousands of steps (3000-9000) before dropping again during warmdown. This happens because the LR is too high for stable binary training -- each step flips thousands of weight signs, some productive, some destructive. The productive and destructive flips roughly cancel out, so the model wanders sideways.

With cosine decay, the LR starts decreasing immediately after warmup. There is no sustained high-LR plateau, so the wandering phase is compressed. More importantly, cosine enables a much higher peak LR (0.008 vs 0.002) because the rapid decay prevents the accumulated noise from causing divergence.

| Schedule | Peak LR | val_bpb |
|----------|---------|---------|
| Linear warmdown 0.3 | 0.002 | 1.654 |
| **Cosine** | **0.008** | **1.629** |

### Cosine LR Sweep

| LR | val_bpb | roundtrip |
|----|---------|-----------|
| 0.002 | 1.653 | 1.667 |
| 0.004 | 1.636 | 1.639 |
| 0.006 | 1.632 | 1.637 |
| **0.008** | **1.629** | **1.635** |
| 0.012 | 1.635 | 1.640 |

The peak is at 0.008. Below that, the model learns too slowly in the available training time. Above that, excessive sign flips early in training prevent the model from finding a good basin.

### Gradient Clipping

| Grad Clip | val_bpb |
|-----------|---------|
| 0.0 (off) | 1.629 |
| **1.0** | **1.626** |

Small improvement. Gradient clipping prevents any single batch from causing a catastrophic cascade of sign flips. In binary networks, a large gradient can flip the sign of many weights simultaneously, and the resulting binary network can be dramatically different from what the optimizer expected.

---

## Sequence Length Scheduling

Training starts at seq_len=128 and doubles at equal time intervals: 128->256->512->1024.

The reasoning: early in training, the model needs to learn basic token frequencies and simple bigram patterns. These require only short context. Processing short sequences is 8x faster than full 1024 (attention is quadratic), giving 8x more gradient updates per second. Once the model has learned local patterns, longer sequences allow it to learn long-range dependencies.

Implementation details: the schedule is based on either wall-clock time (if MAX_WALLCLOCK_SECONDS > 0) or step count (if using iterations). Each torch.compile graph is cached during warmup by running one forward-backward pass at each sequence length.

| Run | Config | val_bpb |
|-----|--------|---------|
| R31 | Cosine + momentum 0.85, no schedule | 1.616 |
| **R33** | **+ seq_len schedule** | **1.597** |
| R34 | + momentum 0.80 | 1.589 |

The 0.019 bpb gain from scheduling is entirely free -- the same total tokens are processed, just in a more efficient order. The model gets ~4x more gradient updates during the first quarter of training.

---

## Batch Size Sweep

Binary STE training strongly prefers smaller batches with more frequent updates. Each sign() decision is discrete -- once a weight's sign flips, the effect on the network is immediate and discontinuous. More frequent updates mean the model can react to the consequences of each sign flip sooner, correcting mistakes before they propagate.

| Batch Size | ms/step | Final bpb |
|-----------|---------|-----------|
| 524288 | 127.9 | 2.070 |
| 262144 | 69.1 | 2.072 |
| 131072 | 39.6 | 2.016 |
| **65536** | **38.6** | **1.999** |
| 32768 | 38.7 | 2.004 |

65536 is the sweet spot. Below that, per-batch gradient noise increases without speed benefit (DDP communication overhead dominates at small batch sizes). Above that, the model makes fewer sign-flip decisions per second, losing the benefit of frequent updates.

With 8 GPUs at 65536 total batch: 8192 tokens per GPU, well within VRAM. Step time is dominated by DDP synchronization, not compute.

---

## Architecture Ablations

### Group Size

The group size controls how many weights share a single scaling factor (alpha). Smaller groups give finer-grained scaling but noisier per-group statistics (fewer elements to average over). Larger groups give stable statistics but coarser approximation.

| Group Size | val_bpb | Size |
|-----------|---------|------|
| 128 | 1.671 | 15.51MB |
| **256** | **1.654** | **15.43MB** |
| 512 | 1.689 | 15.37MB |
| 1024 | 1.680 | 15.36MB |

256 is optimal. At 128, per-group mean(|w|) over 128 elements is noisy. At 512+, a single alpha must represent weights with different magnitudes, losing precision.

### Layers vs MLP Width

| Layers | MLP | Params | val_bpb |
|--------|-----|--------|---------|
| **10** | **4x** | **117M** | **1.654** |
| 13 | 3x | 125M | 1.745 |
| 10 | 5x | 138M | 1.671 |

Wider MLP is strictly better than deeper for binary networks. Each layer applies sign() to its output, which is a lossy operation that compounds across depth. More layers = more compounding information loss. A wider MLP gives more capacity per layer without the compounding. The 10L 4x config fits the 16MB budget optimally.

### Wider Model (768d) vs Standard (1024d)

| Config | val_bpb | Size |
|--------|---------|------|
| **1024d x 10L** | **1.589** | 15.37MB |
| 768d x 18L (embed=512) | 1.634 | 15.90MB |
| 768d x 18L (embed=256) | -- | 15.50MB |

Even with 80% more layers, the narrower model is worse. Binary networks lose information per layer, so depth hurts more than width helps.

### BPE Vocabulary

| BPE Size | val_bpb | FP Params | Size | Fits? |
|----------|---------|-----------|------|-------|
| **1024** | **1.654** | 0.85MB | 15.43MB | YES |
| 8192 | 1.673 | 2.70MB | 16.83MB | NO |

Smaller vocabulary saves 1.85MB of embedding FP params, allowing more binary parameters within the 16MB budget. The larger vocabulary doesn't compensate for the lost binary capacity.

### Embedding Dimension

| Embed Dim | Storage | val_bpb | Roundtrip | Size | Notes |
|-----------|---------|---------|-----------|------|-------|
| 256 | FP8 | 1.589 | 1.602 | 15.37MB | R34 best 10-min |
| **384** | **FP8+BF16 scales** | **1.574** | **1.578** | **15.96MB** | **R40 best overall** |
| 512 | FP8+FP8 scales | 1.569 | 2.435 | 15.66MB | Roundtrip catastrophic |
| 512 | FP8+BF16 scales | -- | -- | 16.26MB | Over budget |

384 embed_dim with BF16 scales is the sweet spot -- richer embedding space within budget, and the BF16 scales avoid roundtrip degradation. 512 embed with FP8 scales destroys roundtrip at long training due to accumulated scale quantization error.

### Logit Softcap

| Softcap | val_bpb |
|---------|---------|
| **10** | **1.671** |
| 15 | 1.684 |

10 is better. Lower softcap constrains logits more, regularizing the model. Uses polynomial approximation (`x * (1 - x^2/3 + x^4/15)`) instead of tanh because tanh doesn't fuse with torch.compile.

### Smear Module

| Smear | val_bpb | Notes |
|-------|---------|-------|
| **Off** | **1.589** | Saves ~1ms/step |
| On | 1.603 | Doesn't help with seq_len scheduling |

Smear didn't help with sequence length scheduling enabled. The scheduling already provides the "easy then hard" curriculum that smear approximates.

### Size Check Runs (T28-T34)

Architecture variants tested for 16MB budget fit:

| Run | Config | Params | Size | Fits? |
|-----|--------|--------|------|-------|
| T28 | 10L 1024d embed=512 | 118.0M | 15.87MB | YES |
| T29 | 11L 1024d embed=256 | 128.8M | 16.80MB | NO |
| T30 | 14L 768d embed=256 | 92.3M | 12.10MB | YES |
| T31 | 20L 768d embed=256 | 131.3M | 17.09MB | NO |
| T32 | 18L 768d embed=256 | 118.3M | 15.43MB | YES |
| T33 | 19L 768d embed=256 | 124.8M | 16.26MB | NO |
| T34 | 18L 768d embed=512 | 118.9M | 15.89MB | YES |

---

## Scale QAT and Roundtrip Gap

### The Problem

During training, per-group weight scales (alpha = mean(|w|)) are computed in float32. At save time, these scales are quantized to FP8 for storage. Each step introduces a tiny error that the model never sees during training. Over 200k steps, the model becomes precisely tuned to float32 scale values that FP8 cannot represent, causing catastrophic roundtrip degradation.

| Run | Steps | FP Storage | Scale Storage | Scale QAT | val_bpb | Roundtrip | Gap |
|-----|-------|-----------|---------------|-----------|---------|-----------|-----|
| R34 | 15k | FP8 | FP8 | No | 1.589 | 1.602 | 0.013 |
| N1 | 200k | FP8 | FP8 | No | 1.569 | 2.435 | **0.866** |
| R40 | 15k | FP8 | BF16 | No | 1.574 | 1.578 | 0.005 |
| **N2** | **100k** | **FP8** | **BF16** | **Yes** | **1.569** | **1.575** | **0.006** |

N1's 100k checkpoint had roundtrip 1.986, 30k checkpoint had 2.121 -- the error compounds monotonically with training steps.

### The Fix

Scale QAT simulates FP8 quantization on scales during the forward pass via STE:
```python
alpha_q = alpha.to(torch.float8_e4m3fn).to(alpha.dtype)
alpha = alpha + (alpha_q - alpha).detach()  # STE
```

The model sees the quantized scale values during training and learns to compensate. Combined with BF16 scale storage (which has negligible quantization error), the roundtrip gap stays below 0.006 bpb even at 100k steps.

---

## Attention Residuals

### Background

Attention Residuals (Kimi Team, 2026) replace standard residual connections with learned depth-wise attention. Instead of `h_l = h_{l-1} + f(h_{l-1})`, each layer attends over ALL prior outputs: `h_l = softmax_weighted_sum(all previous outputs)`. This allows later layers to selectively retrieve information from any earlier layer, bypassing lossy intermediate sign() operations.

### Implementation

Two modes were implemented:
- **Mode 2 (pass-level):** One stored tensor per block. 10 query vectors x 1024 dim = 10K params.
- **Mode 1 (sub-layer):** One stored tensor per sub-layer (attention + MLP). 20 queries x 1024 dim = 20K params.

Queries are zero-initialized so the model starts with uniform weights (equivalent to standard residual). Keys are RMSNorm'd stored outputs. No projection matrices needed.

### Results

| Run | Mode | Steps | ms/step | val_bpb | roundtrip | sliding |
|-----|------|-------|---------|---------|-----------|---------|
| R40 | U-Net (0) | 15560 | 38.6 | 1.574 | 1.578 | -- |
| R41 | AttnRes (2) | 11560 | 51.7 | 1.594 | 1.598 | -- |
| R42 | AttnRes (1) | -- | -- | crash | -- | -- |
| N2 | U-Net (0) | 100k | 39.0 | 1.569 | 1.575 | 1.539 |
| N3 | AttnRes (2) | 100k | 51.8 | 1.583 | 1.596 | 1.563 |

### Analysis

AttnRes adds 33% overhead (51.7ms vs 38.6ms) from the depth-wise softmax computation over stored tensors. In the 10-minute track, this overhead means ~4000 fewer training steps, which more than negates any architectural benefit. Even at 100k steps (N3 vs N2), U-Net wins by 0.021 bpb in roundtrip.

The overhead comes from: storing 10+ tensors, computing 10 einsum operations for logits, softmax, and weighted sum each forward pass. Torch.compile partially fuses these but the softmax reduction dimension is too small (10 elements) for efficient GPU execution.

Mode 1 (sub-layer) crashed with an Inductor OOM error -- the backward graph with 20 stored tensors exceeds Triton's register file limits for the fused RMSNorm backward kernel.

Conclusion: for binary networks, simple weighted skip connections (U-Net) provide sufficient error correction at much lower overhead than learned depth-wise attention.

---

## Complete Run Log

### T-series: RTX 5090 Testing

| Run | Config | Steps | val_bpb | Size | Notes |
|-----|--------|-------|---------|------|-------|
| T7 | relu2, mode 2, 15L 1024d | 200 | 1.807 | 26.08MB | Mode 2 first test |
| T9 | relu2, mode 1, 16L 1024d | 50 | 2.058 | 18.46MB | Size check |
| T10 | relu2, mode 1, 15L 1024d | 50 | 2.060 | 17.46MB | Size check |
| T11 | relu2, mode 1, 12L 1024d | 50 | 2.056 | 14.46MB | Fits |
| T12 | relu2, mode 1, 16L 1024d | 22500 | 2.569+ | -- | Diverged |
| T13 | relu2, mode 1, 16L LR=0.01 | 2000 | 1.879 | -- | Still diverged |
| T14 | signsq, mode 1, 16L | 2000 | 1.985 | 27.44MB | No compression |
| T15 | signsq, mode 1, 16L (sign-sort) | 50 | 2.079 | 27.35MB | Sign-sort no help |
| T16 | signsq, mode 1, 16L (sign-sort) | 5000 | 1.941 | 27.51MB | Sign-sort no help |
| T17 | signsq, mode 1, 11L 1024d | 500 | 1.842 | 19.65MB | Over |
| T18 | signsq, mode 1, 11L 1024 BPE | 500 | 2.001 | 17.71MB | Over |
| T19 | signsq, mode 1, 10L 1024 BPE | 500 | 2.042 | 16.15MB | Over |
| T20 | signsq, mode 1, 10L g=256 | 500 | 2.019 | 15.76MB | **Fits** |
| T21 | signsq, mode 1, 10L 262k batch | 15000 | diverged | -- | LR too high |
| T22 | signsq, mode 1, EMA, LR=0.005 | 1500 | 1.995 | 15.75MB | EMA helped |
| T23 | signsq, mode 1, no EMA, LR=0.005 | 1500 | 2.005 | 15.75MB | LR sufficient |
| T25 | signsq, mode 1, 10L LR=0.005 | 2500 | 1.984 | 15.76MB | Best mode 1 |
| T26 | Triton kernel (per-row alpha, buggy) | 1000 | 2.451 | 14.39MB | Kernel bug |
| T27 | Triton kernel (per-row, no compile) | 1000 | 2.481 | 14.39MB | Bug confirmed |
| T28 | Triton kernel (per-group, fixed) | 1000 | 2.028 | 15.68MB | **Kernel works** |

### S-series: 8xH100 Scaling

| Run | Config | Steps | val_bpb | Size | Notes |
|-----|--------|-------|---------|------|-------|
| S1 | BF16 scales, 524k batch | 5000 | 2.120 | 16.35MB | Over |
| S2 | FP8 scales, 524k batch | 5000 | 2.070 | 15.75MB | Fits |
| S3 | FP8 scales + compress reg 0.01 | 5000 | 2.089 | 15.75MB | Reg hurts |
| S4 | FP8 scales, 262k batch | 5000 | 2.072 | 15.76MB | Same quality |
| S5 | FP8 scales, 131k batch | 5000 | 2.016 | 15.76MB | Better |
| **S6** | **FP8 scales, 65k batch** | **5000** | **1.999** | **15.76MB** | **Best batch** |
| S7 | FP8 scales, 32k batch | 5000 | 2.004 | 15.78MB | Diminishing returns |
| S8 | LR=0.002, 65k batch | 5000 | 2.044 | 15.75MB | Too conservative |
| S9 | LR=0.003, 65k batch | 5000 | 1.995 | 15.76MB | Good |
| S10 | LR=0.004, 65k batch | 5000 | 1.994 | 15.76MB | Best LR |
| S11 | Gram NS library, 65k batch | 1000 | 2.004 | 15.51MB | Slightly slower |

### R-series: Record Attempts

| Run | Key Changes | val_bpb | RT bpb | Size | Fits? |
|-----|-------------|---------|--------|------|-------|
| R1 | LR=0.003, 600s | 2.074 | 2.075 | 15.68MB | YES |
| R2 | LR=0.001 | 2.121 | 2.124 | 15.67MB | YES |
| R3 | **Mode 2** (MLP down BWN) | 1.699 | -- | 15.67MB | YES |
| R4 | Mode 2 + EMA@60% | 1.668 | 1.787 | 15.64MB | YES |
| R5 | Mode 2 + EMA@0% | 1.674 | 1.909 | 15.59MB | YES |
| R6 | Triton per-row (buggy) | 2.323 | 2.363 | 14.82MB | YES |
| R7 | Triton fixed, LR=0.002 | 1.659 | 1.667 | 15.66MB | YES |
| R8 | Triton, LR=0.001 | 1.700 | 1.707 | 15.67MB | YES |
| R9 | Triton bf16, BF16 scales | 1.676 | 1.679 | 15.67MB | YES |
| R10 | Triton bf16, FP8 scales | 1.654 | 1.663 | 15.43MB | YES |
| R11 | BF16 everything | 1.665 | 1.666 | 16.35MB | NO |
| R12 | 13L MLP 3x | 1.745 | 1.752 | 16.37MB | NO |
| R13 | 10L MLP 5x | 1.671 | 1.685 | 18.09MB | NO |
| R14 | 11L MLP 4x | 1.708 | 1.707 | 16.89MB | NO |
| R15 | g=512 | 1.689 | 1.694 | 15.37MB | YES |
| R16 | g=1024 | 1.680 | 1.685 | 15.36MB | YES |
| R17 | g=128, softcap=15 | 1.684 | 1.691 | 15.51MB | YES |
| R18 | g=128, softcap=10 | 1.671 | 1.675 | 15.51MB | YES |
| R19 | 8192 BPE | 1.673 | 1.684 | 16.83MB | NO |
| R20 | Gram NS library | 1.740 | 1.743 | 15.47MB | YES |
| R21 | Our Gram NS (bf16) | 1.713 | 1.716 | 15.48MB | YES |
| R22 | Original NS, 3 steps | 1.671 | 1.672 | 15.42MB | YES |
| R23 | NS 2 steps | 1.684 | 1.733 | 15.37MB | YES |
| R24 | NS 5 steps | 1.713 | 1.719 | 15.47MB | YES |
| R25 | Cosine LR=0.004 | 1.636 | 1.639 | 15.43MB | YES |
| R26 | Cosine LR=0.002 | 1.653 | 1.667 | 15.41MB | YES |
| R27 | Cosine LR=0.006 | 1.632 | 1.637 | 15.45MB | YES |
| R28 | Cosine LR=0.008 | 1.629 | 1.635 | 15.46MB | YES |
| R29 | Cosine LR=0.012 | 1.635 | 1.640 | 15.46MB | YES |
| R30 | + Grad clip 1.0 | 1.626 | 1.631 | 15.46MB | YES |
| R31 | + Momentum 0.85 | 1.616 | 1.627 | 15.38MB | YES |
| R32 | Momentum 0.75 | 1.617 | 1.645 | 15.36MB | YES |
| R33 | + Seq len schedule | 1.597 | 1.612 | 15.39MB | YES |
| **R34** | **+ Momentum 0.80** | **1.589** | **1.602** | **15.37MB** | **YES** |
| R35 | Momentum 0.75 + schedule | 1.591 | 1.613 | 15.35MB | YES |
| R36 | 768d 18L embed=512 | 1.634 | 1.651 | 15.90MB | YES |
| R37 | 1024d 10L embed=512 | 1.602 | 1.623 | 15.82MB | YES |
| R38 | 1024d 10L embed=512 + smear | 1.603 | -- | 15.83MB | YES |
| R39 | R34 + EMA@60% | 1.590 | 1.612 | 15.36MB | YES |
| **R40** | **embed=384, BF16 scales** | **1.574** | **1.578** | **15.96MB** | **YES** |
| R41 | AttnRes mode 2 | 1.594 | 1.598 | 15.91MB | YES |
| R42 | AttnRes mode 1 | crash | -- | -- | -- |

### P-series: Push/Submit (10-min track, 3 seeds)

| Run | Seed | val_bpb | roundtrip | sliding (s=48) | gap |
|-----|------|---------|-----------|----------------|-----|
| P3 | 42 | 1.582 | 1.591 | 1.556 | 0.009 |
| P4 | 7 | 1.605 | 1.615 | 1.580 | 0.010 |
| P5 | 1337 | 1.598 | 1.600 | 1.565 | 0.002 |
| **Mean** | -- | **1.595** | **1.602** | **1.567** | **0.007** |

### N-series: Notable Track

| Run | Config | Steps | val_bpb | roundtrip | sliding | Gap |
|-----|--------|-------|---------|-----------|---------|-----|
| N1 | embed=512, FP8 scales, no QAT | 200k | 1.569 | 2.435 | -- | 0.866 |
| **N2** | **embed=384, BF16 scales, scale QAT** | **100k** | **1.569** | **1.575** | **1.539** | **0.006** |
| N3 | AttnRes mode 2 | 100k | 1.583 | 1.596 | 1.563 | 0.013 |

---

## EGGROLL Exploration

### Background

EGGROLL (Sarkar et al. 2026) uses rank-r low-rank perturbations for efficient evolution strategies. Instead of sampling full-rank noise matrices, it samples A in R^(m x r) and B in R^(n x r) and forms E = (1/sqrt(r)) * AB^T. This enables gradient-free optimization that bypasses the STE entirely, directly optimizing the loss function over the binary weight space.

The motivation for trying EGGROLL on our binary network: the STE is a fundamentally approximate gradient. EGGROLL evaluates the true loss function (with actual sign() and quantization), so it could potentially find better solutions than STE-based gradient descent.

### Implementation

Three approaches were implemented:

1. **Full perturbation:** Perturb all 115M binary weight parameters directly. Each perturbation adds sigma * (1/sqrt(r)) * AB^T to the float weights before binarization.

2. **Layer-limited perturbation:** Perturb only the last N layers (controlled by EGGROLL_LAYERS). Reduces dimensionality from 115M to 11.5M-34M.

3. **LoRA perturbation:** Create LoRA adapter pairs (A, B) for each binary weight matrix. Perturb only the LoRA parameters (~614K params at rank 4). Before each forward pass, merge LoRA into base weights, evaluate, then unmerge. Final model merges LoRA permanently.

### Results: Full Perturbation

| Run | Start | Pop | Sigma | LR | Rank | Fitness | Result |
|-----|-------|-----|-------|-----|------|---------|--------|
| E1 | Random | 256 | 0.01 | 0.001 | 1 | -6.936 | No learning |
| E2 | Random | 256 | 0.5 | 0.1 | 1 | -6.937 | No learning |
| E3 | Pretrained | 256 | 0.01 | 0.001 | 1 | -10.48 | Diverged |
| E4 | Pretrained | 256 | 0.0001 | 0.00001 | 1 | -10.46 | Diverged slowly |
| E5 | Pretrained | 4096 | 0.0001 | 0.0001 | 1 | -2.98 | Stable, flat |
| E6 | Pretrained | 4096 | 0.0001 | 0.00001 | 8 | -3.24 | Slow divergence |
| E7 | Pretrained | 4096 | 0.0001 | 0.000001 | 8 | -3.01 | Stable, flat |

From scratch (E1-E2), ES cannot navigate the 115M-dimensional landscape at any sigma or population size. Even 4096 population provides zero useful gradient signal.

From pretrained weights (E3-E7), the perturbation scale is critical. Too large (E3, sigma=0.01): every perturbation destroys the trained model, so the "best" direction is just "least bad." Too small (E4, sigma=0.0001): fitness differences become noise-dominated. The sweet spot (E5, sigma=0.0001, pop=4096) is stable but shows zero improvement -- every perturbation direction is uphill from the STE-found basin.

### Results: Layer-Limited Perturbation

| Run | Layers Perturbed | Params | val_bpb after 10 steps | Degradation |
|-----|-----------------|--------|----------------------|-------------|
| E7 | All 40 tensors | 115M | 1.633 | +0.064 |
| E8 | Last 3 blocks (12 tensors) | 34M | 1.629 | +0.060 |
| E9 | Last 1 block (4 tensors) | 11.5M | 1.609 | +0.040 |

Fewer parameters to perturb means less damage per step, but still no improvement. The model is at a local optimum in every direction, even when only searching a 11.5M-dimensional subspace.

### Results: LoRA Perturbation

| Run | LoRA Rank | Pop | Sigma | LR | LoRA Params | val_bpb after 10 steps |
|-----|-----------|-----|-------|-----|-------------|----------------------|
| E10 | 4 | 4096 | 0.01 | 0.001 | 614K | 1.573 (+0.004) |
| E11 | 4 | 16384 | 0.01 | 0.001 | 614K | -- (3.7 min/step) |

LoRA brings the perturbation dimensionality down to 614K -- manageable for ES. E10 with pop=4096 was nearly stable (only +0.004 degradation vs +0.040 for direct perturbation of the same parameters). But still no improvement, and E11 at pop=16384 was too slow at 229s/step to be practical.

### Why EGGROLL Cannot Improve on STE+Muon

The fundamental issue is signal-to-noise ratio. With rank-1 perturbations in d-dimensional space, the cosine similarity between any random perturbation and the true gradient is approximately 1/sqrt(d). For d=115M, this gives ~0.00009. Population size N improves this by sqrt(N), so pop=4096 gives ~0.006 -- still 99.4% noise.

The EGGROLL paper's successful pretraining used a 256-dim model with up to 1M population. For 115M params, the required population would be orders of magnitude larger than is practical.

LoRA reduces d to 614K, giving ~0.04 per perturbation and ~2.5 with pop=4096. Better, but the LoRA subspace may not contain the improvement direction. The STE+Muon optimizer has access to 115M-dimensional gradient information per step, which is fundamentally more informative than 4096 scalar fitness samples.

---

## Multi-Seed Variance

Three seeds (42, 7, 1337) were run with the P1 config to estimate variance:

| Metric | Seed 42 | Seed 7 | Seed 1337 | Mean | Std |
|--------|---------|--------|-----------|------|-----|
| val_bpb | 1.582 | 1.605 | 1.598 | 1.595 | 0.012 |
| roundtrip | 1.591 | 1.615 | 1.600 | 1.602 | 0.012 |
| sliding (s=48) | 1.556 | 1.580 | 1.565 | 1.567 | 0.012 |
| gap | 0.009 | 0.010 | 0.002 | 0.007 | 0.004 |

Standard deviation of ~0.012 bpb across seeds. This is typical for binary networks where early sign choices cascade -- a different random initialization puts the model into a different basin, and small differences compound through the sign() operations.

---

## Final Configuration

### P1: 10-Minute Track Submission (R40 config)

```bash
# Architecture
NUM_LAYERS=10 MODEL_DIM=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
EMBED_DIM=384 VOCAB_SIZE=1024 ACTIVATION=signsq ATTN_RES=0
# XNOR
XNOR_GROUP_SIZE=256 BINARIZE_ACTIVATIONS=2 USE_TRITON_KERNEL=1
# Storage
FP_STORAGE=FP8 SCALE_STORAGE=BF16
# Optimizer
MATRIX_OPTIMIZER=muon MATRIX_LR=0.008 MUON_MOMENTUM=0.80
MUON_BACKEND_STEPS=3 MUON_WD=0.04
LR_SCHEDULE=cosine GRAD_CLIP_NORM=1.0
# Schedule
SEQ_LEN_SCHEDULE=1 TRAIN_BATCH_TOKENS=65536
MAX_WALLCLOCK_SECONDS=600
```

**Best single run: R40 -- 1.574 val, 1.578 roundtrip, 15.96MB**
**Three-seed mean: 1.602 +/- 0.012 roundtrip, 1.567 +/- 0.012 sliding**

### N2: Notable Track (100k steps)

Same as P1 but:
```bash
MAX_WALLCLOCK_SECONDS=0 ITERATIONS=100000 CHECKPOINT_EVERY=25000
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=16 TEMP_SCALING=1
```

**Result: 1.569 val, 1.575 roundtrip, 1.539 sliding, 15.91MB**

---

## Reproduction

### Requirements

- 8x NVIDIA H100 80GB SMX (or equivalent)
- PyTorch 2.10.0+cu128
- FlashAttention 3
- Triton 3.6.0
- Python 3.13

### Setup

```bash
bash setup.sh
conda activate golf
pip install brotli zstandard --break-system-packages
```

### Training

```bash
# 10-minute track
bash run_cuda_xnor_v2.sh

# Notable track (100k steps, ~65 minutes)
bash run_cuda_xnor_notable.sh

# EGGROLL exploration
bash run_cuda_eggroll.sh
```

### Data

FineWeb 10B dataset with 1024 BPE tokenizer. 80 training shards, 1 validation shard (~40.5M tokens).

---

## Key Insights

1. **Binary networks need frequent, small updates.** Batch size 65536 >> 524288 for quality. Each sign() is a discrete decision -- more decisions per second means faster convergence.

2. **Full XNOR activation binarization has a quality ceiling around 2.0 bpb** due to the MLP information bottleneck. Mode 2 (skipping MLP down proj) breaks through to 1.575.

3. **Momentum should be lower than standard (0.80 vs 0.95)** because STE gradient noise is amplified by momentum, causing destructive sign oscillations.

4. **Cosine LR schedule is essential** for binary STE training. Flat LR with warmdown wastes 70% of training time in a divergent plateau.

5. **Sequence length scheduling provides free improvement** -- short sequences at the start give 8x more gradient updates during the phase where the model needs to learn token frequencies.

6. **Wider is better than deeper** for binary networks. Each sign() compounds information loss across layers, but wider MLP gives more capacity per layer.

7. **EMA is harmful** for binary models -- the averaged weights have less decisive signs that don't survive quantization.

8. **Scale QAT is essential for long training runs.** Without it, FP8 scale quantization error accumulates over steps and causes catastrophic roundtrip degradation (0.87 bpb gap at 200k steps).

9. **Attention Residuals add overhead without benefit for binary networks.** The 33% slower steps reduce training progress more than depth-wise attention helps. Simple U-Net skips are sufficient.

10. **EGGROLL cannot improve on STE+Muon at 115M parameters.** The signal-to-noise ratio of zeroth-order methods is too low for practical population sizes. Even LoRA-based EGGROLL (614K params, pop=4096) shows no improvement from the STE-found basin.

---

## References

- Rastegari, M. et al. "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks." ECCV 2016.
- Sarkar, B. et al. "Evolution Strategies at the Hyperscale." arXiv:2511.16652, 2026.
- Zhang, J. et al. "Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon." dao-ailab, 2026.
- Kimi Team. "Attention Residuals." 2026.

---

## License

This project is part of the OpenAI Parameter Golf Challenge submission.
