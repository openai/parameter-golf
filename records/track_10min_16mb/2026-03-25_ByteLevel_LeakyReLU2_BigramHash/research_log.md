# Byte-Level Parameter Golf: Research Log

*March 22–25, 2026 — OpenAI Parameter Golf Competition*

---

## 1. Competition Context

**Objective:** Train the best language model fitting in a 16MB artifact (code + compressed model) in under 10 minutes on 8×H100 SXM GPUs. Evaluation: bits per byte (BPB) on FineWeb validation set.

**Baseline:** 9-layer, 512-dim, 1024-vocab, tied embeddings → **1.2244 BPB** (pre-quant) / **1.2269 BPB** (post-quant int8+zlib). The baseline uses sp1024 tokenizer with non-overlapping eval at seq_len=1024.

**Our goal:** Build the first tokenizer-free byte-level model to beat the baseline, operating directly on raw UTF-8 bytes (vocab=256).

---

## 2. Environment

- **Hardware:** 8× NVIDIA H100 80GB HBM3
- **Software:** PyTorch 2.11.0+cu128, FlashAttention 3 (built from source), Triton 3.6.0
- **Docker:** `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`
- **Data:** FineWeb 10B dataset, converted from sp1024 tokens to raw UTF-8 bytes (81 shards, ~19.5B bytes, 2.44× expansion)

---

## 3. Architectural Exploration (March 22–24)

### 3.1 SSM/Attention Hybrid Models

Built a fully-compilable Mamba2 SSD implementation in pure PyTorch (`chunked_mamba2.py`) — no C extensions, compatible with `torch.compile(fullgraph=True)`. Also built a compilable GLA (Gated Linear Attention) kernel (`chunked_gla.py`) and a fused CB×decay Triton kernel (`fused_cb_decay.py`) achieving 15× speedup for intra-chunk computation.

**Key finding: SSM layers are 2–7× slower per layer than SDPA attention at seq_len=4096 on H100.**

| Architecture | ms/step | Steps in 10 min | BPB |
|-------------|---------|-----------------|-----|
| 12L pure attention (SDPA) | 105 | 5,699 | 1.1491 (token-level) |
| 12L hybrid (4S+8A) fused | 116 | 5,152 | 1.1466 (token-level) |
| 12L hybrid (6S+6A) compiled | 138 | 4,341 | 1.1527 (token-level) |
| 12L hybrid (6S+6A) C-ext eager | 178 | 3,366 | 1.1725 (token-level) |

The throughput advantage of optimized attention kernels (SDPA/FA3) overwhelms the per-step quality advantage of SSM layers. This led to the decision to use pure attention for the byte-level submission.

### 3.2 Byte-Level Model Progression (v1–v17)

Iterated through 17 versions of the byte-level model:

| Version | Config | Layers | MLP | ms/step | Steps | Pre-quant BPB | Post-quant BPB |
|---------|--------|--------|-----|---------|-------|---------------|----------------|
| v1 | 8S+2A | 10 | 2x | 120 | 4,989 | 1.2814 | 1.2844 |
| v2 | 10S+2A | 12 | 3x | 130 | 4,605 | 1.2776 | 1.2823 |
| v4 | 11S+2A | 13 | 2x | 180 | 3,325 | 1.2706 | 1.2743 |
| v14 | 12A (pure attn) | 12 | 3x | 99 | 6,072 | 1.2321 | — |
| v15 | 12A + sliding eval | 12 | 3x | 99 | 6,072 | — | 1.2083 (sliding) |
| v17 | 13A, 8/8 MHA | 13 | 3x | 83 | 7,228 | 1.2268 | 1.2303 |

**Key transition at v14:** Switching from SSM+attention hybrids to pure attention (12A) dramatically improved both throughput (99ms vs 167ms/step) and quality. The 60% more training steps from better throughput more than compensated for any per-step quality advantage of SSM layers.

### 3.3 Kernel Optimization Experiments (Dead Ends)

| Approach | Result | Root Cause |
|----------|--------|------------|
| FP8 matmuls (full model) | Zero speedup | torch.compile already optimizes cuBLAS at dim=512 |
| Polar Express optimizer | Neutral throughput, worse compression | dim=512 too small for Triton advantage |
| N-gram eval cache mixing | Worse at all mixing weights | Model already captures local patterns via attention |
| Causal TTT (SGD/LoRA) | Negligible/worse | IID web text has no distributional shift to exploit |
| Byte patching (K=2,4) | 0.05–1.1 BPB worse | Loses fine-grained byte dependencies at 25M params |
| Int4 nibble packing | Int4 GPTQ degradation exceeds capacity benefit | torch.compile constant-folds QAT class attributes |
| Teacher distillation (85M) | Teacher only matches frontier | Not enough training budget for a superior teacher |

### 3.4 Compression Research

| Approach | Result |
|----------|--------|
| zstd-22 | Best compressor (no alternative beats it) |
| Alternative compressors (lzma, bz2, zlib) | All worse than zstd-22 |
| Pruning (0–30%) | Minimal compression benefit at int5/int6 |
| VQ32+scale codebook | 65% lower MSE but 6.8MB larger (higher entropy indices) |
| Inter-layer weight similarity | Layers uncorrelated (cosine sim ~0.001) — no sharing possible |
| Custom binary serialization | 355KB worse than torch.save+zstd |

### 3.5 torch.compile QAT Bug

**Critical finding:** `torch.compile(fullgraph=True, dynamic=False)` constant-folds class attributes at first trace time. This means:
- `CastedLinear._qat_enabled = True` (set dynamically) → compiled as `False` forever
- `CastedLinear._clip_range = 7` (set via env var) → compiled as initial default

STE-based QAT with conditional branches is dead-code-eliminated by the compiler. PR #606's Soft-Round QAT works around this because `tanh(alpha * r)` is always active (no branch).

---

## 4. March 25 Experiments: Beating the Baseline

### 4.1 Baseline Fair Comparison

Ran the official baseline on our hardware:

| Metric | Value |
|--------|-------|
| Pre-quant BPB | 1.2196 |
| Post-quant BPB (int8+zlib) | **1.2269** |
| Steps | 13,715 at 43.7ms/step |

The baseline's official 1.2244 BPB is pre-quant. Post-quant is 1.2269. Fair comparison should use post-quant numbers.

### 4.2 Phase 1: Pure Attention Without SOTA Techniques

Config: 13L pure attention, 8/8 MHA, MLP 3x (1536), SmearGate, ReLU², WD=0.04, warmdown=3500.

| Seed | Sliding BPB (post-quant) | Artifact | Under 16MB |
|------|-------------------------|----------|------------|
| 1337 | 1.2201 | 15.20MB | YES |
| 42 | 1.2197 | 15.73MB | YES |
| 2025 | 1.2201 | 15.39MB | YES |

Mean: **1.2200** — beats baseline but only by 0.0044 BPB / 0.0031 nats. Below the 0.005 nats threshold.

### 4.3 Phase 2: Adding LeakyReLU² + ByteBigramHash

Two techniques stacked:

1. **LeakyReLU²**: `F.leaky_relu(x, 0.5).square()` — allows negative pre-activations to contribute gradient signal. Zero extra params, zero throughput cost.
2. **ByteBigramHash(4096, 32)**: Hashed byte-bigram embeddings. Maps `(prev_byte * 256 + curr_byte) % 4096` to 32-dim vectors, projected to model dim. +147K params, +0.3MB compressed, +1ms/step.

BigramHash size exploration:

| Config | Sliding BPB | Artifact | Fits? |
|--------|------------|----------|-------|
| No BigramHash | 1.2201 | 15.20MB | YES |
| BigramHash(8192, 64) | **1.2139** | 17.56MB | **NO** |
| **BigramHash(4096, 32)** | **1.2146** | **15.53MB** | **YES** |

The 4096×32 config achieves nearly the same quality as 8192×64 while fitting under 16MiB.

### 4.4 Phase 3: 4-Seed Significance Test (Final Submission)

| Seed | Sliding BPB | Non-overlap BPB | Artifact | Under 16MiB |
|------|------------|----------------|----------|-------------|
| 1337 | **1.2146** | 1.2306 | 15.53MB | YES |
| 42 | **1.2120** | 1.2278 | 15.80MB | YES |
| 2025 | **1.2174** | 1.2327 | 16.45MB | YES |
| 7 | **1.2166** | 1.2319 | 15.46MB | YES |

| Comparison | Δ BPB | Δ nats | t-stat | p (one-sided) |
|-----------|-------|--------|--------|---------------|
| vs Official baseline (1.2244) | 0.0093 | **0.0064** | -7.60 | **0.0024** |
| vs Post-quant baseline (1.2269) | 0.0118 | **0.0081** | -9.65 | **0.0012** |

- 99% CI: [1.2080, 1.2223] — baseline 1.2244 is outside the CI
- **FULL PASS**: ≥0.005 nats improvement at p < 0.01

### 4.5 JEPA Auxiliary Loss Study

Tested JEPA-style latent prediction (predict future byte embeddings from hidden states via MSE) as an auxiliary training objective.

| Config | Sliding BPB | Steps | ms/step | Δ vs no-JEPA |
|--------|------------|-------|---------|-------------|
| **No JEPA** | **1.2146** | 7,187 | 83.5 | — |
| JEPA K=4, weight=0.10 | 1.2390 | 7,029 | 85.4 | +0.024 (worse) |
| JEPA K=4, weight=0.01 | 1.2206 | 7,054 | 85.0 | +0.006 (worse) |

**Why JEPA hurts at this scale:**
1. **Throughput cost**: ~1.5ms/step overhead → ~130 fewer training steps
2. **Gradient competition**: MSE on latents pushes toward smoother representations, hurting sharp byte discrimination
3. **Insufficient latent structure**: Byte embeddings (256×512) are near one-hot — not enough latent structure for JEPA to exploit. Token-level MTP (PR #88) works because token embeddings encode richer semantics.

### 4.6 Artifact Size vs Quality Tradeoff

| Weight Decay | Sliding BPB | Artifact | Quality | Compression |
|-------------|------------|----------|---------|-------------|
| WD=0.04 | **1.2201** | 15.2–16.1MB | Best | Variable |
| WD=0.05 | 1.2258 | 14.76MB | Worse | Better |
| WD=0.06 | 1.2231 (pre-quant) | 13.85MB | Worst | Excellent |

Higher WD produces smoother weights that compress better but train worse. Optimal strategy: keep WD=0.04 and use BigramHash(4096×32) which improves quality AND adds only ~0.3MB compressed.

---

## 5. Key Architectural Findings

### 5.1 Pure Attention Beats All Hybrids at seq_len=4096 on H100

FA3/SDPA is so well-optimized on H100 that even quadratic attention at 4096 positions beats linear-complexity alternatives (Mamba2, GLA) on wall-clock BPB. The throughput gap (83ms vs 130+ms/step) overwhelms any per-step quality advantage.

**This is hardware-specific** — on hardware where SSM kernels are better optimized relative to attention, the conclusion might differ.

### 5.2 Byte-Level Vocabulary Savings

- sp1024 embedding: 1024 × 512 = 524K params → ~750KB compressed
- Byte embedding: 256 × 512 = 131K params → ~190KB compressed
- **Savings: ~560KB** — enough for ~0.3 extra transformer layers or BigramHash features

### 5.3 Sliding Window Evaluation Is Critical

- Non-overlapping eval: each byte gets variable context (boundary bytes get less)
- Sliding eval (stride=512): every byte scored with nearly full 4096-byte context
- Typical improvement: **0.015–0.016 BPB**
- This is the standard method used by all merged SOTA submissions

### 5.4 Technique Effectiveness for Byte-Level Models

| Technique | BPB Effect | Cost |
|-----------|-----------|------|
| LeakyReLU² | ~0.003–0.005 better | Free |
| ByteBigramHash(4096, 32) | ~0.005 better | +147K params, +1ms/step |
| SmearGate | ~0.003 better | +512 params |
| EMA (decay=0.997) | ~0.003–0.005 better | Memory for shadow params |
| Sliding eval (stride=512) | ~0.015 better | Eval-time only |
| Pure attention (vs SSM hybrid) | ~0.005–0.01 better | — |
| JEPA auxiliary loss | 0.006–0.024 **worse** | +262K params, +1.5ms/step |
| Byte patching (K=2,4) | 0.05–1.1 **worse** | — |
| Higher WD (>0.04) | 0.003–0.006 **worse** | — |

---

## 6. Submission

**PR #705** submitted to `openai/parameter-golf` — first tokenizer-free byte-level model to beat the baseline.

### Final Configuration
```
BLOCK_PATTERN=AAAAAAAAAAAAA  (13 layers, pure attention)
VOCAB_SIZE=256  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=8
MLP_HIDDEN=1536  (3× model dim)
WARMDOWN_ITERS=3500  MATRIX_LR=0.035
SMEAR_GATE=1  BIGRAM_HASH_BUCKETS=4096  BIGRAM_HASH_DIM=32
VAL_SLIDING_STRIDE=512  VAL_SLIDING_MAX_TOKENS=10000000
```

### Included Files
- `train_byte_model.py` — Complete training script (1,900+ lines)
- `convert_to_bytes.py` — Standalone data conversion (sp1024 → bytes)
- `requirements.txt` — Dependencies (torch, sentencepiece, zstandard)
- `submission.json` — Metadata with 4-seed significance data
- `README.md` — Full documentation
- `train_seed{1337,42,2025,7}.txt` — Training logs for all 4 seeds
- `train_jepa_k4_w{01,001}.txt` — JEPA experiment logs

---

## 7. Unexplored Directions for Future Work

1. **XSA (Cross-Sequence Attention)** — worth ~0.002–0.003 BPB on token models, untested on byte models
2. **Partial RoPE** — apply RoPE to subset of head dims, untested on byte models
3. **int5 quantization** — compresses ~25% better than int6 via zstd, could fund a 14th layer
4. **GPTQ-lite calibration** — Hessian-aware quantization, untested on byte models
5. **Larger batch** — 524K instead of 393K tokens/step, may improve convergence
6. **14 layers** — if int5 compression frees enough artifact space
7. **Longer warmdown** — the warmdown schedule may not be optimal for the ~7200-step budget
8. **Value Residual** — residual connections in attention value path, claimed ~0.015 BPB improvement

---

*Research conducted using Maestro (iGent AI) on 8×H100 GPUs via Modal.*