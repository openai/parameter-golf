# Why Novel Architectures Fail at 16MB: Throughput-Quantization Co-optimization in Parameter Golf

**Non-record research submission** | 6 experiments on 8xH100 SXM | Base: PR #549 (1.1194 BPB)

## Summary

We systematically evaluated 6 architectural innovations from recent papers (March 2026) on the PR #549 SOTA stack. All failed. The unified finding: **at 16MB/600s, the binding constraint is not model quality but throughput-quantization co-optimization.** The SOTA stack is a local optimum where every component (Parallel Muon, torch.compile, int6 per-row quantization, parameter banks) is co-designed for H100 tensor core throughput. Any modification — even theoretically superior — breaks this pipeline and loses more from overhead than it gains in quality.

## The Throughput Tax

At 83ms/step (PR #549's speed), each millisecond of per-step overhead costs ~7 training steps. Each step at convergence improves BPB by ~0.001. **Therefore, any technique must improve BPB by at least 0.007 per millisecond of overhead it adds.** No technique we tested clears this bar.

## Experiments

### 1. MUD Optimizer (arXiv:2603.17970) -- Negative Result

**Hypothesis:** Replace Newton-Schulz iteration with Cholesky whitening for 10-50% faster optimizer steps.

**Result:** 5% SLOWER (87.9ms vs 83ms). NaN divergence requiring diagonal regularization (1e-6). Final BPB: 1.1581 (vs 1.1194 SOTA).

**Why it failed:** `torch.linalg.solve_triangular` doesn't support bf16 on CUDA — requires float32 cast. H100 tensor cores run batched matrix multiply (NS5) at 989 TFLOPS; triangular solve is memory-bandwidth-bound at ~200 GB/s. The paper's FLOP advantage (12x fewer ops) is irrelevant when the bottleneck is memory bandwidth, not compute.

**Insight:** Optimizer innovations must match the batched-GEMM-on-tensor-cores paradigm. Sequential operations (triangular solve, scan, recurrence) cannot compete on current hardware.

### 2. Information-Maximizing Architecture -- Mixed Result

**Hypothesis:** LeakyReLU(0.9)^2 + XSA-all(11) + Partial RoPE 12/64 + Progressive LN Scale improve by preserving information flow.

**Result:** 1.1261 BPB (89ms/step, 6,737 steps). Better than PR #287 (1.1271) but 0.0067 behind SOTA.

**Why it didn't beat SOTA:** XSA-all adds ~6ms/step (XSA on 7 additional layers), costing ~400 steps. The -0.002 BPB gain from XSA-all doesn't compensate. Progressive LN Scale and Partial RoPE 12/64 were approximately neutral.

**Insight:** XSA follows diminishing returns — the deepest 4 layers capture most of the self-value bias. Extending to all layers trades throughput for marginal quality.

### 3. Hourglass FFN (arXiv:2602.06471) -- Negative Result

**Hypothesis:** Split MLP into K=2 sub-blocks with residual connections for deeper per-layer computation at same parameter count.

**Result:** Pre-quant 1.1539, int6 roundtrip **1.4811** (+0.33 BPB quantization gap). Catastrophic.

**Why it failed:** Splitting the MLP weight bank into two sub-blocks creates weight distributions that int6 per-row quantization cannot handle. Standard MLP weights have heterogeneous row magnitudes — int6's per-row scaling naturally adapts. Split sub-block weights have more uniform, smaller magnitudes — the quantization grid becomes too coarse relative to the weight variance.

**Insight:** MLP shape affects quantizability. Any architectural change must preserve the weight distribution characteristics that int6+lzma is optimized for.

### 4. nGPT Hypersphere Normalization (arXiv:2410.01131) -- Negative Result

**Hypothesis:** Normalize all vectors to unit norm on a hypersphere for 4-20x faster convergence, eliminating LayerNorm.

**Result:** Pre-quant 1.3632, int6 roundtrip **1.7134** (+0.35 quant gap). 122ms/step (46% slower). Artifact only 8.38MB (weights compress well under lzma but quality destroyed).

**Why it failed:**
1. **Quantization incompatibility:** Unit-norm weights concentrate all values in a narrow range (+-0.044 for d=512). Int6's 64 levels can't resolve the angular relationships the model relies on. Small quantization errors destroy the precise geometry of the hypersphere.
2. **Throughput:** F.normalize called 44 times per forward pass + post-step weight normalization = 46% overhead.
3. **Convergence:** The 4-20x speedup claim (tested at 0.5-1B scale) doesn't transfer to 27M scale with SmearGate/XSA/BigramHash, all designed for unnormalized residual streams.

**Insight:** Weight normalization and low-bit quantization are fundamentally incompatible. Normalized weights need angular-aware quantization, not per-row uniform quantization.

### 5. TrigramHash Embedding -- Marginal Negative Result

**Hypothesis:** Extend BigramHash to 3-gram context for -0.008 BPB at ~221KB artifact cost.

**Result:** 1.1298 BPB (98ms/step, 6,098 steps). Quant gap healthy (+0.009).

**Why it didn't help net:** Hash computation + embedding lookup + projection adds ~15ms/step overhead, costing ~1,100 steps. The -0.008 BPB gain is eaten by the -0.010 BPB loss from fewer steps.

**Insight:** Even cheap operations (hash + lookup) fail the throughput tax at 83ms/step. The bar for "zero overhead" is extremely high — only changes to constants (activation slopes, initialization values) truly qualify.

### 6. SSM-Transformer Hybrid (GatedDeltaNet, ICLR 2025) -- Negative Result

**Hypothesis:** Replace middle 4 transformer layers with GatedDeltaNet (linear recurrence) for long-range context without quadratic attention.

**Result:** 1.2516 BPB (282ms/step, 2,126 steps). Artifact 17.78MB (over budget).

**Why it failed:**
1. **No torch.compile:** GatedDeltaNet's Triton kernels break `torch.compile(fullgraph=True)`. Without compile, step time explodes 3.4x.
2. **Over budget:** GatedDeltaNet adds ~6x hidden_size^2 params per layer (1.58M/layer), pushing past 16MB.
3. **Memory-bound:** Recurrent scan operations can't use H100 tensor cores.

**Positive finding:** Per-step loss quality matches transformers (loss 2.25 at step 1000 for both). GatedDeltaNet learns at equivalent rate per gradient update — it's purely a throughput problem.

**Insight:** SSM-transformer hybrids need torch.compile support and hardware-native scan kernels to become competitive. The FLA library's Triton kernels are fast but not compile-compatible.

## The Unified Finding

The PR #549 SOTA stack is not just a good architecture — it's a **co-optimized system** where:

1. **Parallel Muon** packs all weights into 4 contiguous 3D banks for batched Newton-Schulz
2. **torch.compile(fullgraph=True)** fuses the entire forward pass into optimized CUDA kernels
3. **Int6 per-row quantization** is calibrated for the specific weight distributions produced by this architecture + optimizer combination
4. **H100 tensor cores** run the batched GEMM operations at peak throughput

Breaking any one of these four pillars cascades into the others:
- New optimizer → breaks batched bank structure → loses Parallel Muon speedup
- New layer type → breaks torch.compile → loses fusion speedup
- New weight distribution → breaks int6 calibration → catastrophic quantization
- New operation type (scan, solve) → can't use tensor cores → memory-bandwidth-bound

**To genuinely beat this SOTA, you need to co-optimize ALL FOUR simultaneously.** The ternary submission (PR #640) succeeded exactly because it did this: different quantization (ternary) + different optimizer (NeoMuon) + different architecture (768d, 8192 BPE) + 250 experiments to co-optimize everything.

## Implications for Small Model Design

These findings transfer beyond this competition:

1. **Throughput-aware architecture search:** At constrained compute, evaluate architectures by BPB-per-second, not BPB-per-step
2. **Quantization-aware architecture design:** Novel MLP shapes and normalization schemes must be validated under target quantization BEFORE committing to them
3. **Co-optimization is mandatory:** At small scale, architecture-optimizer-quantization-hardware form a tightly coupled system. Optimizing one in isolation is insufficient.
4. **The "throughput tax" formula:** For any technique adding T ms/step overhead at S ms/step baseline, it must improve BPB by at least T/(S * 600/S) * step_bpb_rate to break even

## Hardware

8x NVIDIA H100 80GB SXM, RunPod. PyTorch 2.8.0+cu128. FA3 via windreamer wheels. Total compute: ~$150 across 7 runs.

## Related Work

- PR #296: Reptile meta-learned TTT (our earlier submission, cited by Issue #140)
- PR #303: XSA + TTT negative interaction study
- PR #318: Neural Cache concept (cross-window KV caching)
- PR #640: Ternary quantization (the paradigm-shift submission that succeeded by co-optimizing everything)

## Author

Xiaoan Liu | NYU | GitHub: @sseanliu
