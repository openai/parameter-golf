> **Mechanistic Interpretability:** For a deep-dive analysis of this model — including SVD utilization, quantization sensitivity, logit lens, and calibration — see [Mechanistic Interpretability of #1105](https://abay.tech/posts/pr-1105-model-autopsy).

## Results: val_bpb 1.0962 (3-seed mean) | 8×H100 SXM | 600s

**sp4608 tokenizer, N_INT6=66 (all int6), QK_GAIN=5.0**

| Seed | Steps | ms/step | Post-quant BPB | **Submission BPB** | Artifact |
|------|-------|---------|----------------|-------------------|----------|
| 42 | 6,388 | 93.9 | 1.0995 | **1.0970** | 15,790,122 |
| 314 | 6,381 | 94.1 | 1.0977 | **1.0952** | 15,789,088 |
| 1337 | 6,379 | 94.1 | 1.0990 | **1.0965** | 15,797,523 |
| **Mean** | **6,383** | **94.0** | **1.0987** | **1.0962** | |

**Post-quant BPB** and **Submission BPB** come from two separate evaluation runs of the same model, not two passes. The sliding-window eval measures the quantized neural model alone; the n-gram eval adds the causal tilt on top. Both are single-pass, single-run evaluations — we report both for ablation (isolating the n-gram contribution).

N-gram eval: ~87s wall (~50s loop, 3-seed mean). Tilted: 23.8% (mean) | Hits: 53.2% (mean).

<details><summary><b>What does the improvement look like? Side-by-side generation (temp=0.8)</b></summary>

**Prompt (50 tokens):** *"Insurance Company Declares Living Man Dead George Johannesen is very much alive. Which is why it was so surpr"*

| #1019 model | **This model** |
| :--- | :--- |
| ising to be insurance company. Living Man? Maybe. All Living Man? It was a miracle. However, right now, Living Man is still dead. Well, why is he not living? Living man Rachel Drobles is a person whose life is tested by the techniques of the car industry. Like all techniques, the life of Rachel is tested by the astounding things she has become. Therefore, whether it is a mistake, or in-residence, or a mistake happening, is perpetuated. Therefore, Lyon Man is dead. Can Living Man H | ising to be insurance company. Living Man is the only insurance company that makes property insurance. It is a company that makes vacation insurance, but it still has the same degree of property insurance. So how does a living man die? So say I think there are ways to teach people how to be insured. The first step is to measure the health of the living man and the stress of his situation. To measure the health of the living man, it is important to measure his or her weight. What is the hazard to the living man? Living Man is the only insurance company that specializes in repairs |

The old model drifts into incoherence ("Rachel Drobles... techniques of the car industry... Lyon Man is dead"). The new model stays on topic — insurance, health measurement, living man — and maintains grammatical coherence throughout. Both are wrong (the real text is about a cancelled driver's license), but the new model's errors are at least topically plausible.

</details>

## Changes vs our #1019

### 1. Fused MLP Kernels: Triton TMA Forward + CUTLASS EVT Backward

**Forward (Triton TMA):** Fuses `F.linear(x, up_w) → LeakyReLU(0.5) → square` into a single kernel. The 302MB intermediate never touches HBM.

**Backward (CUTLASS EVT):** Fuses `(go @ down_w.T) * act_grad` into a single CUTLASS 3.x kernel via Epilogue Visitor Tree. The elementwise multiply runs in the GEMM epilogue while tiles are still in registers — eliminating one 302MB write + read per layer.

**Key design insight — pre-computed activation gradient:** We store the activation gradient in the forward pass instead of the pre-activation:

```
Standard:  store pre, recompute grad in backward with branch
Ours:      act_grad = (pre > 0) ? 2·pre : 0.5·pre    ← one branch, forward only
           post    = 0.5 · act_grad · pre              ← branch-free recovery
           dpre    = (go @ W_down.T) * act_grad         ← branch-free backward
```

The identity `post = 0.5 · act_grad · pre` holds for both signs because:
- pre > 0: act_grad = 2·pre → 0.5 · 2pre · pre = pre² = LeakyReLU(0.5)(pre)² ✓
- pre ≤ 0: act_grad = 0.5·pre → 0.5 · 0.5pre · pre = 0.25·pre² = (0.5·pre)² ✓

This eliminates all branching from the backward, reducing the CUTLASS EVT epilogue to a trivial 3-node tree: `Sm90EVT<multiplies, AccFetch, AuxLoad>`. No conditionals in the kernel.

CUTLASS EVT is a hard dependency — no silent fallback. See [Appendix A.3](#a3-kernel-benchmarks) and [A.4](#a4-step-time-profile) for detailed benchmarks.

### 2. Fast Causal N-Gram Tilt & Subword Certainty (~0.0025 BPB, ~295× speedup)

**Architecture Shift: Sparse Auxiliary Memory**
This PR replaces the old eval-time n-gram mixing path with a fast, legal, single-pass causal n-gram tilt system. The core change is that the n-gram is no longer treated as a second language model. Instead, it acts as a sparse auxiliary memory that proposes a hinted token from the strict prefix, while the neural model remains the full normalized distribution. We then apply a one-token exponential tilt directly on the GPU.

<details><summary><b>Motivation & Interpretability</b></summary>

This work was guided by the interpretability results in our [#1019 model autopsy](https://abay.tech/posts/pr-1019-model-autopsy) and [#1105 model autopsy](https://abay.tech/posts/pr-1105-model-autopsy). Those analyses showed that the model is not broadly weak at language modeling; it is specifically weak at exact copy/repetition. In particular, it has very limited induction capability, while much of the remaining loss is in categories like numbers, punctuation, and whitespace where generic short-order n-grams do not help much.

That changed the design target. Instead of building “better PPM everywhere,” we focused on the narrow places where n-grams are actually complementary:
- High-order token memory for exact repeats.
- Within-word memory for BPE completion.
- Word-start memory for local token prediction.

</details>

<details><summary><b>The Key Insight: Mechanical Subword Certainty</b></summary>

Initially, within-word BPE completions seemed redundant since the neural baseline already assigns high probability to these tokens. However, the most significant BPB drop (~0.002, characterized on the MLP 3.0× model) was unlocked by aggressively lowering the `within_threshold` from 0.80 to 0.25, allowing the expert to fire on 35.7% of positions.

Why it works: While the neural model knows subword patterns, it inherently hedges its bets by distributing probability mass across alternatives. The n-gram expert acts as a mechanical override, capturing the absolute certainty of BPE completions that the neural model refuses to assign a 1.0 probability to.

</details>

Measured eval time (8×H100): **~88s** (setup 37s + loop 50s). Our C++ open-addressing hash achieves **~295× speedup** over naive Python PPM implementations — the enabling constraint that makes causal n-gram tilt feasible within the 600s eval budget. See [Appendix A.5](#a5-n-gram-engineering-details) and [A.6](#a6-n-gram-benchmarks) for full engineering details and benchmarks.

### 3. Brotli-11 Compression (replaces LZMA-9)

−581 KB (−5.9%) vs LZMA-9. Independently discovered; #1089 (@mikeapedia) also uses Brotli.

### 4. Memmap Multi-Shard Data Pipeline + GPU Prefetch

Coprime-stride sampling, daemon thread, CUDA stream prefetch. Credit: @DeepReinforce (#726).

### 5. MLP 3.5× (1536 → 1792 hidden dim)

**Motivated by mechanistic analysis:** [SVD analysis of our #1019 model](https://abay.tech/posts/pr-1019-model-autopsy) showed MLP at 94.4% rank utilization (fully packed) while attention Q sat at 72.6% (spare capacity). The model was parameter-starved in MLP, not attention — so we made MLP wider.

Increases hidden dim from 3.0 × 512 = 1536 to 3.5 × 512 = 1792 (+2.88M params). With sp4608 (which removed BigramHash and SmearGate), the full model is 31.85M params and fits under 16 MB with uniform int6.

Impact: −0.003 BPB from capacity, +13ms/step on 2×H100 (bigger GEMMs). Credit: #185 (@dttdrv), #344 (@aryanbhosale).

### 6. LR Floor (0.05)

During warmdown, learning rate normally decays to 0. With `lr_floor=0.05`, it stops at 5% of peak instead. Prevents the optimizer from stalling, which helps with quantization-sensitive weight distributions still being refined at end of training.

Impact: ~0.001 BPB. Credit: #130 (@mohosy).

### 7. Vocab 4608

Inspired by #1218 (@ClarkF), which established 4096 as effective. We measured β(V) — bytes per token — across intermediate vocab sizes. Controlled comparison (same architecture, seed 42):

| Vocab | val_loss (nats) | β (bytes/tok) | Submission BPB | Δ BPB |
|---|---|---|---|---|
| 4096 | 2.5311 | 3.320 | 1.0977 | — |
| 4608 | 2.5858 | 3.393 | 1.0970 | −0.0007 |

+2.2% bytes per token, +0.055 nats per-token loss. The larger vocab also freed ~1 MB by making BigramHash and SmearGate redundant (removed), enabling uniform int6 for all 66 layers.


## Negative Results

- **Turbo-Muon (AOL + Polar Express NS4):** +0.0018 BPB worse on 8×H100 AND artifact over 16MB. Early convergence advantage at step 500 doesn't hold at 7000+ steps. Reverted to standard NS5.
- **2:4 Structured Sparsity:** +0.672 BPB. Dead.

**Note:** This model still has known inefficiencies — the sp4608 architecture has not been fully tuned (hyperparameters, layer count, MLP ratio, and quantization bit allocation were carried over from the sp1024 stack). We believe further BPB reductions are achievable.

---

## Appendix

### A.1 Prior Results

<details><summary><b>Prior results: sp1024, val_bpb 1.1052 (3-seed mean)</b></summary>

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | Post-EMA val_loss (pre-quant, pre-ngram) | Artifact |
|------|-------|---------|--------------|-----------------|------------------------------------------|----------|
| 314 | 6,844 | 87.7 | 1.1253 | **1.1046** | 1.90000 | 14,519,698 |
| 999 | 6,846 | 87.7 | 1.1256 | **1.1052** | 1.90050 | 14,517,302 |
| 1337 | 6,828 | 87.7 | 1.1261 | **1.1059** | 1.90130 | 14,525,480 |
| **Mean** | **6,839** | **87.7** | **1.1257** | **1.1052** | **1.90060** | |

Mixed quantization: 10 layers int6, 56 layers int5, no pruning needed.

| Baseline | nats | BPB | Δ nats | Δ BPB | Welch's t | df | p |
|---|---|---|---|---|---|---|---|
| Our #1019 (current SOTA) | 1.88218 | 1.11474 | **−0.01607** | **−0.00952** | −22.28 | 3.09 | < 0.01 |
| Our #549 (prior SOTA) | 1.89002 | 1.11938 | **−0.02392** | **−0.01417** | −28.15 | 3.95 | < 0.01 |

</details>

<details><summary><b>Calibration regression (sp1024 model)</b></summary>

ECE increased from 0.24% (#1019 model) to 1.26% (sp1024 model) — the mixed int5/int6 quantization introduces slight overconfidence. Model entropy dropped from 1.899 to 1.847 nats (more confident) while accuracy dropped from 54.99% to 54.46%. Not yet re-measured on the sp4608 all-int6 model.

</details>

<details><summary><b>Prior results (val_bpb 1.1125, 3-seed)</b></summary>

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 314 | 6,844 | 87.7 | 1.1253 | **1.1123** | 1.87802 | 14,519,698 |
| 999 | 6,846 | 87.7 | 1.1256 | **1.1124** | 1.87821 | 14,517,302 |
| 1337 | 6,828 | 87.7 | 1.1261 | **1.1129** | 1.87910 | 14,525,480 |
| **Mean** | **6,839** | **87.7** | | **1.1125** | **1.8784** | |

</details>

<details><summary><b>SLOT study (removed from submission — causality violation)</b></summary>

SLOT (Selective Logit Offset Tuning) optimizes a 512-dim delta vector at the last hidden layer using AdamW (lr=0.003, 5 steps) per sliding-window batch. It gave −0.0037 BPB (1.1125 → 1.1088), but **violates causality**: the delta has shape `[1,1,512]` and is optimized using targets at all positions, then applied to all positions — so position t's prediction is influenced by future tokens through the shared delta. Removed from submission code; results below are for reference only.

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 314 | 6,812 | 87.7 | 1.1256 | **1.1086** | 1.87174 | 14,519,020 |
| 999 | 6,833 | 87.7 | 1.1259 | **1.1086** | 1.87187 | 14,522,798 |
| 1337 | 6,811 | 87.7 | 1.1265 | **1.1093** | 1.87306 | 14,526,779 |
| **Mean** | **6,819** | **87.7** | | **1.1088** | **1.8722** | |

Credit: #609 (@saml212).

</details>

<details><summary><b>Prior results: fused kernels + Brotli only (val_bpb 1.1138, 3-seed)</b></summary>

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|---------------|-----------------|-----------------|----------|
| 314 | 7,176 | 83.4 | 1.1325 | **1.1133** | 1.8798 | 15,507,095 |
| 42 | 7,173 | 83.7 | 1.1325 | **1.1134** | 1.8800 | 15,498,065 |
| 999 | 7,176 | 83.5 | 1.1339 | **1.1146** | 1.8820 | 15,512,666 |
| **Mean** | **7,175** | **83.5** | | **1.1138** | **1.8806** | |

Delta vs #549: −0.00943 nats. Welch's t = −10.26, df ≈ 3.78, p < 0.01.

</details>

### A.2 Throughput Recovery (sp1024 stack)

<details><summary><b>Throughput progression (sp1024, prior to sp4608 migration)</b></summary>

| Submission | ms/step | BPB | What changed |
|---|---|---|---|
| Our #549 | 83.4 | 1.1194 | Leaderboard SOTA baseline |
| Our #1019 (merged SOTA) | 86.7 | 1.1147 | +Full GPTQ, +XSA-all, +BigramHash 3072. **+3.3ms regression.** |
| This PR (kernels only) | 83.5 | 1.1138 | +Fused MLP kernels, +Brotli. **Regression erased.** |
| This PR (sp1024 full stack) | 87.7 | 1.1052 | +MLP 3.5×, +mixed int5/int6, +LR floor. |

Our #1019 traded throughput for quality — full Hessian GPTQ and BigramHash 3072×112 added 3.3ms/step. Fused MLP kernels recover that regression. With sp4608, BigramHash was removed entirely (redundant at larger vocab), and all layers use int6.

</details>

### A.3 Kernel Benchmarks

<details><summary><b>Kernel benchmarks + incremental deltas (2×H100)</b></summary>

**Per-layer kernel timing:**

| Variant | dpre time | Δ per layer | Δ per step (×11) |
|---|---|---|---|
| cuBLAS unfused | 1.221 ms | baseline | baseline |
| Triton precomp | 1.105 ms | −0.116 ms | −1.275 ms |
| **CUTLASS Pingpong** | **1.073 ms** | **−0.148 ms** | **−1.623 ms** |

CUTLASS vs Triton: +0.032 ms/layer, +0.347 ms/step kernel-level.

**End-to-end training (35 steps, seed=42):**

| Config | Step avg | Δ |
|---|---|---|
| Triton fwd + Triton bwd | 313.90 ms | baseline |
| Triton fwd + CUTLASS EVT bwd | 313.47 ms | −0.43 ms |

Kernel-level 0.347ms translates to 0.43ms end-to-end (cache/scheduling interactions).

**8×H100:** 86.7ms (our #1019, unfused) → 83.5ms (this PR) = **−3.2ms/step (−3.7%)**.

</details>

### A.4 Step-Time Profile

<details><summary><b>Step-time profile — where all 313ms goes (2×H100, Nsight)</b></summary>

| Component | Share | |
|---|---|---|
| Flash Attention 3 (fwd+bwd) | 20.1% | FA3 forward alone is 17.5% — single largest kernel |
| **Fused MLP (Triton+CUTLASS)** | **13.5%** | **Our optimization target** |
| cuBLAS GEMMs (MLP bwd dW/dx, attn proj) | 19.1% | Remaining unfused matmuls |
| torch.compile fusions (cross-layer) | 21.6% | **Preserved — see below** |
| Unfused elementwise (LN, residuals) | 21.0% | |
| Communication + other | 4.7% | NCCL, copies, softmax |

**Why surgical fusion, not full-MLP autograd.Function:** The 21.6% from torch.compile's cross-layer fusions (RMSNorm backward, residual adds, RoPE backward) only exists because these ops are visible to the compiler. Wrapping the full MLP backward in `autograd.Function` makes it opaque to Inductor — all backward GEMMs plus cross-layer fusion run in eager mode, **2.7× slower net** (identified in our #670). We fuse only forward and one backward GEMM+pointwise, preserving the compiler's scope.

Top individual kernels:
| Kernel | Share |
|---|---|
| FA3 forward (device_kernel) | 17.3% |
| Fused MLP (\_fused\_leaky...) | 13.4% |
| cuBLAS GEMM (nvjet 256×128) | 12.2% |
| elementwise_kernel | 7.8% |
| vectorized_elementwise_kernel | 5.6% |
| vectorized_layer_norm_kernel | 4.3% |

Wall-clock breakdown: forward+backward compute ~94%, NCCL ~1.6%, CPU overhead ~4.1%.

</details>

### A.5 N-Gram Engineering Details

<details><summary><b>Engineering Overhaul</b></summary>

Previous attempts at n-gram blending using flat tables and Python/NumPy logic were bottlenecked by severe hash collisions and massive FFI overhead. Initial runs with a logistic mixer yielded a catastrophic +0.210 BPB degradation because collision noise was inflating token probabilities.

By migrating to an open-addressing scheme (64M entries, 26-bit) to store exact keys, we eliminated false positives, pushing token PPM accuracy to 82.3%. To solve the execution bottleneck, we deployed a highly optimized pipeline:
- **C++ Fused Kernels:** Moved the exponential tilt and blending logic entirely to a custom C++ operator (`fused_expert_blend.cpp`).
- **FFI Bottleneck Eradicated:** Switched to nanobind batch calls instead of per-token ctypes, achieving a 9,583× reduction in FFI overhead.
- **GPU Utilization:** By keeping the tilt on the GPU and eliminating blend_max wait times, GPU utilization spiked from ~10% to 87-94%.

</details>

### A.6 N-Gram Benchmarks

- Measured eval time (8×H100): **~88s** (setup 37s + loop 50s)
- vs naive Python PPM (O(history_size) brute-force scan, 7.3K tok/s): **~295× faster** (our C++ hash: 2,161,532 tok/s)
- GPU utilization: **87–94%** (9,583× FFI overhead reduction via nanobind vs per-token ctypes)

The ~295× speedup vs naive Python was the enabling constraint: a brute-force per-token PPM would take hours on 44M+ tokens; our C++ open-addressing hash with batched nanobind calls runs in ~20s (n-gram lookup only), well within the 600s eval budget.

### A.7 Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Tokenizer | **sp4608 (vocab 4608)** | **This work** (inspired by #1218 (@ClarkF)) |
| Layers | 11 (512d, 8 GQA / 4 KV heads) | Baseline |
| MLP | **3.5× (1792)**, LeakyReLU(0.5)² | **This work** (concept: #185 (@dttdrv), #344 (@aryanbhosale)) |
| MLP Forward | **Fused Triton TMA kernel** | **This work** (profiling: our #670) |
| MLP Backward | **CUTLASS EVT Pingpong + pre-computed act_grad** | **This work** |
| Attention | XSA on all 11 layers | #478 (@gowtham0992) |
| ~~BigramHash~~ | Removed — redundant at vocab 4608, freed space for all-int6 | Previously: Our #1019 |
| RoPE | Partial (16/64 dims) | #315 (@jfprincz) |
| LN Scale | 1/√(layer+1) | #315 (@jfprincz) |
| VE128 | Layers 9-10 | #374 (@unnir) |
| ~~SmearGate~~ | Removed — redundant at vocab 4608, freed space for all-int6 | Previously: #65 (@aquariouseworkman) |
| U-Net skips | Encoder-decoder connections | #289 |
| Weight avg | EMA(0.997) + SWA(every 50) | #401 (@newjordan) |
| Quantization | **Full Hessian GPTQ all-int6 (66 layers, AR self-gen)** | **This work** (GPTQ: #535 (@raahilshah)) |
| Compression | **Brotli quality=11** | **This work** (independently: #1089 (@mikeapedia)) |
| Data Pipeline | **Memmap multi-shard + GPU prefetch** | #726 (@DeepReinforce) |
| Warmdown | 4000 iterations, **LR floor 0.05** | #364 (@shikhar1729), **LR floor: #130 (@mohosy)** |
| Optimizer | Parallel Muon (NS5) | Our #399 |
| Late QAT | STE at LR scale < 0.15 | #286 (@chris-buckley) |
| Selective pruning | ±1 by reconstruction error | #609 (@saml212) |
| Eval | Sliding window (stride=64) | Standard |
| Flash Attention 3 | Hopper kernels | #122 (@mtybadger) |

**Calibration legality:** AR self-generated (64 seqs × 2048 tokens, temp=0.8). No val data, no train data accessed during quantization. Same method as our [#1019](https://github.com/openai/parameter-golf/pull/1019).

### A.8 Setup & Reproduction

```bash
# 1. Python dependencies (torch 2.11.0 + CUDA 13.0 + FA3)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2110
pip install sentencepiece brotli scikit-build-core nanobind

# 2. Set library paths BEFORE building (required for linking)
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$(python3 -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0] + '/lib')"):${LD_LIBRARY_PATH:-}

# 3. CUTLASS headers (header-only, no build needed for CUTLASS itself)
cd /opt && git clone --depth 1 --branch v3.7.0 https://github.com/NVIDIA/cutlass

# 4. Build CUTLASS EVT extension
cd cutlass_evt_fusion
CUTLASS_PATH=/opt/cutlass python3 setup.py build_ext --inplace
cd ..

# 5. Build Fast N-Gram Fused Extension
pip install .

# 6. Download data
python3 data/cached_challenge_fineweb.py --variant sp4608

# 7. Train (3 seeds)
for SEED in 42 314 1337; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```

