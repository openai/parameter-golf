> **Mechanistic Interpretability:** For a deep-dive analysis of this model — including per-matrix rate-distortion, recurrence error amplification, and skip gate analysis — see [Mechanistic Interpretability of this submission](https://abay.tech/posts/pr-1420-model-autopsy).

# Triple Loop + Fused Kernels + Parallel Residuals + N-gram Tilt

**val_bpb: 1.08309** (5-seed mean, std=0.00044)

| Seed | Steps | **SW BPB** | **Tilt BPB** | Artifact |
|-|-|-|-|-|
| 1    | 4771 | 1.08271 | **1.08256** | 15,978,345 |
| 42   | 4769 | 1.08391 | **1.08376** | 15,975,585 |
| 1234 | 4692 | 1.08344 | **1.08330** | 15,973,639 |
| 1337 | 4756 | 1.08301 | **1.08287** | 15,974,187 |
| 2025 | 4755 | 1.08309 | **1.08295** | 15,970,317 |
| **Mean** | | **1.08323** | **1.08309** | |

## Changes

* **One extra loop pass through layers 4-5.** PR #1394 passes through layers 4-5 three times total (NUM_LOOPS=2, giving 15 virtual layers from 11 physical). I add a fourth pass (NUM_LOOPS=3), giving 17 virtual layers. The encoder becomes `[0,1,2,3,4,5,4,5]` and the decoder `[4,5,4,5,6,7,8,9,10]`. It costs about 200 training steps, but the extra depth more than compensates. Quadruple looping (19 virtual) was worse because the step count drops too far.

* **Activate looping earlier (0.35 instead of 0.50).** At 0.50, half the training budget runs without the looped layers doing anything. I swept `{0.30, 0.35, 0.40, 0.50}` on seed 1234. 0.35 won, though 0.40 was close. Below 0.35 the model doesn't get enough non-looped warmup and quality degrades.

* **Fused MLP kernels (Triton TMA forward + CUTLASS EVT backward).** This took the most engineering effort and gave the most BPB back. The forward fuses `leaky_relu(fc(x), 0.5).square()` into a single Triton TMA kernel so the 403MB intermediate never hits HBM. The backward fuses `(grad_out @ proj.weight) * act_grad` into a CUTLASS 3.x Epilogue Visitor Tree, running the elementwise multiply while tiles are still in registers. Together: ~10% higher throughput, +127 training steps in the same 600s. I initially tried wrapping the entire MLP in a custom `autograd.Function`, but that killed `torch.compile`'s cross-layer fusions and made everything 2.7x slower. The trick was to fuse *surgically*, just the forward activation and one backward GEMM, and let the compiler handle the rest. Details in [Appendix A.1–A.3](#a1-fused-mlp-kernels-design--implementation).

* **Parallel residuals for layers 7-10.** GPT-J style ([Wang & Komatsuzaki, 2021](https://github.com/kingoflolz/mesh-transformer-jax)): attention and MLP both read from the same pre-residual input, outputs summed in parallel. I expected this to mostly help quantization (less interference between attention and MLP during GPTQ calibration), and it did tighten the gap slightly. The bigger surprise was +68 training steps from the faster forward pass. I also tried Hessian-Aware SDClip from [PR #1412](https://github.com/openai/parameter-golf/pull/1412) alongside this, but it made things worse with triple looping. It probably needs its own λ tuning for the deeper architecture.

* **Eval-time n-gram tilt (causality-fixed).** The original submission had a causality violation in the within-word and word-start hint channels: `is_bnd`/`is_ws` flags were derived from `tokens_[p]` (the target token being predicted), which made the hint-gating decision depend on the target. This was caught by @Gusanidas in review. The fix splits the flags into two sets: prefix-derived flags (`tokens_[p-1]`) for hint gating, and target-derived flags (`tokens_[p]`) for post-scoring state updates. However, the within-word and word-start channels cannot produce useful hints without target-dependent gating — they either fire too broadly or at the wrong positions. After testing all causal alternatives (prev_tok gating, state-based gating, disabling channels), the winning configuration uses **token_hint only** (orders 8-16), which was always fully causal. The remaining token_hint channel provides a consistent -0.00014 BPB across all seeds. The improvement is real but small — most of the original -0.0029 delta came from the (now-removed) target-dependent gating in within/word channels. Full details in [Appendix A.4](#a4-n-gram-tilt-architecture--interpretability).

<details><summary><b>N-gram legality (<a href="https://github.com/openai/parameter-golf/issues/1017">#1017</a> conditions)</b></summary>

**Update (post-review fix):** The original submission had a Rule 1 violation in the within-word and word-start hint channels. The `is_bnd`/`is_ws` flags used to gate hint generation were derived from `tokens_[p]` (the target), making the decision of *whether to produce a hint* depend on the token being predicted. This was caught by @Gusanidas. The fix removes the within-word and word-start channels from hint output entirely — they cannot produce useful hints without target-dependent gating. Only the `token_hint` channel (orders 8–16) remains, which was always fully causal. The n-gram delta dropped from -0.0029 to -0.00014 BPB.

Audited against the four conditions proposed in [#1017](https://github.com/openai/parameter-golf/issues/1017) for eval-time adaptation:

**Condition 1, Causal dependence** (`p_t` depends only on artifact + `x_1...x_{t-1}`): `compute_hashes` reads `tokens[pos - k - 1]` for k=0,1,..., all strictly before position `pos`. `token_hint` looks up hash tables containing only entries inserted by prior iterations. The target token `tokens[pos]` is read only for the post-scoring *update* phase.

**Condition 2, Full normalized distribution**: The tilted distribution is `p_tilt(t) = p_model(t) · exp(β · 1[t==hint]) / Z` where `Z = 1 + p_model(hint) · (exp(β) - 1)`. Proper probability distribution over the full vocabulary.

**Condition 3, Score-before-update**: Hint and beta are written to output arrays before `token_update` inserts `tokens[pos]` into the tables.

**Condition 4, Single left-to-right pass**: `get_hints_batch` processes positions sequentially. The sliding window scores each token exactly once.

</details>

* **Double-buffered async data prefetch.** Background thread + pinned memory + separate CUDA stream. I built this to work around the virtualized disk I/O on cloud H100 instances (see below), but it ended up helping in every setting I tested.

* **PyTorch 2.9.1 instead of 2.11.** See below.

## What the model looks like inside

I ran per-matrix rate-distortion, recurrence error amplification, and skip gate analysis on the trained model. Three things stood out:

**Loop layers are 2.2x more sensitive to quantization than non-loop layers.** Blocks 4 and 5 get reused across passes, so rounding error in those weights compounds. The single most sensitive matrix in the entire network (block 4's value projection) has 80x the BPB-per-byte cost of the least sensitive. This suggests mixed-precision quantization (more bits for loop layers) is the biggest remaining opportunity.

**The third loop pass contributes 63% of what the second does.** I measured a contraction ratio of 0.634 across passes: each loop iteration changes the representation by ~63% of the previous one. A hypothetical 4th pass would add only 0.63³ = 25% new information, which matches the empirical finding that quadruple looping hurts. The 3rd pass at 63% is clearly worth the step cost; the 4th at 25% is not.

**All 8 skip connections are load-bearing.** Gates are 0.61-0.70 (sigmoid), meaning roughly 35% encoder / 65% decoder blend. The first loop pass's skip connections (skips 2,3) have the highest weight norms (21.9, 19.5 vs 2.8-13.8 for others), so the first encoder pass through layers 4-5 is the most important information source for the decoder.

<details><summary><b>What the progress looks like: three models on the same prompt (temp=0.8)</b></summary>

**Prompt (50 tokens):** *"Insurance Company Declares Living Man Dead George Johannesen is very much alive. Which is why it was so surpr"*

**Ground truth:** *ising when the Canadian man received a letter addressed "To the Estate of George Johannesen." Even more surprising is that it came from his insurance company, who should really be on top of such things...*

| PR #1019 (1.1147 BPB) | PR #1105 (1.0962 BPB) | **This PR (1.08014 BPB)** |
| :--- | :--- | :--- |
| ising to be insurance company. Living Man? Maybe. All Living Man? It was a miracle. However, right now, Living Man is still dead. Well, why is he not living? Living man Rachel Drobles is a person whose life is tested by the techniques of the car industry. Like all techniques, the life of Rachel is tested by the astounding things she has become. Therefore, whether it is a mistake, or in-residence, or a mistake happening, is perpetuated. Therefore, Lyon Man is dead. Can Living Man H | ising to be insurance company. Living Man is the only insurance company that makes property insurance. It is a company that makes vacation insurance, but it still has the same degree of property insurance. So how does a living man die? So say I think there are ways to teach people how to be insured. The first step is to measure the health of the living man and the stress of his situation. To measure the health of the living man, it is important to measure his or her weight. What is the hazard to the living man? Living Man is the only insurance company that specializes in repairs | ising when the Canadian man received a letter addressed "To the Estate of George Johannesen" George Johannesen was a retired professional who was a lucrative investor in Canada. His estate was worth about $1 billion. His death last month at the age of 73 was a direct shock to the entire estate and he was still alive. That is why he was so shocked. In 2005 he was a member of the Canadian As |

#1019 drifts into incoherence ("Rachel Drobles... techniques of the car industry... Lyon Man is dead"). #1105 stays on topic but loops on "Living Man is the only insurance company." This model picks up the actual narrative thread ("the Canadian man received a letter"), invents plausible biographical details, and maintains coherence throughout. All three are wrong about what happens next, but the errors become progressively more plausible.

</details>

## Debugging the platform

This was the hardest submission I've worked on. Most of the time went to infrastructure, not the model.

**Virtualized disks tank throughput.** The cloud H100 instances I rented use virtio block devices. The coprime-stride data loader from [#726](https://github.com/openai/parameter-golf/pull/726) does random reads across 143 shards, which is fine on bare metal but brutal on a virtual disk. That's what led me to build the async prefetch. It turned out to help everywhere, not just on virtualized storage.

**PyTorch 2.9.1 vs 2.11: a full day lost.** I could not reproduce results from other submissions. Training the same architecture with the same seed gave 0.0042 BPB worse results on torch 2.11. (I initially measured a 0.015 gap, which turned out to be a wrong model file on the server. The real gap, once I controlled for that, was 0.0042.) I swapped Triton versions, disabled autocast, forced cuBLAS backends, diffed Inductor-generated kernels. The root cause was two independent issues:

1. **Autocast backward changed in PR [pytorch#165068](https://github.com/pytorch/pytorch/pull/165068)** (landed Dec 2025, present in 2.11, absent from 2.9.1). Two lines in `cached_cast()` add an `AutoGradMode enable_grad(true)` guard on weight casts, inserting extra `ToCopyBackward` nodes into the autograd graph. This changes floating-point accumulation order by 1 ULP of bf16 (7.15e-7) in saved activations, which compounds over 5000 momentum steps into +60KB of weight entropy. The model goes from fitting at 16.00MB (no pruning) to 16.06MB (5.4% pruning needed). I verified eval is version-invariant to 0.00003 BPB; the entire gap is from training.

2. **Inductor over-fusion in backward codegen**: Inductor 2.11's `mix_order_reduction` fuses `_fused_rms_norm_backward` into adjacent kernels, producing fewer but larger Triton kernels (65 functions / 11,855 lines vs 71 / 11,292 in 2.9.1). The fatter kernels hit register pressure and cost +5.93ms per backward pass (+8.8%). In a 600s budget, that's ~57 lost training steps. I submitted a fix that disables `mix_order_reduction` by default (aligning open-source with fbcode, where it was already off): [pytorch/pytorch#179494](https://github.com/pytorch/pytorch/pull/179494).

Separately, our fused CUTLASS kernel crashed on torch 2.11 because Triton 3.6.0's `TensorDescriptor.from_tensor()` tries to access `.data_ptr()` on FakeTensors during `torch.compile` tracing. I traced that through Inductor's `FallbackKernel` codegen and submitted a second fix: [pytorch/pytorch#179422](https://github.com/pytorch/pytorch/pull/179422). Two PyTorch PRs from a golf competition.

In time-budgeted competitions, the platform *is* the model. A 6ms/step Inductor regression can cost as much BPB as most algorithmic innovations.

<details><summary><b>How this submission came together</b></summary>

The first few days were mostly wasted. I tried improving the architecture directly: 12 layers, SwiGLU, mixed int5/int8 per layer. Nothing worked. The model was 930KB over the 16MB budget and MLP weights alone were 69% of the compressed artifact. Brotli-11 was already within 1-2% of Shannon entropy. There was nowhere to go.

Worse: a new optimizer schedule I'd been developing (Mixed NS5, a convergent Newton-Schulz coefficient ramp) changed the weight distribution enough that the model no longer fit in the 16MB budget. It was 930KB over, and aggressive pruning to fit destroyed the quality gains.

Then I lost a full day to PyTorch version divergence (described above). Besides the upstream fix, the useful thing that came out of it was a proof that compressed model size is a chaotic function of training hyperparameters. 1 ULP of bf16 rounding (7.15e-7) in a saved activation compounds over 5000 momentum steps into 60KB swings in Brotli output. I also proved that L2 weight decay is scale-invariant under max-abs quantization: `Q(γW) = Q(W)`. All the per-bank WD tuning I'd been doing was chasing noise.

Once I stopped trying to control compression through training and focused on what was actually deterministic (GPTQ deadzone for size, n-gram tilt for eval), things moved fast. Clean reproduction of the baseline. Pivot to SP8192 + SDClip. Triple looping. Fused kernels. Parallel residuals. Each gain was small but they stacked: 45 experiments, five seeds, 1.08014 BPB.

</details>

## What didn't work

<details><summary><b>Innovations that worked on earlier models but not here</b></summary>

**Mixed NS5 coefficient schedule.** On our SP4608 model this was worth -0.0066 BPB for free: use the standard Muon polynomial `(3.4445, -4.775, 2.0315)` to ramp singular values toward 1, then switch to the convergent polynomial `(1.875, -1.25, 0.375)` which has `p(1)=1, p'(1)=0` to lock them in. The split adapts per bank based on aspect ratio as a proxy for condition number. On the SP8192 architecture the coefficient schedule produced weight distributions that were hostile to Brotli compression: the model was 500KB over budget and needed 46% pruning.

**EC-GPTQ (entropy-constrained rounding).** Inside the GPTQ inner loop, I added an element-wise deadzone: `dz = λ · d / s²`, where d is the Hessian diagonal and s is the scale. Borderline ±1 values get rounded to 0 when the GPTQ error compensation cost is small. On the SP4096 architecture this achieved 10x better rate-distortion than uniform deadzoning (0.5×10⁻⁵ BPB/KB vs 6.8×10⁻⁵). On the SP8192 + SDClip architecture it was harmful: SDClip's `c = k·σ` already controls entropy per row, and adding EC-GPTQ on top just introduced extra quantization damage for no compression benefit.

**Per-bank weight decay tuning.** MLP is 69% of the compressed model. I tried giving MLP slightly lower WD (0.07 vs 0.09) to improve quality, offset by higher attention WD. Even ±0.005 from the baseline was catastrophic: lower MLP WD means larger MLP weights, which Brotli can't compress cheaply, so the artifact blows up.

**L2 weight decay as a compression lever.** I proved mathematically that L2 WD is scale-invariant under max-abs quantization: `Q(γW) = round(W / (max|W|/31)) = Q(W)`. Multiplying all weights by a constant changes nothing about the quantized integers. This was useful to understand (it meant all the WD-based compression tuning I'd been doing was chasing noise), but it also closed a door.

</details>

| Tried | Effect | Why it failed |
|-------|--------|---------------|
| EC-GPTQ λ=0.0005 on SDClip | +0.00087 worse | SDClip k=12.85 already near-optimal |
| Quadruple loop (NUM_LOOPS=4) | +0.00164 worse | Too few training steps |
| Loop layers 3-4 | +0.00066 worse | Suboptimal depth for recurrence |
| Loop layers 5-6 | +0.00247 worse | Suboptimal depth for recurrence |
| EMA decay 0.998 | +0.00117 worse | Over-smoothing |
| EMA decay 0.996 | +0.00014 worse | Marginal difference |
| Hessian SDClip λ=0.175 | +0.00063 worse | Not tuned for triple loop |
| enable_looping_at=0.30 | +0.00013 worse | Not enough non-loop warmup |
| ETLB (eval-time logit bias) | -0.00020 better | Takes 615s, doesn't fit in 600s eval budget |

## Code size

All code ships as part of the artifact: `train_gpt.py`, CUTLASS EVT source, and the n-gram C++ source. For a competition run, these would be bundled into a single LZMA-compressed blob.

| | Uncompressed | LZMA-9 |
|---|---|---|
| train_gpt.py | 64,137 | |
| cutlass_evt_fusion/ (3 files) | 9,095 | |
| ngram/fused_expert_blend.cpp | 21,589 | |
| **Total** | **73,674** | **19,668** |

`train_gpt.py` is minified with `python-minifier` (annotations, pass statements, and docstrings removed; variable names preserved). `submission.py` (143 bytes) is the entry point: it decompresses `train_gpt.py.lzma` and executes it. For a competition run, `torchrun` would invoke `submission.py` instead of `train_gpt.py`. Total code cost: 19,811 bytes. All 5 seeds fit under 16MB with 1.8-9.9KB headroom. The unminified `train_gpt.py` (64KB) is included in the PR for readability.

## Requirements

- PyTorch 2.9.1+cu128
- Flash Attention 3 (Hopper): `pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`
- CUTLASS EVT extension (compiled for sm_90a, source included)
- SentencePiece, Brotli, NumPy
- 8×H100 80GB SXM

```bash
SEED=1234 NUM_LOOPS=3 ENABLE_LOOPING_AT=0.35 PARALLEL_RESIDUAL_START=7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

<details><summary><b>Full component lineage: every piece traced to its origin PR</b></summary>

| Component in this submission | Origin | Author |
|---|---|---|
| **This PR** | | |
| Triple depth recurrence (NUM_LOOPS=3) | This work | @abaybektursun |
| Earlier loop activation (enable_at=0.35) | This work | @abaybektursun |
| Triton TMA fused MLP forward | [#1105](https://github.com/openai/parameter-golf/pull/1105), ported to SP8192 here | @abaybektursun |
| CUTLASS EVT fused MLP backward | [#1105](https://github.com/openai/parameter-golf/pull/1105), ported to SP8192 here | @abaybektursun |
| Eval-time n-gram tilt (C++ open-addressing) | [#1105](https://github.com/openai/parameter-golf/pull/1105), re-tuned for SP8192 here | @abaybektursun |
| Double-buffered async data prefetch | This work | @abaybektursun |
| PyTorch Inductor bug fixes (2 upstream PRs) | [pytorch#179422](https://github.com/pytorch/pytorch/pull/179422), [pytorch#179494](https://github.com/pytorch/pytorch/pull/179494) | @abaybektursun |
| **Our prior submissions** | | |
| AR Self-Gen GPTQ + XSA-all + BigramHash (merged SOTA) | [#1019](https://github.com/openai/parameter-golf/pull/1019) | @abaybektursun |
| LeakyReLU² + Legal Score-First TTT + Parallel Muon | [#549](https://github.com/openai/parameter-golf/pull/549) | @abaybektursun |
| TTT negative results (why this submission does not use TTT) | [#756](https://github.com/openai/parameter-golf/pull/756), [#1103](https://github.com/openai/parameter-golf/pull/1103) | @abaybektursun |
| **Architecture** | | |
| SP8192 vocabulary | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| SDClip quantization (c = k·σ) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| GPTQ on embeddings (int8) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| Tied embeddings (init_std=0.005) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| SP4096→8192 vocab scaling | [#1218](https://github.com/openai/parameter-golf/pull/1218) | @clarkkev |
| MLP 4.0× width, higher WD (0.085) | [#1218](https://github.com/openai/parameter-golf/pull/1218) | @clarkkev |
| Depth recurrence (loop layers 4-5) | [#1204](https://github.com/openai/parameter-golf/pull/1204) | @msisovic |
| Parallel residuals (GPT-J style) | [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) (2021), adapted in [#1204](https://github.com/openai/parameter-golf/pull/1204) | @kingoflolz, @msisovic |
| MuonEq-R (row-normalized Muon) | [#1217](https://github.com/openai/parameter-golf/pull/1217) | @bigbag |
| U-Net sigmoid-gated skip connections | [#289](https://github.com/openai/parameter-golf/pull/289), refined in [#1089](https://github.com/openai/parameter-golf/pull/1089) | @integrate-your-mind, @mikeapedia |
| XSA on all layers | [#265](https://github.com/openai/parameter-golf/pull/265) (partial), [#478](https://github.com/openai/parameter-golf/pull/478) (all layers) | @unnir, @gowtham0992 |
| Partial RoPE (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz |
| LN Scale (1/√(layer+1)) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz |
| LeakyReLU(0.5)² activation | [#185](https://github.com/openai/parameter-golf/pull/185) | @dttdrv |
| Logit softcap (30.0) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz |
| QK gain (4.0) | [#1125](https://github.com/openai/parameter-golf/pull/1125) | @jainpranjal97 |
| **Optimizer** | | |
| Muon (Newton-Schulz orthogonalization) | [#399](https://github.com/openai/parameter-golf/pull/399) (parallel variant) | @abaybektursun |
| EMA (decay=0.997) | [#315](https://github.com/openai/parameter-golf/pull/315), [#401](https://github.com/openai/parameter-golf/pull/401) | @jfprincz, @newjordan |
| Warmdown (0.667 frac, linear to 0) | [#364](https://github.com/openai/parameter-golf/pull/364) | @shikhar1729 |
| Muon momentum warmup (0.92→0.99) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |
| **Quantization & Compression** | | |
| Full Hessian GPTQ (actorder + Cholesky) | [#535](https://github.com/openai/parameter-golf/pull/535), integrated in [#1060](https://github.com/openai/parameter-golf/pull/1060) | @raahilshah, @dexhunter |
| Brotli-11 + byte shuffle compression | [#1089](https://github.com/openai/parameter-golf/pull/1089) | @mikeapedia |
| **Evaluation** | | |
| Sliding window (stride=64) | [#122](https://github.com/openai/parameter-golf/pull/122) | @mtybadger |
| Flash Attention 3 (Hopper) | [#122](https://github.com/openai/parameter-golf/pull/122) | @mtybadger |
| **Data** | | |
| ShuffledSequenceLoader (memmap + weighted sampling) | [#1394](https://github.com/openai/parameter-golf/pull/1394) | @clarkkev |

This competition is deeply collaborative. Nearly every component traces through multiple contributors. I've tried to credit the earliest PR that introduced each technique, but many were refined across several submissions.

</details>

---

## Appendix

### A.0 Ablation: fused 5-seed without parallel residuals

<details><summary><b>5-seed results: fused kernels + triple loop + n-gram, no parallel residuals</b></summary>

| Seed | Steps | Sliding BPB | N-gram BPB | Artifact |
|-|-|-|-|-|
| 1    | 4703  | 1.08336 | **1.08041** | 15,974,896 |
| 42   | 4704  | 1.08468 | **1.08175** | 15,974,993 |
| 1234 | 4680  | 1.08296 | **1.08007** | 15,971,965 |
| 1337 | 4697  | 1.08363 | **1.08077** | 15,970,370 |
| 2025 | 4702  | 1.08390 | **1.08101** | 15,970,844 |
| **Mean** | | **1.08371** | **1.08080** | |

5-seed mean: **1.08080 BPB** (std=0.00064). Seed 1234 n-gram was run in terminal (1.08007), not logged to file.

Adding parallel residuals (layers 7+) improves seed 1234 from 1.08007 to **1.07971** (-0.00036), primarily from +68 extra training steps due to the faster parallel forward pass. Full parallel-residuals 5-seed results are in the main table above (mean 1.08014).

</details>

### A.1 Fused MLP Kernels: Design & Implementation

These kernels were first developed for [PR #1105](https://github.com/openai/parameter-golf/pull/1105) on the SP4608 architecture. This submission ports them to the SP8192 + triple-loop architecture and integrates the CUTLASS EVT backward with `torch.compile`'s tracing.

<details><summary><b>Forward (Triton TMA): fuses F.linear + LeakyReLU(0.5) + square</b></summary>

Fuses `F.linear(x, up_w) -> LeakyReLU(0.5) -> square` into a single kernel. The 403MB intermediate never touches HBM.

Uses Triton's Tensor Memory Access (TMA) descriptors for H100-native global-to-shared memory loads. Block sizes `128x256x64` with 8 warps, 4 pipeline stages. The kernel performs the GEMM accumulation in FP32, then applies activation and squaring inline before writing back to BF16.

The interleaved write pattern splits the accumulator into two halves via `tl.reshape + tl.permute + tl.split`, writing activation gradient and post-activation to separate output buffers in a single pass.

</details>

<details><summary><b>Backward (CUTLASS EVT): fuses (go @ down_w.T) * act_grad</b></summary>

Fuses `(go @ down_w.T) * act_grad` into a single CUTLASS 3.x kernel via Epilogue Visitor Tree. The elementwise multiply runs in the GEMM epilogue while tiles are still in registers, eliminating one 403MB write + read per layer.

I store the activation gradient in the forward pass instead of the pre-activation. This removes all branching from the backward:

```
act_grad = (pre > 0) ? 2*pre : 0.5*pre    <-- one branch, forward only
post     = 0.5 * act_grad * pre            <-- branch-free recovery
dpre     = (go @ W_down.T) * act_grad      <-- branch-free backward
```

The identity `post = 0.5 * act_grad * pre` holds for both signs:
- pre > 0: act_grad = 2·pre → 0.5 · 2pre · pre = pre² ✓
- pre ≤ 0: act_grad = 0.5·pre → 0.5 · 0.5pre · pre = (0.5·pre)² ✓

This reduces the CUTLASS EVT epilogue to a trivial 3-node tree: `Sm90EVT<multiplies, AccFetch, AuxLoad>`.

</details>

<details><summary><b>Why surgical fusion, not full-MLP autograd.Function</b></summary>

`torch.compile`'s cross-layer fusions (RMSNorm backward, residual adds, RoPE backward) account for ~21.6% of step time. Wrapping the full MLP backward in `autograd.Function` makes it opaque to Inductor, so everything runs in eager mode at 2.7x slower net (I hit this in my [#670](https://github.com/openai/parameter-golf/pull/670)). So I fuse only the forward activation and one backward GEMM+pointwise, preserving the compiler's scope over everything else.

</details>

### A.2 Kernel Benchmarks

<details><summary><b>Per-layer timing and end-to-end</b></summary>

| Variant | dpre time | Delta per layer | Delta per step (x11) |
|---|---|---|---|
| cuBLAS unfused | 1.221 ms | baseline | baseline |
| Triton precomp | 1.105 ms | -0.116 ms | -1.275 ms |
| CUTLASS Pingpong | 1.073 ms | -0.148 ms | -1.623 ms |

End-to-end (35 steps, seed=42, 2xH100):

| Config | Step avg | Delta |
|---|---|---|
| Triton fwd + Triton bwd | 313.90 ms | baseline |
| Triton fwd + CUTLASS EVT bwd | 313.47 ms | -0.43 ms |

On 8xH100: unfused 4553 steps → fused 4680 steps in 588s (+127 steps, +2.8%).

</details>

### A.3 Step-Time Profile

<details><summary><b>Where all 313ms goes (2xH100, Nsight Systems)</b></summary>

| Component | Share |
|---|---|
| Flash Attention 3 (fwd+bwd) | 20.1% |
| Fused MLP (Triton+CUTLASS) | 13.5% |
| cuBLAS GEMMs (MLP bwd dW/dx, attn proj) | 19.1% |
| torch.compile fusions (cross-layer) | 21.6% |
| Unfused elementwise (LN, residuals) | 21.0% |
| Communication + other | 4.7% |

</details>

### A.4 N-Gram Tilt

The n-gram system was originally developed in [PR #1105](https://github.com/openai/parameter-golf/pull/1105) for SP4608 models. This submission ports it to SP8192. Source code: `ngram/fused_expert_blend.cpp` (C++ open-addressing hash, nanobind FFI) and `ngram/eval_ngram.py` (tilt math + sliding window). Eval time on 8xH100: ~90s.

<details><summary><b>Post-review causality fix</b></summary>

The original submission had three hint channels: `token_hint` (orders 8–16), `within_hint` (within-word BPE completion), and `word_hint` (word-start prediction). @Gusanidas identified that `within_hint` and `word_hint` used `is_bnd`/`is_ws` flags derived from `tokens_[p]` (the target token) to gate whether a hint was produced — a Rule 1 violation.

**What was invalid:** The gating decision "should I produce a hint at this position?" depended on whether the target token was a word boundary or had a leading space. This meant the probability distribution P(x_t | x_1...x_{t-1}) changed depending on the value of x_t itself.

**What was tried to salvage within/word channels:**
- Deriving `is_bnd`/`is_ws` from `tokens_[p-1]` (prefix): semantically inverted, delta = +0.00033 (harmful)
- Gating on `within_len_` state only: fires too broadly, delta = +0.00120 (harmful)
- Disabling within/word entirely (token_hint only): delta = **-0.00014** (helpful)

**Conclusion:** The within/word channels' -0.0025 BPB contribution came entirely from target-dependent gating. Without it, they add noise. Only `token_hint` (orders 8–16) produces a legitimate improvement. The fix removes within/word from hint output while keeping their state updates (dead code, no effect).

**Parameter sweep (token_hint only, 4M token subset, 8 GPUs in parallel):**

| base_beta | thresh_scale | table_bits | stride | delta |
|-----------|-------------|------------|--------|-------|
| **1.5** | **0.75** | **26** | **1** | **-0.000083** |
| 1.5 | 0.50 | 26 | 1 | -0.000081 |
| 2.0 | 0.75 | 26 | 1 | -0.000079 |
| 2.0 | 0.50 | 26 | 1 | -0.000074 |
| 1.0 | 1.00 | 26 | 1 | -0.000073 |
| 0.5 | 0.50 | 26 | 1 | -0.000046 |
| 3.0 | 0.50 | 26 | 1 | -0.000020 |
| 5.0 | 0.50 | 26 | 1 | +0.000214 |

Full-val delta with best params (beta=1.5): consistent **-0.00014 BPB** across all 5 seeds. The improvement is real but small.

</details>

<details><summary><b>Causality proof (token_hint channel)</b></summary>

The surviving `token_hint` channel is a textbook online n-gram with strict lookup-then-update discipline:

```cpp
for (int i = 0; i < n; i++) {
    int64_t p = pos[i];
    compute_hashes(tokens_, p, ...);         // (1) hash from tokens[p-1], tokens[p-2], ...
    token_hint(hashes, ..., tok_hint, ...);  // (2) LOOKUP in tables built from pos < p
    hints[i] = tok_hint;                     // (3) emit hint
    token_update(hashes, ..., tok);           // (4) INSERT tokens[p] AFTER hint is emitted
}
```

| Condition | Requirement | Status |
|---|---|---|
| Causal dependence | `p_t` depends only on artifact + `x_1...x_{t-1}` | PASS |
| Full normalized distribution | Proper softmax over full vocab | PASS |
| Score-before-update | Score fixed before any `x_t`-dependent update | PASS |
| Single left-to-right pass | No rescoring | PASS |

</details>

### A.5 Data Prefetch

<details><summary><b>Double-buffered async prefetch</b></summary>

Background thread prepares next batch in pinned memory while GPU trains. Separate CUDA stream for H2D overlap.

On the PR #1334 architecture: +39 steps, +0.7% throughput. The extra steps landed in a worse compression region (+40KB), so the net effect was actually harmful for that architecture. On PR #1394's `ShuffledSequenceLoader` with memmap, the data pipeline is already efficient enough that prefetch isn't the bottleneck.

</details>

### A.6 ETLB (Eval-Time Logit Bias)

<details><summary><b>Algorithm and results</b></summary>

From [PR #1399](https://github.com/openai/parameter-golf/pull/1399). Learns a vocab-sized bias vector via SGD on already-scored context tokens, carried across sliding windows:

1. Forward pass (no grad) → logits
2. 5 SGD steps (lr=0.05) on context tokens (first 1984 of 2048)
3. Score stride tokens (last 64) with `logits + bias`
4. Carry bias forward, clamped to [-3, 3]

Result (seed 1234, double-loop config on torch 2.11): n-gram only 1.08152 → ETLB + n-gram 1.08132 (-0.00020). Not re-tested on the final triple-loop fused config.

Rejected: takes 615s, doesn't fit in 600s eval budget.

</details>

### A.7 Setup & Reproduction

<details><summary><b>Full build instructions</b></summary>

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece brotli numpy

export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):${LD_LIBRARY_PATH:-}

cd /opt && git clone --depth 1 --branch v3.7.0 https://github.com/NVIDIA/cutlass
cd cutlass_evt_fusion && CUTLASS_PATH=/opt/cutlass python3 setup.py build_ext --inplace && cd ..

rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

SEED=1234 NUM_LOOPS=3 ENABLE_LOOPING_AT=0.35 PARALLEL_RESIDUAL_START=7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

</details>
