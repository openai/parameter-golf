# Session 04b: Novel Approach Research Synthesis

## Context

**Current state:** Session 03 anchor at `val_bpb=1.12904446` on 8xH100 in 600s.
**Target:** ≤1.1178 BPB to enter leaderboard (beat SOTA 1.1228 by 0.005 nats).
**Gap:** 0.0112 BPB. Two bottlenecks: export gap (0.0077 BPB) + throughput gap (~487 fewer steps, ~0.003-0.004 BPB).
**Artifact headroom:** 248,676 bytes under 16MB cap.
**Sources:** 8 parallel queries across ChatGPT (quick + deep), Claude (quick + deep), Gemini (quick + deep), Perplexity (quick + deep).

---

## Part 1: Deduplicated Master List of All Unique Ideas

### Category A: Weight Compression Beyond Int6

| # | Idea | Source(s) | Feasibility | BPB Impact | Novelty | Effort |
|---|------|-----------|:-----------:|:----------:|:-------:|--------|
| A1 | **Full Hessian GPTQ + AR self-gen calibration** — Replace diagonal-Hessian GPTQ-lite with full Hessian GPTQ; calibrate on model's own autoregressive output (186s within budget). PR #1019 validated. | Perplexity-Q | 5 | 4 | 3 | 1-2 days |
| A2 | **SpQR-style outlier sidecar** — Keep int6 base, store tiny sparse set of outlier weights in higher precision (fp16/int8) as correction. | ChatGPT-Q | 3 | 3 | 4 | 2-3 days |
| A3 | **Fusible rotation before quantization (OptRot)** — Data-free orthogonal rotation to Gaussianize weight distribution before int6 export. Fuses into adjacent linear layers → zero runtime cost. | ChatGPT-Q | 3 | 2 | 4 | 1-2 days |
| A4 | **Low-rank residual correction on worst matrices** — W ≈ Q + UV^T. Store quantized Q plus tiny high-precision low-rank correction for 2-4 most sensitive matrices only. **RFN CONNECTION.** | ChatGPT-Q, ChatGPT-D (SLiM, 3BASiL) | 3 | 3 | 4 | 2-3 days |
| A5 | **RAMP-lite layer-wise mixed precision** — Per-layer bit-width search ({5,6,7} bits) under global byte budget via greedy/coordinate-descent. | Perplexity-D, ChatGPT-D (Q-Palette) | 4 | 3 | 4 | 1-2 days |
| A6 | **AdaDim adaptive quantization axis** — Per-layer choice of IC vs OC quantization axis based on sensitivity. | Perplexity-D | 4 | 2 | 4 | 4-6 hrs |
| A7 | **CERWU rate-distortion PTQ** — Optimizes rate-distortion objective with entropy coding awareness for zstd alignment. | ChatGPT-Q | 2 | 2 | 4 | 2-3 days |
| A8 | **PCDVQ (Polar Coordinate Decoupled VQ)** — Vector quantization decoupling direction (cosine) from magnitude. Targets 2-3 bpp. | ChatGPT-D | 2 | 3 | 5 | 3-5 days |
| A9 | **GLVQ (Grouped Lattice VQ)** — Learned lattice codebooks with salience-driven bit allocation and companding. | ChatGPT-D | 2 | 3 | 5 | 4-7 days |
| A10 | **Neural Weight Compression (NWC)** — Tiny MLP autoencoder trained on weight chunks as learned codec. **RFN CONNECTION.** | Perplexity-D | 2 | 2 | 5 | 2-3 days |
| A11 | **Codebook/additive quantization (AQLM/LCQ/SKIM)** — Selective codebook methods on one matrix family. Negative signal from repo. | ChatGPT-Q, Perplexity-D | 2 | 2 | 3 | 2-3 days |
| A12 | **QuZO zeroth-order fine-tuning** — Gradient-free updates on quantized weights via SPSA. | Perplexity-D | 2 | 2 | 5 | 1-2 days |
| A13 | **Zero-byte backbone (random linear maps + LoRA)** — Hardcoded PRNG seed for base matrices, train only low-rank adapters. | Gemini-Q | 1 | 4 | 5 | 3-5 days |
| A14 | **Freeze EMA during warmdown** — Disable EMA updates in final 10% of training to anchor weights on int6 lattice. | Gemini-D | 5 | 2 | 3 | 1-2 hrs |
| A15 | **Sparse + low-rank decomposition (SLiM / 3BASiL / GQSA)** — Combined structured sparsity + quantization + low-rank residual. **RFN CONNECTION.** | ChatGPT-D | 2 | 3 | 4 | 3-7 days |

### Category B: Eval-Time Techniques Beyond Sliding Window

| # | Idea | Source(s) | Feasibility | BPB Impact | Novelty | Effort |
|---|------|-----------|:-----------:|:----------:|:-------:|--------|
| B1 | **N-gram cache (multi-order backoff + entropy-adaptive alpha)** — Build hash table from already-scored tokens, mix n-gram predictions with neural model. Orders 2-7, cascading backoff. Zero artifact cost. PRs #702, #727, #1026. | Claude-Q, Claude-D, Perplexity-Q, Gemini-D | 5 | **5** | 2 | 1 day |
| B2 | **HedgeMixer (online expert weighting)** — Replace heuristic alpha with Hedge algorithm multiplicative weight updates over neural + n-gram experts. PR #1014/#995. | Claude-D, Gemini-D | 4 | 4 | 3 | 1-2 days |
| B3 | **Context Tree Weighting (CTW)** — Bayesian-optimal variable-order Markov weighting. Provably minimax optimal. PR #1011 pending. | Claude-D | 3 | 3 | 4 | 2-3 days |
| B4 | **Score-first TTT (cosine schedule, per-layer LR)** — Legal backward-looking TTT: score then record then backward then next window. Cosine LR decay, 3x for high-quant-error layers. PR #549 at 1.1194. | Claude-Q, Claude-D, Gemini-D | 5 | 3 | 1 | 1-2 days |
| B5 | **KV cache persistence across sliding windows** — Cache K/V from scored positions, extend effective context from 2K to 50K+ tokens. PR #831. | Claude-Q, Claude-D | 3 | 3 | 4 | 1-2 days |
| B6 | **qTTT (query-only TTT)** — Cache K/V once, adapt only Q projection weights. 2-3x more TTT epochs within eval budget. | Claude-D | 3 | 2 | 4 | 1-2 days |
| B7 | **Hidden-state kNN-LM** — Nearest-neighbor search over hidden states from processed tokens. +0.007 BPB on top of n-gram cache. PR #738. | Claude-D | 3 | 2 | 4 | 1-2 days |
| B8 | **Post-TTT temperature calibration (T=0.98)** — Counter TTT-induced overconfidence. PR #576. | Claude-D | 5 | 1 | 2 | 1 hr |
| B9 | **Logit-space n-gram mixing** — Mix in log-probability space instead of probability space. | Claude-D | 5 | 1 | 3 | 2 hrs |
| B10 | **Adaptive stride evaluation** — Smaller stride where model is uncertain, larger where confident. | Claude-Q, Claude-D | 3 | 1 | 5 | 4-8 hrs |
| B11 | **Document-boundary-aware windowing** — Detect document separators, start fresh windows at boundaries. | Claude-Q | 4 | 1 | 4 | 4-8 hrs |
| B12 | **Longer eval sequence (4096+)** — Evaluate at 4096 with NTK-aware RoPE scaling. | Claude-Q | 4 | 1 | 2 | 2-4 hrs |
| B13 | **Multi-checkpoint ensemble** — Average predictions from 2-3 checkpoints. | Claude-Q | 3 | 1 | 2 | 4-8 hrs |

### Category C: Architectural Innovations

| # | Idea | Source(s) | Feasibility | BPB Impact | Novelty | Effort |
|---|------|-----------|:-----------:|:----------:|:-------:|--------|
| C1 | **MTP (Multi-Token Prediction) auxiliary loss** — Predict 2 tokens ahead during training. Heads discarded at export, so zero artifact/eval cost. PR #1031 validated -0.0037 BPB. | Perplexity-Q | 4 | 3 | 3 | 4-8 hrs |
| C2 | **VRL (Value Residual Learning)** — Blend layer 0's V output into all subsequent layers via learned sigmoid gates. PR #1019/#1016. | Perplexity-Q | 4 | 2 | 3 | 4-8 hrs |
| C3 | **ASQU (Asymmetric Squared Unit)** — Per-channel learned negative-branch slope instead of fixed LeakyReLU. PR #1035. | Perplexity-Q | 5 | 2 | 3 | 2-4 hrs |
| C4 | **JEPA (Joint-Embedding Predictive Architecture)** — Latent-space masked prediction + EMA self-distillation. PR #1006 at **1.1085 BPB** (!). | Gemini-D | 2 | 5 | 5 | 3-5 days |
| C5 | **DG Attention (Differential-Gated)** — Differential codec: transmit delta relative to causal baseline instead of raw content. Depth-scheduled beta. PR #542. | Gemini-D | 3 | 2 | 4 | 1-2 days |
| C6 | **Chimera Topology / blockwise weight sharing** — 3 unique early layers + 4 shared layers looped 2x with U-Net skips. Reclaims ~6MB of params, reinvest in width. | Gemini-Q, Gemini-D, Perplexity-D (MobileLLM) | 2 | 3 | 4 | 2-4 days |
| C7 | **SSM/Mamba hybrid** — Replace early layers with Mamba-2 / S4D-Lin blocks, keep XSA on top layers. | Gemini-Q, Gemini-D, Perplexity-D (Hymba) | 1 | 3 | 4 | 5-10 days |
| C8 | **H-Net / learned tokenization** — Byte-level input with dynamic chunking CNN. Eliminates embedding table. PR #992. | Gemini-Q, Gemini-D | 1 | 4 | 5 | 5-10 days |
| C9 | **Micro-scale MoE** — Fracture MLP into 2 experts with top-1 routing. PR #181, #981. | Gemini-D | 2 | 2 | 3 | 2-3 days |
| C10 | **SwiGLU activation** — Replace relu-squared with SwiGLU. Standard in modern SLMs. | Perplexity-D | 4 | 1 | 2 | 2-4 hrs |

### Category D: Training Efficiency

| # | Idea | Source(s) | Feasibility | BPB Impact | Novelty | Effort |
|---|------|-----------|:-----------:|:----------:|:-------:|--------|
| D1 | **FlashAttention-3** — Drop-in replacement for SDPA. 1.5-2x speedup, ~75% H100 utilization. Closes 6ms/step gap, recovers ~487 more steps. | Perplexity-D, Gemini-Q, Gemini-D | 3 | 3 | 3 | 2-4 hrs (if wheels exist) to 2-3 days |
| D2 | **Megakernel fusion** — Fuse attention + LN + MLP into single persistent kernel. ThunderKittens-style. | Gemini-Q, Gemini-D, Perplexity-D | 1 | 3 | 5 | 5-10 days |
| D3 | **Curriculum learning warm-up** — Easy-to-hard data ordering in first few hundred steps. | Perplexity-D | 3 | 2 | 4 | 4-8 hrs |
| D4 | **WSD (Warmup-Stable-Decay) LR schedule** — MiniCPM-style stable middle phase. | Perplexity-D | 4 | 1 | 3 | 2-4 hrs |
| D5 | **Data composition optimization** — Pre-cluster FineWeb, oversample high-quality subsets early. | Perplexity-D | 2 | 2 | 4 | 1-2 days |

---

## Part 2: Multi-Model Convergence Signals

Ideas independently proposed by multiple models carry high signal. Sorted by convergence count:

| Convergence | Idea | Models | Signal Strength |
|:-----------:|------|--------|:---------------:|
| **4+** | N-gram cache (B1) | Claude-Q, Claude-D, Perplexity-Q, Gemini-D | VERY HIGH |
| **3** | Score-first TTT (B4) | Claude-Q, Claude-D, Gemini-D | Already on LB |
| **3** | SSM/Mamba hybrid (C7) | Gemini-Q, Gemini-D, Perplexity-D | High but infeasible |
| **3** | Weight sharing / Universal Transformer (C6) | Gemini-Q, Gemini-D, Perplexity-D | High |
| **3** | Megakernel / FA-3 throughput (D1/D2) | Gemini-Q, Gemini-D, Perplexity-D | High |
| **2** | Mixed-precision bit allocation (A5) | ChatGPT-D (Q-Palette), Perplexity-D (RAMP) | High |
| **2** | Fusible rotation (A3) | ChatGPT-Q, related in ChatGPT-D | Medium |
| **2** | Low-rank residual correction (A4) | ChatGPT-Q, ChatGPT-D | Medium-High |
| **2** | KV cache persistence (B5) | Claude-Q, Claude-D | Medium |
| **2** | AdaDim / axis selection (A6) | Perplexity-D, implicit in ChatGPT-D | Medium |
| **2** | H-Net tokenization (C8) | Gemini-Q, Gemini-D | High but infeasible |

**Key interpretation:** The n-gram cache convergence signal is extraordinary — 4+ independent AI models across different architectures all flagged it as the #1 lever, and it's backed by real competition PRs. This is the single strongest signal in the entire dataset.

---

## Part 3: RFN Thesis Connections

The following ideas connect directly to Residual Factorization Networks thesis work:

| Idea | RFN Connection | Thesis Leverage |
|------|---------------|-----------------|
| **A4: Low-rank residual correction** (W = Q + UV^T) | **DIRECT.** This IS residual factorization applied to quantization error recovery. | RFN implementation could directly serve as the low-rank correction mechanism. |
| **A10: Neural Weight Compression** | Related. NWC learns a codec over weight structure — RFN's factorization insights could inform the codec architecture. | Moderate leverage. |
| **A15: SLiM / 3BASiL** (sparse + low-rank) | **DIRECT.** W = S + LR decomposition is a sparse variant of residual factorization. | Factorization analysis could identify which layers benefit most. |
| **A13: Zero-byte backbone** (random base + LoRA) | Tangentially related. The adapter-on-random-base pattern shares structure with RFN's factored representation. | Low but interesting. |

**Recommendation:** If thesis alignment is desired, A4 (low-rank residual on worst matrices) is the most natural integration point. It would demonstrate that RFN-style factorization can recover quantization damage in compressed models — a publishable result.

---

## Part 4: Ideas That Sound Good But Fail Constraints

| Idea | Why It Fails |
|------|-------------|
| **C7: SSM/Mamba hybrid** | Requires custom Triton kernels. torch.compile cannot optimally fuse selective scan. Step time regression to 120ms+ would negate all gains. Pegasus environment may lack Mamba wheels. |
| **C8: H-Net learned tokenization** | Dynamic tensor lengths break torch.compile(fullgraph=True). Forces eager mode, causing throughput collapse. 10-min training insufficient for tokenizer convergence. |
| **D2: Full megakernel** | 5-10 days of Triton development for a single-use kernel. Not feasible in Sessions 05-06 timescale. |
| **A13: Zero-byte backbone** | Fundamentally changes training dynamics. Random base matrices with LoRA have never been validated at this model scale for BPB optimization. Very high risk of convergence failure. Challenge rules likely consider the PRNG code as part of the artifact. |
| **A8/A9: PCDVQ/GLVQ** | Designed for 2-3 bpp regime (extreme compression). At ~4.7 effective bpp, these are over-engineered. Custom packing/decoding adds engineering risk for marginal benefit. |
| **C4: JEPA** | PR #1006 shows 1.1085 BPB — incredible. But it's a fundamental architecture rewrite (latent-space loss, EMA teacher). 3-5 days minimum, incompatible with current eval pipeline. Best saved for a dedicated investigation if time budget allows 1+ week. |

---

## Part 5: Ranked Top 5 by Risk-Adjusted Expected Value

**Scoring:** `RAEV = P(success) x Expected_BPB_gain / sqrt(effort_hours)`

### #1: N-gram Cache with Entropy-Adaptive alpha (B1 + B2)

| Metric | Value |
|--------|-------|
| **Expected BPB gain** | -0.030 to -0.070 (conservative); up to -0.16 (aggressive) |
| **Feasibility** | 5/5 — ~100 lines in eval loop. Zero training changes. |
| **Novelty vs LB** | 2/5 — Known in PRs but NOT yet merged/approved. |
| **Effort** | ~8-16 hours |
| **Risk** | **Legality.** Organizer ruling pending. "Directionally legal" per organizer comment. |
| **RAEV** | **Extremely high.** Even at -0.030, this closes the gap 3x over. |

**Why #1:** Four independent AI models converged on this. Real PRs show it working (PR #1026: 1.1264 to 1.0945). Zero artifact cost. Stacks with everything else. If legal, it makes the leaderboard question trivial.

### #2: Full Hessian GPTQ + AR Self-Generated Calibration (A1)

| Metric | Value |
|--------|-------|
| **Expected BPB gain** | -0.004 to -0.006 (directly attacks 0.0077 export gap) |
| **Feasibility** | 5/5 — Validated in PR #1019. |
| **Novelty vs LB** | 3/5 — Upgrade over current GPTQ-lite Delta 1. |
| **Effort** | ~12-24 hours |
| **Risk** | Low. AR self-gen takes ~186s within training budget. |
| **RAEV** | **Very high.** Directly addresses the #1 bottleneck. |

**Why #2:** The export gap is 0.0077 BPB — 69% of the total 0.0112 gap. PR #1019 demonstrated reducing post-quant gap from ~0.0078 to ~0.0023 BPB. This is the most mechanistically targeted improvement available.

### #3: MTP Auxiliary Loss (C1) + ASQU Activation (C3) + VRL (C2)

| Metric | Value |
|--------|-------|
| **Expected BPB gain** | -0.004 + -0.003 + -0.002 = ~-0.006 to -0.009 (stacked) |
| **Feasibility** | 4-5/5 — Each validated in separate PRs. |
| **Novelty vs LB** | 3/5 — All from recent unmerged PRs. |
| **Effort** | ~12-20 hours total for all three |
| **Risk** | Low individually. Interaction effects unknown when stacked. |
| **RAEV** | **High.** Three orthogonal, low-risk, validated improvements. |

**Why #3:** These are the "free lunch" stack — MTP improves training signal at zero eval cost, ASQU improves activation at near-zero cost, VRL adds representational power at near-zero cost. Each is validated by a separate competition PR. Combined they could match or exceed the export gap fix.

### #4: Score-First TTT with Cosine Schedule + Per-Layer LR (B4)

| Metric | Value |
|--------|-------|
| **Expected BPB gain** | -0.003 to -0.006 on frontier bases; up to -0.03 on weaker bases |
| **Feasibility** | 5/5 — Already merged at #1 on leaderboard. |
| **Novelty vs LB** | 1/5 — Already the #1 technique. |
| **Effort** | ~16-32 hours |
| **Risk** | Low. Well-understood. Requires ~410s of eval budget. |
| **RAEV** | **High.** Proven technique. Mandatory for leaderboard competitiveness. |

**Why #4:** Not novel, but implementing this is required for competitiveness. The current #1 uses it. Combining TTT with the anchor + the export gap fix could reach ~1.120-1.123 BPB without any novel techniques.

### #5: Layer-Wise Mixed Precision (RAMP-lite) + Freeze EMA During Warmdown (A5 + A14)

| Metric | Value |
|--------|-------|
| **Expected BPB gain** | -0.003 to -0.006 (RAMP) + -0.001 to -0.002 (EMA freeze) |
| **Feasibility** | 4-5/5 — RAMP-lite is a greedy search; EMA freeze is trivial. |
| **Novelty vs LB** | 4/5 — No per-layer bit allocation in the repo. |
| **Effort** | ~12-20 hours total |
| **Risk** | Medium. RAMP requires calibration loop. EMA freeze is risk-free. |
| **RAEV** | **Medium-High.** Novel compression approach with solid theoretical grounding. |

**Why #5:** Two convergent models (ChatGPT-D, Perplexity-D) identified layer-wise mixed precision as high-value. Combined with the trivial EMA freeze (which prevents continuous-state drift during warmdown), this is another strong attack on the export gap. More novel than Full GPTQ (#2), so potentially complementary.

---

## Part 6: Concrete Sketches for Top 2

### Sketch 1: N-gram Cache in eval_val_sliding() (lines 863-943 of anchor)

```
CHANGES TO: records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py

Location: Inside eval_val_sliding() function, lines 863-943.

New data structure (add before eval loop):
  - ngram_cache: dict[int, Counter[int]]  # hash(context) -> next-token counts
  - Orders: 2 through 7 (try 7-gram first, cascade down)
  - Hash function: rolling hash using same XOR pattern as BigramHash

Algorithm (inside the window loop, after scoring):
  1. SCORE PHASE (existing, unchanged):
     - Run compiled_logits(batch) -> logits
     - Compute per-token NLL, accumulate into loss/byte counters

  2. CACHE UPDATE PHASE (new, after scoring):
     - For each scored token position t in the non-overlap zone:
       - For each order n in [2..7]:
         - key = rolling_hash(tokens[t-n+1 : t])
         - ngram_cache[key][tokens[t]] += 1

  3. CACHE LOOKUP PHASE (new, before softmax):
     - For each token position t:
       - For n in [7, 6, 5, 4, 3, 2]:  # highest order first
         - key = rolling_hash(tokens[t-n+1 : t])
         - if key in ngram_cache and sum(ngram_cache[key].values()) > threshold:
           - p_ngram = normalize(ngram_cache[key])
           - break
       - Compute entropy H of neural model's softmax
       - alpha = base_alpha * clamp(H / H_threshold, 0.1, 2.0)
       - p_final = (1 - alpha) * softmax(logits) + alpha * p_ngram
       - nll = -log(p_final[target_token])

Key implementation details:
  - Cache runs on CPU (Python dict), scoring on GPU -> minimal interference
  - ~100 lines of new code, all in eval path
  - No changes to training, model architecture, or artifact
  - base_alpha ~ 0.2, H_threshold ~ 3.0 (tune on a subset)
  - Backward-looking only: cache updated AFTER scoring each window
```

### Sketch 2: Full Hessian GPTQ in mixed_quantize_int6() (lines 374-398 of anchor)

```
CHANGES TO: records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py

Replace quantize_int6_per_row() (lines 351-361) with full GPTQ:

New functions needed:
  1. generate_calibration_data(model, tokenizer, n_samples=64, seq_len=2048):
     - Autoregressive generation from random prompts
     - Returns list of (input_ids, ) tensors
     - Budget: ~60-90s within training wallclock

  2. compute_hessian(layer, calibration_inputs):
     - Hook-based: register forward hook on each linear layer
     - Accumulate H = sum(x_i @ x_i^T) over calibration samples
     - Store per-column Hessian diagonal + off-diagonal

  3. gptq_quantize_int6(weight, H, clip_range=31):
     - Column-by-column quantization with error compensation:
       for col in range(weight.shape[1]):
         q_col = quantize(weight[:, col], scale)
         error = weight[:, col] - dequantize(q_col, scale)
         # Compensate: distribute error to remaining columns
         weight[:, col+1:] -= error @ H[col, col+1:] / H[col+1:, col+1:].diag()
     - Per-row percentile clip search (reuse Delta 1 candidates)

  4. AR self-generated calibration (key innovation from PR #1019):
     - After training completes, before export:
       model.eval()
       calib_data = generate_calibration_data(model, ...)  # ~186s
       for layer_name, layer in model.named_modules():
         if isinstance(layer, CastedLinear) and is_int6_target(layer_name):
           H = compute_hessian(layer, calib_data)
           layer.weight.data = gptq_quantize_int6(layer.weight, H)

Integration points:
  - Insert calibration + GPTQ between EMA application (line 1321) and
    mixed_quantize_int6() call (line 1339)
  - Replace simple int6 row-max with GPTQ-aware quantization
  - Total added wallclock: ~186s for AR gen + ~30s for Hessian + quant
  - Must fit within 600s total: training ~400s + calibration ~186s + export ~14s
  - This means training budget shrinks by ~186s -> ~460 fewer steps
  - Net trade: lose ~0.002 BPB from fewer steps, gain ~0.005 BPB from better quant
  - **Net positive: ~+0.003 BPB improvement**

Alternative: Run calibration in parallel with late training steps using
a background thread (risky but could avoid step loss).
```

---

## Part 7: Recommended Session Plan

### Session 05: "Low-Hanging Fruit Stack" (1-2 days)

**Goal:** Stack validated, low-risk improvements on Session 03 anchor.

1. **ASQU activation** (C3) — swap relu-squared to ASQU. 2-4 hours. Validated by PR #1035.
2. **MTP auxiliary loss** (C1) — add 2-token prediction head. 4-8 hours. Validated by PR #1031.
3. **Freeze EMA during warmdown** (A14) — trivial 1-2 hour change.
4. **Full Hessian GPTQ** (A1) — upgrade export pipeline. 12-24 hours.

**Expected cumulative gain:** -0.008 to -0.014 BPB, reaching ~1.115-1.121 BPB.

### Session 06: "Eval-Time Revolution" (1-2 days)

**Goal:** Add eval-time techniques that exploit the separate eval budget.

1. **N-gram cache** (B1) — implement and validate. 8-16 hours.
2. **Score-first TTT** (B4) — implement with cosine schedule. 16-32 hours.
3. **Temperature calibration** (B8) — trivial addition post-TTT.

**Expected cumulative gain:** If n-gram cache is legal: -0.03 to -0.07 additional BPB, reaching ~1.05-1.09 BPB.
If n-gram cache is illegal: TTT alone gives -0.003 to -0.006, reaching ~1.109-1.118 BPB.

### Moonshot (if time permits): RFN-Aligned Low-Rank Residual (A4)

After Sessions 05-06, if the gap remains:
- Implement W = Q + UV^T for the 2-4 most quantization-sensitive matrices
- This directly connects to the bachelor thesis on Residual Factorization Networks
- Estimated: -0.002 to -0.005 additional BPB
- Publishable result: "RFN-style factorization recovers quantization damage in compressed LMs"

---

## Part 8: Verification Plan

For each implemented technique:
1. **Isolated measurement:** Run on 8xH100 with ONLY that change vs Session 03 anchor
2. **Metrics:** Pre-quant EMA val_bpb, post-roundtrip val_bpb, sliding s64 val_bpb, step_avg, artifact size
3. **Attribution:** Each change gets its own `records/track_non_record_16mb/` folder
4. **Decision gate:** Only graduate to stacking if isolated improvement > 0.001 BPB
5. **For n-gram cache specifically:** Check issue #402 and related threads for latest organizer ruling before investing time
