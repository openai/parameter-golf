# SOTA-13 Research Summary
> Research date: April 7, 2026 | Current best: ~1.109 BPB | Target: <1.100 BPB

---

## 1. Modded-NanoGPT Speedrun — Technique Inventory

The [KellerJordan speedrun](https://github.com/KellerJordan/modded-nanogpt) tracks every technique that improved the H100 8-GPU record. As of the latest README (78 records, current = 2.812 min):

### A. Techniques we already have in sota_12
| Technique | Record | Status |
|-----------|--------|--------|
| ReLU², QK-Norm, zero-init projections | #5 | ✅ |
| U-Net pattern skip connections | #11, #15 | ✅ |
| Value embeddings (VE) | #14, #15 | ✅ |
| Sliding window attention | #16 | ✅ |
| FP8 head | #19 | ✅ (lm_head) |
| Long-short attention + YaRN | #20, #31 | ✅ (XSA) |
| Multi-token prediction (MTP) | #53 | ✅ |
| Bigram Hash Embedding | #62, #68 | ✅ |
| Value embed gating | #55 | ✅ |
| SWA / EMA | various | ✅ |
| GQA | internal | ✅ |

### B. Techniques **NOT yet in sota_12** (candidates for sota_13)
| Technique | Record | Estimated BPB gain | Complexity |
|-----------|--------|--------------------|-----------|
| **Smear module** (1-token look-back embed) | #34, 2.547→2.527 | -0.003 | Low |
| **Sparse attention gate** | #28, 2.812 | -0.002 | Medium |
| **NorMuon** (normalized gradient Muon) | #41, 2.358→2.345 | -0.002 | Low |
| **Cautious Weight Decay w/ schedule** | #43, 2.313→2.284 | -0.002 | Low |
| **Backout** (residual skip from 2/3 pt) | #40, 2.447→2.358 | -0.004 | Medium |
| **Partial Key Offset** | #49, 2.146→2.128 | -0.001 | Low |
| **Paired Head Attention** | #58, 1.878→1.820 | -0.005 | Medium |
| **Batch size schedule** | #46, 2.203→2.193 | -0.001 | Low |
| **Max seq_len schedule** | #72 | -0.001 | Low |
| **Exponential decay of residual stream** | #45 | -0.001 | Low |
| **Polar Express** (Newton-Schulz replacement) | #38, 2.483→2.476 | -0.001 | Medium |

---

## 2. Quantization Literature

### 2.1 AWQ — Activation-Aware Weight Quantization (MLSys 2024 Best Paper)
> *Lin et al., 2023 — arXiv:2306.00978*

**Core idea:** Not all weights are equally important. 1% of "salient" channels (identified by activation magnitude, not weight magnitude) dominate quantization error. Instead of mixed-precision quantization, AWQ **scales up salient channels** by a factor `s` before quantization, then folds `s` into the adjacent layer (zero inference cost).

$$s = \left(\frac{\|\mathbf{x}_j\|_\text{max}}{\max_k \|\mathbf{x}_k\|_\text{max}}\right)^{0.5}$$

**Applicability:** We can add an AWQ pre-scaling step before our GPTQ. Collect activation statistics on the calibration data → compute per-output-channel scale → scale the weight rows → run GPTQ on scaled weight → bake scale into next layer bias or `lm_head`. **Zero inference overhead, potentially -0.003 to -0.005 BPB better GPTQ.**

### 2.2 QuIP# — Hadamard Incoherence + E8 Lattice Codebooks (ICML 2024)
> *Tseng et al., 2024 — arXiv:2402.04396*

**Core idea:**
1. **Hadamard pre-rotation**: Apply random Hadamard transform `H·W·H^T` before quantizing. This makes the weight distribution more Gaussian (incoherence property), reducing extreme outliers that hurt quantization.
2. **E8 lattice codebooks**: Use vector quantization in 8-dimensional space using the densest-known 8D sphere packing (E8 lattice) instead of scalar quantization.
3. **Fine-tuning phase**: After quantization, fine-tune a small number of steps to recover fidelity.

**Applicability:** The Hadamard rotation (step 1) is the most implementable here. A fast Walsh-Hadamard Transform (WHT) can be computed in O(n log n). It's a zero-cost equivalent transformation — multiply weights by `H`, activations by `H^T`. Our GPTQ calibration process then runs on the rotated weight matrix, which has better Gaussian properties. **High expected gain: -0.004 to -0.008 BPB.**

### 2.3 LLM-FP4 — Per-Channel Floating-Point Quantization (EMNLP 2023)
> *Liu et al., 2023 — arXiv:2310.16836*

**Core idea:** FP4 format handles long-tail distributions better than INT4. Per-channel activation quantization, where scaling factors are folded into weights as exponential biases.

**Applicability:** Less directly applicable (we use int6). But the *per-channel activation scaling* insight is useful for our AWQ implementation.

### 2.4 QuIP — Incoherence Processing (NeurIPS 2023)
> *Chee et al., 2023 — arXiv:2307.13304*  

**Original QuIP before QuIP#:** Applied random unitary transforms to weights before quantization. QuIP# replaced the random unitary with a fast Hadamard transform for practical speed.

---

## 3. Architecture and Training Literature

### 3.1 nGPT — Normalized Transformer on Hypersphere (arXiv:2410.01131)
> *Loshchilov et al., 2024*

**Core idea:** All vectors (embeddings, MLP projections, attention matrices, hidden states) are unit-norm normalized. The residual stream travels on a hypersphere, with each layer adding a displacement toward the target. Claims **4-20× faster training** (fewer steps to same loss).

**Why not for sota_13 directly:** Requires full architecture rewrite. The gains are in sample efficiency (training steps), not inference quality or model compression. Since our bottleneck is inference quality after quantization, nGPT doesn't directly help.

### 3.2 TTT-Linear / TTT-MLP — Test-Time Training Layers (arXiv:2407.04620)
> *Sun et al., 2024 — "Learning to (Learn at Test Time)"*

**Core idea:** The hidden state of a sequence model is itself a machine learning model. At test time, the model's hidden state (a linear model or 2-layer MLP) is updated via self-supervised learning on the incoming context. TTT-Linear has linear complexity and TTT-MLP is quadratically expressive.

**Our TTT (sota_12):** We do document-level TTT — update Adam parameters based on early tokens of each document before scoring the rest. This is a different (and legal) form. The TTT-MLP paper suggests the hidden state approach, but integrating this architecturally would be a major change.

**Improvement for sota_13:** Can we improve our existing TTT by:
1. Adding **MTP loss during TTT** (not just next-token CE)  
2. Using **Muon instead of SGD for TTT** (Muon should be better for matrix params)
3. Increasing **TTT epochs from 3 to 5** for better adaptation

### 3.3 Scaling Laws — SWA + Constant-LR + Cooldown (NeurIPS 2024 Spotlight)
> *Hägele et al., 2024 — arXiv:2405.18392*

**Core idea:** Constant LR with cooldown is equivalent to cosine schedule for scaling laws. **Stochastic Weight Averaging (SWA) provides free performance improvement** along the training trajectory.

**Already in sota_12:** `SWA_ENABLED=1, SWA_EVERY=50`. We should verify this is working optimally.

---

## 4. From Competition PRs (Our Track)

### 4.1 Current Competition Landscape
| PR | BPB | Key Idea | Notes |
|----|-----|----------|-------|
| #1204 | 1.1063 | Parallel Residuals + Mini Depth Recurrence | Our base |
| #1179 | ~1.1105 | Mixed quant, XSA, coprime loader | Pre-#1204 |
| #1176 | 1.0914 | QK-Gain 4.0 + Muon-TTT + SLOT | SLOT legality disputed |
| #1184 | 0.9485 | Scylla tokenizer + Full GPTQ + FA3 | Byte bug — invalid |

**External best (uncontested):** 1.1063 (PR #1204). We need to get below 1.10.

### 4.2 KellerJordan Smear Module (Record #34)
**Smear = "smear token embeddings 1 position forward"**
- Each token's embedding is a mix of its own embedding and the previous token's.
- Implementation: `embed_smeared[i] = embed[i] + alpha * embed[i-1]`; alpha is a learned scalar.
- This gives the model an implicit 1-token look-back at the embedding level (before attention).
- **Benefit:** Lookahead features for free, essentially a learned "context bleed."
- **Our cost:** ~1 scalar parameter. Very cheap.

### 4.3 Backout Skip Connection (Record #40, -0.089 min)
**"Skip from 2/3 of training point to pre-lm_head"**
- A direct residual connection from the hidden state at layer `⌊2L/3⌋` to the pre-lm_head transform.
- This is an architectural modification, adding an extra skip pathway.
- Different from U-Net (which skips between encoder/decoder layers).

### 4.4 NorMuon (Record #41)
- Modified Muon where the gradient normalization step also normalizes across layers.
- "Spectral norm" of the gradient matrix → normalized update direction.

### 4.5 Cautious Weight Decay (Record #43 + #50)
- Weight decay is applied cautiously: only if the weight change is in the same direction as the gradient signal.
- Implemented as a mask: `mask = (update * p.grad > 0)`, apply WD only where `mask=True`.
- **Free improvement:** Just a modification to the optimizer step.

---

## 5. Synthesis: SOTA-13 Plan

### Priority Ranking for Implementation

#### 🔴 Tier 1 — Maximum Expected Impact (implement all)

**1. AWQ-style Pre-Scaling + Hadamard Rotation before GPTQ**
- **Impact:** -0.004 to -0.010 BPB better GPTQ quality
- **How:** Before GPTQ, collect activation L2 norms per channel → compute AWQ scale → scale weight rows → apply WHT-based Hadamard rotation to both weight and calibration data → run GPTQ → fold scale back
- **Code location:** Modify `gptq_quantize_linear()` to add pre-rotation and channel scaling
- **Param cost:** 0 parameters at inference (equivalent transform)

**2. recur_passes=2 at inference**
- **Impact:** -0.002 to -0.004 BPB (2× depth for recurrence layers)
- **How:** `RECUR_PASSES=2` already parameterized. Just set it.
- **Code:** Already in sota_12. Zero new code.

**3. Smear Module (1-token look-back)**  
- **Impact:** -0.002 to -0.003 BPB (from speedrun record)
- **How:** learned scalar `smear_alpha`; `embed_out = embed + smear_alpha * F.pad(embed, (0,0,1,0))[:-1]`
- **Code location:** Add in `Embedding` forward before positional encoding
- **Param cost:** 1 scalar (negligible)

#### 🟡 Tier 2 — Good Impact, Moderate Risk

**4. Cautious Weight Decay (CWD)**
- **Impact:** -0.001 to -0.002 BPB
- **How:** Modify Muon and Adam to only apply WD where gradient and momentum agree
- **Code:** Optimizer step modification (~10 LOC)

**5. Per-Layer Quantization Budget (int8 for boundary layers)**
- **Impact:** -0.002 to -0.004 BPB (layer 0 and layer 10 stay int8, use int6 for middle)
- **How:** Pass `bits=8` for layers 0 and 10 in GPTQ loop
- **Code:** Minor change to GPTQ dispatch

**6. TTT with MTP Loss**
- **Impact:** -0.001 to -0.002 BPB
- **How:** During TTT chunk training, compute MTP prediction loss (predict t+1, t+2) in addition to t+1
- **Code:** Add MTP head to TTT loss computation

**7. 4-gram Hash Embedding**
- **Impact:** -0.001 to -0.002 BPB (extends trigram → 4-gram)
- **How:** same hash trick as bigram/trigram; `h4 = hash(token[i-3], token[i-2], token[i-1], token[i])`
- **Param cost:** ~0 (reuses same embedding table)

#### 🟢 Tier 3 — Worthwhile but Lower Priority

**8. Batch Size Schedule** (small→large over training)
- From speedrun record #46. Use smaller batches early (better gradient noise), larger batches later.

**9. max_seq_len Schedule** (ramp up context window over training)
- From speedrun record #72. Train on short sequences early, extend later (similar to attention window warmup we already have).

**10. Paired Head Attention** (speedrun record #58)
- Two adjacent attention heads share Q/K matrices but have separate V/O. Reduces Q/K parameter budget, allows more heads.

#### ⚪ Tier 4 — Ambitious / Architecture Overhaul

**11. SLOT (Sequence-Level Optimization at Test Time)**
- **What:** PR #1176 reports 1.0914 BPB but is under review for causality violation.
- **Wait** for official review decision before implementing.

**12. Full TTT-MLP hidden state**
- Architecturally integrate TTT-MLP as an attention layer replacement.
- Very high implementation cost; unclear benefit within competition rules.

---

## 6. SOTA-13 Implementation Plan (Concrete Changes vs sota_12)

### New features to add:
1. **Hadamard pre-rotation + AWQ channel scaling in GPTQ** — modify `gptq_quantize_linear()`
2. **Smear module** — add `smear_alpha` to `Embedding`, apply on forward pass
3. **Cautious Weight Decay** — modify `Muon.step()` and Adam WD step
4. **Per-layer bits in GPTQ** — `gptq_bits_by_layer` dict, int8 for layers 0 and 10 (lm_head uses FP8 already)
5. **TTT MTP loss** — add 2nd-token prediction to `run_legal_ttt()` 
6. **4-gram hash embedding** — extend `BigramHashEmbed` to `NGramHashEmbed` with n=2,3,4

### Hyperparameter changes:
- `recur_passes`: 1 → 2 (in runner)
- `ttt_epochs`: 3 → 5 (more adaptation)
- `ttt_chunk_size`: 32768 → 16384 (smaller chunks = more TTT steps = better adaptation per doc)
- `warmdown_iters`: 5500 → 6000 (more cooldown budget for final quantization)
- `gptq_calib_batches`: 256 → 512 (better Hessian estimation)

### Expected total gain:
- AWQ + Hadamard GPTQ: -0.004 to -0.008 BPB
- recur_passes=2: -0.002 to -0.004 BPB  
- Smear module: -0.002 to -0.003 BPB
- Cautious WD: -0.001 to -0.002 BPB
- Per-layer bits: -0.001 to -0.002 BPB
- TTT MTP: -0.001 BPB
- **Total (optimistic):** -0.011 to -0.020 BPB → from 1.109 → **~1.089-1.098 BPB** 🎯

---

## 7. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Hadamard rotation may destabilize STE QAT training | Apply rotation only at GPTQ time (post-training), not during training |
| AWQ scale mismatch if trigram/bigram embeds are not scaled | Skip embed layers in AWQ (they're int8 fp16 already) |
| recur_passes=2 may cause compilation retracing | No — `recur_passes` is a static value at compile time |
| smear_alpha gradient may interact badly with SWA | SWA averages over smear_alpha too — fine |
| TTT MTP loss may overfit to calibration distribution | Keep TTT epochs low (≤5) and chunk size ≤16K |

---

## 8. References

| Source | Paper / URL |
|--------|-------------|
| Modded-NanoGPT README | https://github.com/KellerJordan/modded-nanogpt |
| AWQ | arXiv:2306.00978 (MLSys 2024 Best Paper) |
| QuIP# | arXiv:2402.04396 (ICML 2024) |
| LLM-FP4 | arXiv:2310.16836 (EMNLP 2023) |
| TTT-MLP | arXiv:2407.04620 |
| nGPT | arXiv:2410.01131 |
| SWA Scaling Laws | arXiv:2405.18392 (NeurIPS 2024 Spotlight) |
| Old Optimizer, New Norm (Muon origins) | arXiv:2409.20325 |
| Competition PR #1204 | Parallel Residuals + Mini Depth Recurrence |
| Competition PR #1176 | Muon-TTT + SLOT (under review) |
