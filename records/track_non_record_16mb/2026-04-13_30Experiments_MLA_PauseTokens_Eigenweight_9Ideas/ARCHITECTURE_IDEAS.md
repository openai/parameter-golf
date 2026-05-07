# Creative Architecture Ideas for Parameter Golf
## Synthesized from codebase analysis + literature review

---

## What the SOTA Already Uses (1.0810 BPB)
- SP8192 vocabulary + GPTQ int6 embedding quantization
- 3-layer depth recurrence (layers 3,4,5 looped, activated mid-training)
- Parallel residuals (GPT-J style, layers 7+)
- QK-Gain 5.25 (learnable per-head query scaling)
- Legal score-first test-time training at eval
- MuonEq-R optimizer, SWA, SD-Clip quantization
- U-Net encoder-decoder skip connections

---

## Tier 1: Highest-Impact Novel Ideas (Build These)

### 1. Ternary Weights (BitNet b1.58)
**What:** Constrain all weights to {-1, 0, +1} with per-group absmax scaling.
**Why:** At 1.58 bits/weight, 16MB gives you **~65M parameters** vs ~15M at fp32+int8. That's 4x more capacity.
**Evidence:** A 73.7M ternary submission already exists in the repo at **1.1570 BPB** — only 0.076 behind SOTA, and it was an early submission without depth recurrence or TTT. Combining ternary with modern techniques could break SOTA.
**Key details:**
- Use NeoMuon optimizer (3 Newton-Schulz steps) to compensate for STE gradient attenuation
- Temperature scaling at eval (T=0.90 optimal)
- Needs custom forward with `sign()` + `absmax` scaling
- **Combine with depth recurrence + parallel residuals for potential SOTA**

### 2. Mamba/SSM Architecture
**What:** Replace transformer attention with selective state space model (Mamba-2).
**Why:** Linear-time sequence processing = more tokens/second = more training steps in 10 min. No quadratic attention cost.
**Evidence:** Mamba matches transformers at 130M-370M params. At tiny scale, the compute savings are proportionally larger.
**Key details:**
- Mamba-2's SSD layer is 2-8x faster than Mamba-1
- Deep-and-thin configuration (24-32 layers, dim 256-384)
- Can combine with ternary weights for extreme parameter count
- **Risk:** Less proven at <10M params, may need custom CUDA kernels

### 3. Multi-Token Prediction Training
**What:** Train to predict N=4 future tokens simultaneously (not just next-1).
**Why:** Free BPB improvement — extra prediction heads improve learned representations during training, then are discarded. Zero inference overhead, minimal parameter overhead.
**Evidence:** Meta's 2024 paper showed consistent gains even at small scale.
**Key details:**
- Add 3 small linear heads predicting tokens at positions t+2, t+3, t+4
- Shared backbone, independent heads
- Discard extra heads at submission — only next-token head matters
- Loss = sum of all 4 prediction losses (optionally weighted)

### 4. Byte Latent Transformer (Entropy-Based Patching)
**What:** Operate on raw bytes but dynamically group them into patches based on next-byte entropy. Hard bytes get smaller patches (more compute).
**Why:** Directly optimizes BPB evaluation metric. Avoids tokenizer overhead where tokens ≠ bytes.
**Evidence:** Meta's BLT (2024) matches tokenizer-based models on BPB.
**Key details:**
- Current competition uses SP1024-SP8192 tokenizers — switching to bytes is a paradigm shift
- Entropy model routes compute dynamically: `the` gets one patch, `Schrödinger` gets many
- Vocab size = 256 (byte-level) saves massive embedding parameters
- **Risk:** Novel, complex implementation. But huge upside if it works.

---

## Tier 2: Proven Techniques Not Yet Combined

### 5. XSA (Exclusive Self Attention)
**What:** Subtract from attention output the component aligned with each token's own value vector. Forces attention to learn orthogonal information.
**Why:** Consistent 0.002-0.003 BPB improvement in existing submissions.
**Details:** Apply only on last 3-4 layers. Efficient GQA implementation exists (2ms overhead).

### 6. SmearGate + BigramHash
**What:** Learned per-dimension gate blending current token with previous token (SmearGate), plus explicit bigram lookup table (BigramHash).
**Why:** Injects explicit bigram structure at embedding layer. 524K params for significant gains.
**Details:** SmearGate initialized near-identity. BigramHash uses `(prev*31 + curr) % 4096` → learned embedding.

### 7. Progressive/Delayed Recurrence
**What:** Don't enable depth recurrence from step 0 — activate it at 35-50% of training.
**Why:** Avoids training instability from architectural changes. Model adapts incrementally.
**Evidence:** Multiple top submissions use this.

### 8. SD-Clip (Standard-Deviation Quantization)
**What:** Clip threshold = k × std(row) instead of searching quantiles. Directly optimizes compressed size entropy.
**Why:** More principled than percentile clipping. k=12.85 for int6.

---

## Tier 3: Creative Moonshots

### 9. TTT Layers (Test-Time Training as Architecture)
**What:** Replace RNN hidden state with a model that literally trains on each test sequence via gradient descent during inference.
**Why:** Model adapts to each test document's distribution. Conceptually perfect for BPB.
**Risk:** Adds massive inference cost. The "legal TTT" variant in SOTA only trains on already-predicted tokens.

### 10. MoE at Tiny Scale (2 Experts)
**What:** 2 experts with top-1 routing. Doubles total parameters for same active compute.
**Why:** 16MB with 2 experts = 32M total params but only 16M active per token.
**Risk:** Routing overhead may dominate at this scale. Expert specialization needs enough training steps.

### 11. Kolmogorov-Arnold Networks (KAN)
**What:** Replace fixed activations + learnable weights with learnable activations (B-splines) + fixed structure.
**Why:** Theoretically more parameter-efficient function approximation.
**Risk:** Immature for language modeling. Memory overhead of B-splines is unclear in 16MB budget.

### 12. MatMul-Free Models
**What:** Replace all matrix multiplications with ternary accumulation operations.
**Why:** Combines naturally with BitNet. Could enable even larger models.
**Risk:** Needs custom CUDA kernels for training speed.

### 13. xLSTM (Hochreiter 2024)
**What:** Extended LSTM with exponential gating and matrix memory.
**Why:** Competitive with transformers and Mamba at medium scale.
**Risk:** Less ecosystem support, less proven at tiny scale.

---

## Concrete Architecture Proposals

### Proposal A: "Ternary Recurrent Transformer" (Most Practical)
Start from existing SOTA stack, add ternary weights:
- Ternary (1.58-bit) weights → ~65M params in 16MB
- 3-layer depth recurrence
- Parallel residuals + XSA on last 4 layers
- SmearGate + BigramHash embedding
- QK-Gain 5.25
- NeoMuon optimizer
- Legal score-first TTT at eval
- **Expected: could push below 1.08 BPB**

### Proposal B: "Ternary Mamba" (Highest Risk/Reward)
- Mamba-2 SSM backbone (24 layers, dim 384)
- Ternary weights → ~50-60M params
- Deep-and-thin per MobileLLM findings
- Multi-token prediction during training
- No attention overhead → faster training → more steps
- **Expected: unknown, but theoretically very parameter-efficient**

### Proposal C: "Byte-Level Patched Transformer" (Novel)
- Byte-level input (vocab=256, tiny embedding table)
- Entropy-based dynamic patching (BLT-style)
- Standard transformer backbone with depth recurrence
- All parameter budget goes to transformer layers, not embeddings
- Directly optimizes BPB metric
- **Expected: novel approach, strong narrative for non-record track**

### Proposal D: Stack Everything on Baseline (Most Conservative)
Take current SOTA train_gpt.py and add:
- XSA on last 4 layers (+0.002-0.003 BPB)
- SmearGate (+~0.002 BPB)
- Progressive recurrence activation (+stability)
- QK-Gain tuning to 5.5+ (+0.001 BPB)
- Hessian-aware SD-Clip (+0.001-0.002 BPB)
- **Expected: incremental but reliable ~0.005-0.008 BPB improvement**

---

## What's NOT Worth Trying
- **HyperNetworks** — shine in meta-learning, not single-task BPB
- **NAS** — no time budget for architecture search
- **NeRF-style weight parameterization** (our SIREN experiment confirms: 3.2s/step, 5.12 BPB — way too slow)
- **Liquid neural networks** — unproven for language
- **Structured pruning** — not useful at this scale

---

## Priority Order for Implementation
1. **Proposal A** (Ternary + existing SOTA techniques) — highest expected payoff
2. **Proposal D** (Stack incrementals on baseline) — safest bet
3. **Multi-token prediction** — easy to add to any architecture
4. **Proposal C** (Byte-level) — novel narrative for creative track
5. **Proposal B** (Ternary Mamba) — highest risk/reward
