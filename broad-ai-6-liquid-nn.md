# Category 6: Liquid Neural Networks & Liquid Foundation Models
*Research date: 2026-03-24 | For: Parameter Golf / 16MB / 10-minute training*

---

## TL;DR for Parameter Golf

Liquid AI's LFM2 series includes a **350M-parameter model** that fits easily under 16MB (quantized to ~175MB FP16, but GGUF Q4 ~200MB — still sizeable). The *architectural ideas*, however, are directly applicable: **short convolutions + sparse attention hybrids** are demonstrably more parameter-efficient than pure transformers at small scale. LFM-1B first beat transformers at 1B in 2024. LFM2.5-1.2B in late 2025 runs under 1GB RAM at 239 tok/s on CPU. The key insight for our task: **replacing or reducing transformer attention with gated convolutions + SSM layers** can hit better accuracy per parameter, train faster per token, and crucially — use near-constant memory regardless of context length.

---

## 1. What Are Liquid Neural Networks?

### Origins: MIT CSAIL, 2016–2022
Liquid Neural Networks (LNNs) were invented by **Ramin Hasani** (PhD thesis, TU Wien / MIT CSAIL) and **Mathias Lechner**, working under **Daniela Rus** at MIT. Inspired by the wiring of *C. elegans* (302 neurons), these are continuous-time, dynamical systems where:

- Neuron state evolves via **ordinary differential equations (ODEs)**
- The **time constant τ is itself input-dependent** (hence "liquid" — the dynamics adapt to the data)
- Even tiny models (19 neurons) could control autonomous vehicles robustly

The ODE formula for a liquid neuron:
```
ẋ = -[1/τ + f(x, I_synapse, t)] · x + f(x, I_synapse, t)
```
Where `τ` adapts based on input — the network literally changes its temporal dynamics in real-time.

**Key Papers:**
- Lechner et al., *Nature Machine Intelligence* 2020 — original LNN paper: https://www.nature.com/articles/s42256-020-00237-3
- Hasani et al., AAAI 2021 — universal approximator proofs: https://ojs.aaai.org/index.php/AAAI/article/view/16936/16743

### Why This Matters
Standard RNNs have fixed dynamics. Transformers have fixed attention patterns (learned but static at inference). Liquid networks have **input-dependent time constants** — the network's internal dynamics shift based on what it's seeing right now. This produces:
- Stronger robustness to distribution shift (proven in autonomous flight nav)
- Better temporal reasoning on irregular time-series
- Parameter efficiency: fewer parameters needed for same expressivity

---

## 2. The Critical Evolution: LTCs → CfCs → Liquid-S4 → LFM → LFM2

The research path from tiny LNNs to production LFMs is a straight line:

### 2022: Closed-form Continuous-time Networks (CfC)
The original LNNs were computationally expensive due to ODE solver requirements. The CfC breakthrough solved this:

- Hasani et al., *Nature MI* 2022: https://www.nature.com/articles/s42256-022-00556-7
- Computed a **closed-form approximation** of the LTC ODE integral
- Result: **1–5 orders of magnitude faster** training/inference vs ODE-based counterparts
- No numerical solver needed; time appears explicitly in closed form
- Can now scale to real deep learning tasks

**Direct relevance for 10-minute training**: CfC's closed-form solution means what previously required iterative ODE integration now runs as a simple equation. Training speed up of 10–100,000x over prior continuous-time models.

### 2022: Liquid-S4 (ICLR 2023)
Merged CfC dynamics into the S4 structured state-space framework:

- Hasani et al., arXiv 2022: https://arxiv.org/abs/2209.12951
- Linear LTC state-space model using S4's diagonal + low-rank decomposition
- Achieved **87.32% average on Long-Range Arena** (new SOTA at time)
- **30% fewer parameters** than S4 on Speech Commands (96.78% accuracy)
- Input-dependent state transition: model learns to adapt its memory compression to the input

### 2023: Hyena (Liquid AI / Stanford / Together AI)
- Poli et al., ICML 2023: https://arxiv.org/abs/2302.10866
- Subquadratic **drop-in replacement for attention**
- Interleaved long convolutions + data-controlled gating
- Transformer quality at **20% less training compute** at sequence length 2K
- **2x faster** than attention at 8K tokens; **100x faster** at 64K tokens
- First truly practical attention-free architecture for language modeling

### 2024: First Generation Liquid Foundation Models (LFM-1B/3B/40B)
Liquid AI's commercial debut:

- Blog: https://www.liquid.ai/blog/liquid-foundation-models-our-first-series-of-generative-ai-models
- LFM-1B: **first non-GPT architecture to significantly outperform transformers at 1B**
  - MMLU: 58.55 vs Llama 3.2 1.2B's 45.46
  - GSM8K: 55.34 vs Llama 3.2's 33.36
- LFM-3B: comparable to Phi-3.5-mini at **18.4% smaller**
- Architecture: "adaptive linear operators whose actions are determined by inputs" — unified framework for SSMs, convolutions, and attention
- **Near-constant inference memory** as context grows (vs. O(n) KV cache for transformers)

### 2025: LFM2 — The Practical Edge AI Family
The current state of the art (released November 2025):

- Technical report: https://arxiv.org/abs/2511.23404
- HuggingFace: https://huggingface.co/LiquidAI
- Model sizes: **350M, 700M, 1.2B, 2.6B** (dense) + **8.3B MoE** (1.5B active)
- Architecture: **hybrid — gated short convolutions + grouped query attention**
  - 350M: 16 layers (10 conv + 6 attention)
  - 1.2B: 16 layers (10 conv + 6 attention)
  - 2.6B: 30 layers (22 conv + 8 attention)
- Training: **hardware-in-the-loop NAS** under edge latency/memory constraints
- **3x faster training** vs previous LFM generation
- **2x faster decode and prefill** on CPU vs similarly-sized models (Qwen3 comparison)
- Knowledge distillation: tempered, decoupled Top-K objective
- Curriculum learning: difficulty-ordered data

### 2025: LFM2.5-1.2B — The Thinking Small Model
Released early 2026 (trained late 2025):

- HuggingFace: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
- **239 tok/s decode on AMD CPU**, 82 tok/s on mobile NPU
- **<1GB RAM** with GGUF quantization
- Extended training: **28 trillion tokens** (vs 10T for LFM2)
- Multi-stage post-training: SFT + length-normalized preference optimization + model merging
- Includes Thinking variant (chain-of-thought reasoning in 1.2B)
- Architecture: 16 layers with "double-gated LIV convolution blocks" + GQA attention

---

## 3. Key Players

| Organization | People | Focus | Key Models |
|---|---|---|---|
| **Liquid AI** (Cambridge MA) | Ramin Hasani, Mathias Lechner, Daniela Rus, Michael Poli, Stefano Massaroli | Commercial LFMs | LFM1, LFM2, LFM2.5 |
| **MIT CSAIL** | Daniela Rus group | Research: LNNs, CfC | Academic papers |
| **Stanford / Together AI** | Michael Poli, Christopher Ré | Hyena, StripedHyena, scaling laws | Hyena, HyenaDNA |
| **NVIDIA** | Albert Gu, Tri Dao | Mamba, S6, SSM scaling | Mamba, Mamba-2, hybrid models |
| **CMU / Gu Lab** | Albert Gu | S4, S4D, Mamba | Linear SSMs |

---

## 4. How LFMs Compare to Transformers at Small Scale

### The Core Advantage: Memory + Speed

| Property | Transformer | LFM/SSM Hybrid |
|---|---|---|
| KV cache (inference) | O(n) — grows with context | ~O(1) — fixed state size |
| Training complexity | O(n²) attention | O(n log n) or O(n) |
| Long-range recall | Excellent | Good (with hybrid attention) |
| Parameter efficiency at <3B | Weaker | Stronger (proven empirically) |
| Irregularly-sampled sequences | Poor | Excellent |
| Training speed | Baseline | 2–3x faster per token |

### Benchmark Evidence (LFM2-1.2B vs transformers of same size)

From the LFM2 technical report (arXiv 2511.23404):
- **IFEval**: LFM2-2.6B achieves 79.56%
- **GSM8K**: LFM2-2.6B achieves 82.41%
- LFM2-350M: beats models 2x its parameter count on several tasks

From NVIDIA's Mamba-2-Hybrid study (arXiv 2406.07887):
- Pure SSMs **match or exceed** transformers on 8 of 12 standard benchmarks at 8B
- **Hybrid (43% Mamba-2 + 7% attention + 50% MLP) beats pure transformer +2.65 avg**
- Hybrid is **up to 8x faster** at inference token generation
- Pure SSMs lag on: in-context learning, 5-shot MMLU, phonebook tasks

### Where Pure Transformers Still Win
At sub-1B scale, attention is expensive but also highly effective for:
- Few-shot in-context learning
- Precise copying/retrieval tasks
- Tasks requiring strong "content-based reasoning"

The LFM2 approach of using **sparse attention (6 out of 16 layers)** retains these strengths while gaining SSM efficiency.

---

## 5. The Architecture Family — Taxonomy

### Liquid Time-Constant (LTC) Networks
- Continuous-time ODE with input-dependent time constants
- Small-scale only (10–1000 neurons); too slow for large-scale pre-2022
- **Best for**: robotics, control, irregularly-sampled time series
- Not directly usable for 16MB LLM training

### Closed-form Continuous-time (CfC)
- Analytical approximation of LTC dynamics; no ODE solver
- **1–5 orders of magnitude faster** than LTC
- Scales to supervised learning; still not transformer-scale
- Architecture: CfC cell replaces LSTM/GRU in sequence models

### Liquid-S4
- CfC-inspired state transition fused into S4 SSM framework
- Achieves S4 efficiency + adaptive dynamics
- **30% parameter reduction** on speech benchmarks
- Runs with standard deep learning infrastructure

### S4 → S5 → S6 (Mamba) → S7 — The SSM Progression
All part of the same research tree:
- **S4** (Gu et al., NeurIPS 2021): https://arxiv.org/abs/2110.13985 — LSSL + HiPPO theory
- **S5** (Smith et al., ICLR 2023): https://arxiv.org/abs/2208.04933 — parallel scan; 87.4% on LRA
- **Mamba** (Gu & Dao, 2023): https://arxiv.org/abs/2312.00752 — selective state spaces (S6); input-dependent; linear-time
- **Hyena** (Poli et al., 2023): https://arxiv.org/abs/2302.10866 — long convolution + data-controlled gating

### LFM2 "Hybrid Liquid" Architecture (2025)
Liquid AI's production design:
- **Gated short convolutions** as primary sequence mixing (not self-attention)
- **Multiplicative gates** for input-dependent dynamics (the "liquid" property)
- Small number of **grouped query attention** blocks for content-based reasoning
- Hardware-in-the-loop NAS: architecture searched under actual device latency constraints
- Training: standard PyTorch + ExecuTorch/llama.cpp for deployment

---

## 6. HOW This Helps Train a Better Model in 16MB / 10 Minutes

This is the money section. Here's the specific mechanism by which each insight applies:

### 6.1 Replace Full Attention with Short Convolutions (Primary Gain)
**What**: LFM2 uses gated short convolutions for 10/16 layers. These are literally cheaper than self-attention (O(n) vs O(n²) in sequence length, and the constant factor is lower).

**For 16MB/10min training**:
- Attention layers require 4 weight matrices (Q, K, V, O) each of size d_model × d_model
- Short conv layer: 1 weight tensor of size d_model × kernel_size — **much smaller**
- A 16-layer model with 10 conv + 6 attention uses ~60–70% fewer parameters in mixing layers
- Training a token takes less compute → more tokens per minute → better model from same budget

**Concrete calc**: For d_model=64, standard attention uses 4 × (64×64) = 16,384 params per layer. Short conv with kernel_size=4: 64×4 = 256 params per layer. Same-count model with fewer mixing params → either you can go deeper, or same depth with more capacity elsewhere (e.g., wider FFN).

### 6.2 Fixed Recurrent State = No KV Cache = Fits in 16MB
**What**: SSM/Mamba-style layers maintain a fixed-size recurrent state. At inference, there's no KV cache that grows with sequence length.

**For 16MB**:
- At 16MB, KV caches for even 512-token context at 4-bit quant are expensive
- A pure recurrent model (no attention) eliminates this entirely
- The model becomes truly **constant-memory** at inference regardless of context length
- This makes the 16MB constraint primarily about *weights*, not *runtime memory*

**Tradeoff**: Pure SSM loses in-context learning ability vs attention. The LFM2 hybrid (sparse attention) is the right answer — retain ~6 attention layers for ICL while gaining recurrent efficiency everywhere else.

### 6.3 Faster Training Per Token = More Learning in 10 Minutes
**What**: LFM2 achieves **3x faster training** vs its previous generation, and **2x faster** decode/prefill vs Qwen3 on CPU. The Hyena paper shows 20% less training compute vs transformers at 2K context.

**For 10-minute training**:
- If training is 2–3x faster, you train on 2–3x more data in the same wall-clock time
- Or: same data, model learns from more steps of gradient descent
- Practical implication: a 10-minute training run becomes equivalent to 20–30 minutes of transformer training

### 6.4 Input-Dependent Gating = Better Expressivity Per Parameter
**What**: The "liquid" property — multiplicative gates that adapt based on input — means each parameter is used more efficiently. The same weight contributes differently to different inputs.

**For 16MB**:
- A model with input-dependent gating can learn more complex functions with fewer parameters
- The CfC/LTC research showed this empirically: smaller models generalize better
- Practical gain: might get GPT-2-small quality in 50–70% of the parameters

### 6.5 Knowledge Distillation + Curriculum Learning (LFM2 Training Pipeline)
**What**: LFM2's training pipeline includes:
- **Tempered, decoupled Top-K KD**: avoids support mismatch in distillation
- **Curriculum learning**: difficulty-ordered data
- **Three-stage post-training**: SFT → preference optimization → model merging

**For 16MB/10min**:
- Distillation from a larger teacher can compress capabilities into 16MB
- Curriculum learning (easy → hard) accelerates convergence — especially valuable when total training time is 10 minutes
- The "decoupled Top-K" distillation approach handles the case where teacher and student have different vocabularies/distributions — relevant for domain-specific small models

### 6.6 Hardware-in-the-Loop NAS — Design the Architecture for Your Target
**What**: LFM2 searched its architecture under real CPU/NPU latency constraints. Not "theoretically efficient" but actually fast on target hardware.

**For 16MB/10min**:
- When targeting a specific hardware profile (e.g., CPU inference), run NAS to find the best conv/attention ratio under your memory budget
- Even without full NAS: the published finding (10 conv + 6 attention for 350M) is a validated starting point
- The 10:6 conv:attention ratio is a reference design you can directly apply

### 6.7 Transfer Function Parametrization = Faster Training Convergence
**What**: Liquid AI's RTF (Rational Transfer Function) paper (arXiv 2405.06147) achieved **35% faster training** over S4 on Long Range Arena while maintaining or improving accuracy.

**For 10-minute training**:
- Frequency-domain transfer function parametrization for SSM layers
- Single FFT computes convolutional kernel spectrum directly
- No recurrent state overhead during training (state-free inference)
- 35% training speedup compounds with the other architectural savings

---

## 7. Related Architectures Worth Knowing

### Mamba (Gu & Dao, 2023)
- **What**: Selective state spaces (S6) — SSM parameters are functions of input (like liquid networks)
- **Who**: Albert Gu (CMU), Tri Dao (Stanford/Together)
- **Benchmark**: Mamba-3B outperforms transformers of same size; matches transformers 2x its size
- **Speed**: 5x higher throughput than transformers
- **For 16MB**: Mamba layer is ~efficient, but the pure-Mamba approach loses on ICL tasks. Hybrid (NVIDIA's Mamba-2-Hybrid) is better.
- Paper: https://arxiv.org/abs/2312.00752

### Mamba-2-Hybrid (NVIDIA, 2024)
- **What**: 43% Mamba-2 + 7% attention + 50% MLP, 8B params
- **Finding**: Beats pure transformer +2.65 avg on 12 benchmarks; up to 8x faster inference
- **Validated up to 128K context** — the hybrid retains long-context ability
- Paper: https://arxiv.org/abs/2406.07887

### RWKV (Peng et al., 2023)
- RNN-style architecture with linear attention approximation
- Fully open source, community-developed
- 1.6B RWKV included in Liquid AI's own benchmark tables (lower performance than LFM-1B)
- Relevant for 16MB: RWKV has tiny inference footprint, but lower quality per parameter vs LFM2 hybrids

### Hyena (Poli et al., 2023)
- **What**: Subquadratic long convolutions + data-controlled gating
- Developed by Michael Poli who later joined Liquid AI
- 100x faster than attention at 64K tokens
- "HyenaDNA" applies this to genomics sequences up to 1M length
- **For 16MB**: Hyena-style long convolutions could replace attention in smaller models

---

## 8. What Liquid AI Has Released (as of March 2026)

From HuggingFace (https://huggingface.co/LiquidAI):
- **LFM2-350M** — 354M params, 10T training tokens, 32K context, open weights
- **LFM2-700M** — 742M params
- **LFM2-1.2B** — 1.17B params
- **LFM2-2.6B** — 2.57B params
- **LFM2-8B-A1B** — 8B total, 1B active (MoE)
- **LFM2-24B-A2B** — 24B total, 2B active (MoE)
- **LFM2.5-1.2B-Instruct** — improved, 28T token training, RL post-training
- **LFM2.5-1.2B-Thinking** — reasoning model
- **LFM2-VL** — vision-language variants
- **LFM2.5-Audio-1.5B** — speech + text
- **LFM2-ColBERT** — retrieval encoder

All models available in GGUF (llama.cpp), ONNX, MLX, vLLM formats.

**License**: LFM Open License v1.0 — not fully Apache 2.0, has commercial restrictions but allows research use.

---

## 9. Practical Implementation Path for Parameter Golf

### Recommended Architecture Stack
Based on everything above, the optimal architecture for **16MB / 10-minute training** draws directly from LFM2:

```
Model target: ~15M parameters (fits in 16MB at INT8 or ~8MB at INT4)

Recommended hybrid design:
- d_model: 256–384
- Layers: 12–16 total
  - ~8-10 gated short convolution layers (kernel_size=4-8)
  - ~4-6 grouped query attention layers (GQA with few heads)
- FFN: small (2x expansion) or removed on conv layers (like the "One Wide FFN" paper)
- Tokenizer: small vocab (16K–32K BPE)

Estimated params:
- Conv layers (256 dim, k=4): 8 × (256×4 + 256×256) ≈ 8 × 66K ≈ 530K
- GQA layers (256 dim, 4 heads): 4 × (256×256×3 + proj) ≈ 4 × 200K ≈ 800K
- FFN layers: 16 × 256×512×2 ≈ 4M
- Embeddings (16K vocab × 256): 4M
- Total: ~10–15M params
```

### Training Recipe from LFM2.5
1. **Curriculum learning** (difficulty-ordered data) — more convergence per minute
2. **Knowledge distillation** from a teacher (even a 1B model) — Top-K tempered
3. **Short conv layers** (gated, multiplicative) as primary mixing
4. **Sparse attention** (few layers) for in-context learning retention
5. Input-dependent gating throughout (the "liquid" property)

### Key Implementation References
- LFM2 weights (350M, Apache-ish): https://huggingface.co/LiquidAI/LFM2-350M
- Mamba PyTorch implementation: https://github.com/state-spaces/mamba
- RTF transfer function parametrization: https://github.com/ruke1ire/RTF
- Hyena implementation: part of Together's open source

---

## 10. Critical Limitations

**What LFMs are NOT good at** (per Liquid AI's own docs):
- Zero-shot code tasks
- Precise numerical calculations
- Time-sensitive information (knowledge cutoff)
- Tasks requiring strong in-context learning at tiny scale

**For 16MB specifically**:
- LFM2-350M in FP16 is ~700MB — still too big for 16MB storage as-is
- The *ideas* apply; the actual checkpoints need aggressive quantization (GGUF Q2) or you design from scratch at 15M params
- At <50M params, there's limited published evidence for LFM-style hybrid architectures. LFM2's validated range starts at 350M.
- The 10-minute constraint: LFM2 was trained on 10T tokens. Any novel training run in 10 minutes is orders of magnitude smaller data — the architecture principles apply, but don't expect to replicate benchmark numbers.

---

## 11. Summary & Verdict

| Insight | Applicability to 16MB/10min | Difficulty | Gain |
|---|---|---|---|
| Gated short convolutions instead of attention | ★★★★★ High | Low (just architecture choice) | 2–3x fewer params in mixing |
| Sparse attention (6/16 layers hybrid) | ★★★★★ High | Low | Retain ICL + efficiency |
| Input-dependent multiplicative gating | ★★★★☆ High | Medium (need right init) | Better expressivity/param |
| Curriculum learning | ★★★★☆ High | Low (data ordering) | Faster convergence |
| Knowledge distillation (Top-K tempered) | ★★★★☆ High | Medium | Compresses teacher knowledge |
| CfC/closed-form dynamics | ★★★☆☆ Medium | High (novel impl) | Better temporal tasks |
| Hardware-in-the-loop NAS | ★★★☆☆ Medium | Very high (requires infra) | Optimal hw efficiency |
| RTF frequency-domain training | ★★★☆☆ Medium | High | 35% training speedup |

**Bottom line**: The LFM2 hybrid architecture (short conv + sparse attention) is the single most actionable finding. It's validated, implemented, available as open weights, and the design rationale is documented in the technical report. For parameter golf at 16MB, this means:

1. Use a **hybrid backbone** — majority conv layers, minority attention layers
2. Apply **gated multiplicative units** (not plain ReLU activations)
3. Use **GQA** (grouped query attention) not full MHA for attention layers
4. Train with **curriculum + distillation** if a teacher is available
5. The 350M LFM2 as a **distillation teacher** is an interesting option even if the student is much smaller

---

## Key URLs

| Resource | URL |
|---|---|
| Liquid AI main site | https://www.liquid.ai |
| LFM1 blog post | https://www.liquid.ai/blog/liquid-foundation-models-our-first-series-of-generative-ai-models |
| Liquid AI research history | https://www.liquid.ai/research/liquid-neural-networks-research |
| LFM2 technical report (arXiv) | https://arxiv.org/abs/2511.23404 |
| LFM2 HuggingFace | https://huggingface.co/LiquidAI |
| LFM2-350M model card | https://huggingface.co/LiquidAI/LFM2-350M |
| LFM2.5-1.2B-Instruct | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |
| CfC paper (Nature MI 2022) | https://www.nature.com/articles/s42256-022-00556-7 |
| Liquid-S4 (ICLR 2023) | https://arxiv.org/abs/2209.12951 |
| Hyena (ICML 2023) | https://arxiv.org/abs/2302.10866 |
| Mamba (Dec 2023) | https://arxiv.org/abs/2312.00752 |
| Mamba-2-Hybrid (NVIDIA 2024) | https://arxiv.org/abs/2406.07887 |
| Mechanistic Hybrid Design (ICML 2024) | https://arxiv.org/abs/2403.17844 |
| RTF transfer function SSM | https://arxiv.org/abs/2405.06147 |
| S4 linear state spaces (NeurIPS 2021) | https://arxiv.org/abs/2110.13985 |
| S5 simplified SSM (ICLR 2023) | https://arxiv.org/abs/2208.04933 |
| Liquid AI Playground | https://playground.liquid.ai |

---

*Research by: Subagent research-cat6-liquid-nn | Completed: 2026-03-24*
