# Category 1: Post-Transformer Architectures
**Research Date:** March 24, 2026  
**Purpose:** Parameter Golf — What architectures could help us train a better model in 16MB / 10 minutes?

---

## Executive Summary

Transformers' quadratic attention scaling is their core problem: memory and compute grow as O(N²) with sequence length. For a 16MB model (roughly 4–16M parameters depending on precision), this means:

1. **Training is slower** than it needs to be — the attention mechanism wastes compute on short contexts too
2. **Inference has KV-cache overhead** — irrelevant for a static 16MB model but reflects architectural bloat
3. **Parameter efficiency suffers** — transformer weight matrices carry structural redundancy

The post-transformer field has produced several architectures that achieve **linear-time training** (O(N log N) or O(N)), **constant-memory inference**, and in some cases **match or exceed transformer quality at smaller sizes**. The most promising for our use case:

- **RWKV-7** — runs as pure RNN at inference, trains like a transformer, no attention at all
- **Mamba/Mamba-2** — selective SSM, 5× inference speedup, trains as fast as transformers (Mamba-2)
- **xLSTM** — Pareto-dominates transformers on scaling laws (ICLR 2026), fastest 7B inference
- **Griffin** (DeepMind) — hybrid gated linear recurrence + local attention, matches Llama-2 on 6× fewer tokens

---

## Architecture Deep Dives

### 1. S4 — The Foundation
**What it is:** Structured State Spaces for Sequence Modeling (Gu et al., 2021). The foundational SSM architecture. Represents sequence data as a continuously evolving latent state using linear recurrence equations with special matrix initialization (HiPPO/Legendre). Exploits mathematical equivalence between recurrence and convolution to enable O(N log N) parallel training via FFT.

**Who built it:** Albert Gu (Stanford/CMU), Chris Ré's group (Stanford/HazyResearch)

**Key benchmarks:** First to demonstrate competitive accuracy on Long Range Arena benchmark while scaling sub-quadratically in sequence length.

**Why it matters for 16MB/10min:**
- Establishes the theoretical foundation everything else builds on
- O(N log N) training = faster iteration on small models
- Pure sequential inference = constant memory (critical when deploying tiny models)
- Variants (S4D, DSS, S5) simplified the approach before Mamba improved it further

**URL:** https://arxiv.org/abs/2111.00396

---

### 2. Mamba — Selective SSMs
**What it is:** Built on S4 but adds *selective* state-space layers (called S6). The key insight: previous SSMs had time-invariant parameters (same state update regardless of input content). Mamba makes state update input-dependent — each timestep's state matrices (A_t, B_t, C_t) are learned functions of the current token. This selective mechanism mimics attention's content-based weighting without quadratic overhead.

Architecture: pure SSM backbone — no attention heads, no separate FFN blocks. Stacked S6 layers with integrated MLP-like projections.

**Who built it:** Albert Gu (CMU) + Tri Dao (Princeton/Together AI), Dec 2023

**Key benchmarks:**
- 3B Mamba outperforms a transformer of the same size
- 3B Mamba matches a transformer 2× its size on language modeling
- ~5× higher inference throughput vs. transformers (no attention overhead)
- Works across language, audio, and genomics
- Context: tested up to millions of tokens with linear scaling

**Caveat — Mamba-1 training issue:** Custom parallel-scan algorithm achieved only ~10% GPU tensor core utilization, making training 2–3× *slower* than transformers. Fixed in Mamba-2.

**Why it matters for 16MB/10min:**
- At 16MB scale, 5× faster inference means faster evaluation loops during research
- No KV-cache means smaller memory footprint even at inference
- Input-dependent gating = better per-token efficiency = more "reasoning" per parameter
- The linear context scaling means training on longer sequences is cheap

**URL:** https://arxiv.org/abs/2312.00752  
**Code:** https://github.com/state-spaces/mamba

---

### 3. Mamba-2 — State Space Duality (SSD)
**What it is:** Fixes Mamba's training efficiency problem. Introduces *Structured State Space Duality* (SSD): by restricting the state transition matrix A_t to a scalar times identity (all state dimensions share the same decay rate), the recurrence can be unrolled and computed via standard dense matrix multiplications rather than bespoke element-wise scans. This unlocks full GPU tensor core utilization.

Mathematical insight: SSM recurrence and attention are "two sides of the same coin" — both can be viewed as special cases of a structured matrix acting on sequences. Mamba-2 proves this formally, enabling cross-pollination of techniques.

**Who built it:** Tri Dao + Albert Gu, 2024

**Key benchmarks:**
- Trains as fast as a transformer (GPU efficiency resolved)
- Maintains 5× inference speedup of Mamba-1
- SSD-based hybrid models (mixing SSD + attention) outperform pure designs at some scales
- Mamba-2 2.7B is competitive with transformer baselines on standard LM evals

**Why it matters for 16MB/10min:**
- Training parity with transformers = no speed penalty for choosing SSM over attention
- At small parameter counts (16MB ≈ 4–16M params), faster training per batch = more training steps in 10 minutes
- The scalar-A constraint is a simpler parameterization = fewer redundant weights = better parameter efficiency at tiny scales

**URL:** https://arxiv.org/abs/2405.21060 (Transformers are SSMs paper)  
**Code:** https://github.com/state-spaces/mamba (includes mamba2 models)  
**HuggingFace:** https://huggingface.co/state-spaces/mamba2-2.7b

---

### 4. RWKV — RNN with Transformer Training
**What it is:** Stands for Receptance Weighted Key Value. An RNN architecture that can be trained in parallel like a transformer (via prefix-sum operations on its recurrence) but runs as a pure RNN at inference — no attention mechanism whatsoever. Uses time-decay and learned gating factors to approximate linearized attention. The hidden state is fixed-size regardless of sequence length (O(1) memory at inference).

**Versions:**
- **RWKV-4** (2023): original, EMNLP 2023 paper
- **RWKV-5 "Eagle" / RWKV-6 "Finch"** (2024): multi-headed matrix-valued states + dynamic recurrence mechanism, improved expressivity
- **RWKV-7 "Goose"** (March 2025): introduces *Dynamic State Evolution* with vector-valued gating and in-context learning rates. **Exceeds TC⁰ expressive limits of attention/linear attention.** Can perform state tracking and recognize all regular languages — provably more expressive than transformers under standard complexity conjectures.
- **RWKV7-G1** (2025–2026): continuously updating production series

**Who built it:** Bo Peng (BlinkDL) + RWKV open-source community (Linux Foundation non-profit since Sep 2023)

**Key benchmarks:**
- RWKV-7 2.9B achieves new 3B SoTA on multilingual tasks, matches current 3B SoTA on English, trained on *fewer tokens* than competing models
- RWKV-6 14B competitive with Llama/Mistral 7–14B on Open LLM leaderboard
- 10–100× lower compute requirements vs. transformers at large context lengths
- Constant memory at inference regardless of sequence length

**Why it matters for 16MB/10min:**
- **Zero attention = no quadratic cost ever.** For a tiny model being trained repeatedly, this means faster batches on whatever hardware you have
- **O(1) inference memory** means no KV-cache overhead — a 16MB model can process arbitrarily long sequences without growing its footprint
- **Simplest training loop** — no FlashAttention custom kernels needed; runs on basic GPU without specialized ops
- The 0.4B model is already released and training-ready — useful reference architecture at near our scale
- RWKV's recurrent formulation is trivially parallelizable at the operator level, meaning you can write a clean, minimal implementation

**URLs:**
- RWKV-7 paper: https://arxiv.org/abs/2503.14456
- Eagle/Finch paper: https://arxiv.org/abs/2404.05892
- Code (Apache 2.0): https://github.com/RWKV/RWKV-LM
- Wiki: https://wiki.rwkv.com/

---

### 5. RetNet — Retentive Network
**What it is:** From Microsoft, proposed as "a successor to Transformer." Solves the "impossible triangle" of sequence modeling: training parallelism + O(1) inference + strong performance — all simultaneously. Uses a *retention* mechanism instead of attention, which supports three computation paradigms:
- **Parallel mode** (training): like transformer attention, fully parallelized
- **Recurrent mode** (inference): O(1) per token, constant memory, no KV-cache
- **Chunkwise recurrent** (long sequences): each chunk encoded in parallel, recurrently summarized across chunks

Uses exponential decay matrices and gating instead of softmax attention.

**Who built it:** Yutao Sun, Li Dong, Furu Wei et al. — Microsoft Research, July 2023

**Key benchmarks:**
- Favorable scaling results vs. transformers on standard LM benchmarks
- O(1) decoding = significantly faster inference throughput than transformers
- Competitive accuracy with same parameter count
- Lower GPU memory during inference (no KV-cache growth)

**Why it matters for 16MB/10min:**
- The chunkwise recurrent mode is highly efficient for training — you control chunk size to trade compute for memory
- Theoretical framework closest to transformers = easiest to port existing transformer training recipes
- O(1) inference removes a fundamental limitation for deployment

**URL:** https://arxiv.org/abs/2307.08621  
**Code:** https://aka.ms/retnet

---

### 6. Hyena — Long Convolutions + Gating
**What it is:** Stanford HazyResearch project. Attention-free layers using long implicit convolutions modulated by data-dependent gating. The *Hyena operator* interleaves:
- Long convolutional filters (e.g. 8k–64k length) computed in O(N log N) via FFT
- Multiplicative gates (inspired by LSTMs) for non-linear content-dependent interactions
- Filter weights *generated by a sub-network* — filter is parameterized implicitly, so the effective receptive field can be enormous with minimal extra parameters

**Who built it:** Michael Poli et al., Stanford/HazyResearch, Feb 2023

**Key benchmarks:**
- 355M Hyena = same perplexity as GPT-2 Medium (355M transformer) on WikiText-103
- 20% reduction in training compute at sequence length 2K vs. transformers
- 2× faster than FlashAttention at sequence length 8K
- 100× faster at sequence length 64K
- 50+ point accuracy improvements over SSM baselines on recall/reasoning tasks with very long sequences

**Why it matters for 16MB/10min:**
- **Implicit parameterization = better parameter efficiency.** The filter is generated by a small network, not stored as explicit weights — you get large effective receptive field at minimal parameter cost
- The O(N log N) FFT convolution scales better than O(N²) attention, even at short sequences where FFT has overhead
- The key parameter-golf insight: Hyena separates "receptive field size" from "parameter count." At 16MB, you can have a model that *acts* like it has much larger context capacity than its parameter count suggests

**URL:** https://arxiv.org/abs/2302.10866

---

### 7. StripedHyena — Hybrid Attention + Hyena
**What it is:** Together AI's production model built on Hyena. A hybrid architecture alternating Hyena gated convolution layers with standard attention layers (the "stripes" — hence the name). Uses FlashFFTConv for efficient Hyena inference. Trained using *compute-optimal scaling protocols* specifically targeting better scaling laws than Chinchilla-optimal transformers.

**Who built it:** Together AI + HazyResearch (Michael Poli et al.), Dec 2023

**Key benchmarks:**
- First alternative architecture competitive with best open-source transformers on *both* short and long context evaluations
- Comparable performance to Llama-2, Yi, and Mistral 7B on OpenLLM leaderboard
- Outperforms on long-context summarization
- >30%, >50%, >100% faster end-to-end training on sequences of 32K, 64K, 128K respectively vs. FlashAttention-v2 baseline
- Autoregressive generation caches >50% smaller than equivalent transformer with grouped-query attention
- Obtains higher quality than transformers at each training compute budget (better scaling laws)

**Why it matters for 16MB/10min:**
- The hybrid approach (mostly Hyena, some attention) demonstrated you can get transformer quality without paying transformer cost — especially for long contexts
- "Model grafting" technique used during training: you can start from transformer checkpoints and graft in Hyena layers — relevant if initializing from a pre-trained tiny transformer
- Compute-optimal scaling protocol = better model quality for a fixed compute budget = directly applicable to 10-minute training constraint

**URLs:**
- Blog: https://www.together.ai/blog/stripedhyena-7b
- HuggingFace (base): https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B
- FlashFFTConv: https://www.together.ai/blog/flashfftconv

---

### 8. xLSTM — Extended Long Short-Term Memory
**What it is:** LSTM modernized with two core innovations:
- **Exponential gating** with normalization/stabilization (critical — the original LSTM's additive memory made gradients unstable at scale)
- **Two variants:**
  - **sLSTM**: scalar memory, scalar update, new memory mixing — sequential, more expressive per parameter
  - **mLSTM**: matrix memory with covariance update rule — *fully parallelizable*, like a transformer but linear complexity

xLSTM blocks stack these into residual architectures. The 7B model (xLSTM 7B, March 2025) is reportedly the fastest and most efficient 7B LLM in terms of inference speed.

**Who built it:** Maximilian Beck, Sepp Hochreiter et al. (NXAI / JKU Linz, Austria), May 2024 (paper), March 2025 (7B model)

**Key benchmarks:**
- **xLSTM Scaling Laws (ICLR 2026):** xLSTM *Pareto-dominates* transformers — lower cross-entropy for the same compute budget, across model sizes 80M–7B
- 3.5× faster training than baseline transformer (xLSTM-7B paper)
- xLSTM 7B = fastest/most efficient 7B LLM compared to Llama- and Mamba-based models
- Linear context scaling at inference (vs. quadratic for transformers)
- Constant memory state size (vs. growing KV-cache for transformers)
- Competitive on RULER long-context benchmark against transformers with and without long-context finetuning

**Why it matters for 16MB/10min:**
- **Pareto-dominates transformers at all scales tested** — this is the single strongest claim for parameter efficiency, directly validated at scales from 80M down
- The scaling laws result (ICLR 2026) means: for a fixed 10-minute compute budget, an xLSTM architecture produces a better model than an equivalent transformer
- mLSTM's parallel training = no training speed penalty vs. transformer
- At 16MB (well under 80M params), xLSTM's advantages likely remain or strengthen — transformers' inductive biases are well-suited to large-scale pretraining but add overhead at small scale

**URLs:**
- Original xLSTM paper: https://arxiv.org/abs/2405.04517
- xLSTM 7B: https://arxiv.org/abs/2503.13427
- Scaling laws (ICLR 2026): https://arxiv.org/abs/2510.02228
- Code: https://github.com/NX-AI/xlstm

---

### 9. Griffin (Hawk) — DeepMind's Hybrid
**What it is:** Google DeepMind's production-scale recurrent architecture. Two variants:
- **Hawk**: pure RNN with gated linear recurrences
- **Griffin**: hybrid — gated linear recurrences + local attention (attending only to a fixed window, not full context)

The gated linear recurrence is simpler than Mamba's selective SSM but achieves competitive performance. Griffin has been scaled to 14B parameters.

**Who built it:** Soham De, Samuel Smith, Albert Gu et al. — Google DeepMind, Feb 2024

**Key benchmarks:**
- **Hawk exceeds Mamba on downstream tasks** (head-to-head, same scale)
- **Griffin matches Llama-2 despite training on 6× fewer tokens** — exceptional sample efficiency
- Griffin can extrapolate to sequences significantly longer than training length
- Matches transformer hardware efficiency during training
- Lower latency and significantly higher throughput at inference vs. transformers

**Why it matters for 16MB/10min:**
- **6× fewer tokens to match transformer quality** = directly relevant. In a 10-minute training window, you can train for fewer iterations and still reach the same quality
- The local attention window in Griffin is a practical compromise: you get attention's expressiveness for nearby tokens, recurrence for long-range, and you avoid the quadratic cost of full attention
- DeepMind's backing means strong engineering (hardware kernels, distributed sharding figured out)

**URL:** https://arxiv.org/abs/2402.19427

---

### 10. Jamba — Hybrid SSM-Transformer (Production)
**What it is:** AI21 Labs' production model and the first commercial-scale hybrid SSM-Transformer deployment. "Joint Attention and Mamba" — alternates Mamba SSM layers with standard transformer attention layers, combined with Mixture-of-Experts routing. Available via API.

Jamba 1.5 (released 2024–2025) demonstrated hybrid SSM-Transformer as production-viable for agentic AI use cases.

**Who built it:** AI21 Labs, March 2024

**Key benchmarks:**
- Better performance and efficiency than either pure Mamba or pure transformer at the same scale
- Handles 256K context window efficiently (the SSM layers handle bulk of context, attention layers handle key retrieval)
- Available on NVIDIA AI Enterprise

**Why it matters for 16MB/10min:**
- Validates that hybrid (few attention layers + many SSM layers) is a practical design pattern
- The architecture insight: you don't need to choose — use 10–20% attention layers for expressiveness, 80–90% SSM layers for efficiency
- This hybrid pattern is directly applicable at small scale: even 1–2 attention layers in a 16MB model might substantially improve quality over pure SSM

**URLs:**
- Blog: https://www.ai21.com/blog/announcing-jamba/
- VentureBeat: https://venturebeat.com/ai/ai21-debuts-jamba-1-5-boosting-hybrid-ssm-transformer-model-to-enable-agentic-ai

---

### 11. Falcon Mamba — Production SSM at Scale
**What it is:** Technology Innovation Institute (UAE) released Falcon Mamba 7B (Oct 2024) — a pure Mamba-based model trained on 5.8 trillion tokens. Demonstrates pure SSM (no attention) can beat transformer-based open-source models of similar size.

**Who built it:** Jingwei Zuo et al., TII UAE, October 2024

**Key benchmarks:**
- Surpasses Mistral 7B, Llama 3.1 8B, Falcon2 11B on standard benchmarks
- On par with Gemma 7B
- Outperforms RecurrentGemma 9B and RWKV-v6 Finch 7B/14B
- First time pure Mamba surpassed all leading open-source transformer models (not just "competitive with")

**Why it matters for 16MB/10min:**
- Proves pure SSM architectures have closed the gap with transformers even without hybrid tricks
- The training recipe (5.8T tokens on pure Mamba) has been made public — extractable lessons for training efficiency
- At 16MB scale, pure SSM has fewer moving parts = simpler, faster training loop

**URLs:**
- Paper: https://arxiv.org/abs/2410.05355
- HuggingFace: https://huggingface.co/tiiuae/falcon-mamba-7b

---

## Cross-Architecture Comparison

| Architecture | Training Complexity | Inference Complexity | KV-Cache | Training Speed vs. Transformer | Quality at Scale | Small-Model Suitability |
|---|---|---|---|---|---|---|
| **Transformer** | O(N²) | O(N²) or O(N) w/ cache | Yes, grows with context | Baseline | Proven SoTA | Moderate — attention adds overhead |
| **S4** | O(N log N) | O(N) or O(1) recurrent | No | Similar or slower (custom ops) | Good on long sequences | Good — simple math |
| **Mamba-1** | O(N) | O(1) | No | 2–3× *slower* (low GPU util) | Matches/beats equal-size transformer | Good — but training inefficiency hurts 10-min budget |
| **Mamba-2** | O(N) | O(1) | No | ~Same as transformer | Matches transformer | **Strong** — training parity, inference speedup |
| **RWKV-7** | O(N) | O(1) | No | Similar or faster at large N | New 3B SoTA multilingual | **Strong** — simple, no special kernels |
| **RetNet** | O(N) parallel | O(1) | No | ~Same as transformer | Competitive | Good — theoretical clarity |
| **Hyena** | O(N log N) | O(1) | No | 20% less at 2K; 100× faster at 64K | Matches transformer at medium scale | **Strong** — implicit params = efficient |
| **StripedHyena** | O(N log N) | O(1) | Smaller | 100%+ faster at 128K | Better scaling laws than transformer | Good — mainly proven at 7B |
| **xLSTM** | O(N) | O(1) | No | ~Same or faster (3.5×) | Pareto-dominates transformer | **Strongest** — validated from 80M down |
| **Griffin** | O(N) | O(1) | No | ~Same | Matches Llama-2 on 6× fewer tokens | **Strong** — sample efficiency wins |
| **Jamba** | Hybrid | Hybrid | Partial | Better than either pure | Better than pure at scale | Moderate — complex |

---

## Parameter Golf Application: 16MB / 10 Minutes

### The Problem Space
- **16MB model ≈ 4M–16M parameters** (fp32 = ~4 bytes/param → 4MB/16M params; fp16 = ~2 bytes/param → 8MB/16M params; int8 = 1 byte/param → 16MB/16M params)
- **10 minutes training** — approximately 600 gradient steps at 1 sec/step, or 6000 at 0.1 sec/step
- The goal: maximize quality per parameter, minimize wasted compute

### Architecture Recommendations (Ranked)

**Tier 1 — Strongest case:**

1. **xLSTM (mLSTM variant)** — Pareto-dominates transformers at all tested scales including 80M params. Trains as fast. Linear inference. Constant memory. The ICLR 2026 scaling laws paper is the strongest empirical evidence for parameter-efficient small models. At 16MB, the matrix memory in mLSTM should hold more information per parameter than transformer attention weights.

2. **RWKV-7** — Proven at 0.4B scale (available model), trains in parallel, runs as pure RNN. No specialized kernels needed. Open-source, Apache 2.0. The linear complexity means even on a laptop GPU you get competitive training throughput. The 0.19B model in RWKV-7's release is closest to our scale — a strong baseline.

3. **Griffin/Hawk** — 6× fewer tokens to match Llama-2 quality is the most directly relevant stat for a 10-minute training window. DeepMind's architecture has strong engineering behind it and the local-attention hybrid is a proven pattern.

**Tier 2 — Strong case:**

4. **Mamba-2** — Training parity with transformers + 5× inference speedup. The SSD formulation is theoretically clean and the code is open-source. The constraint (scalar-A matrix) actually acts as a regularizer which might help at tiny scales.

5. **Hyena / Implicit convolutions** — If parameter efficiency is the primary goal, Hyena's implicit parameterization of filters is potentially the most radical approach: the effective receptive field can exceed what explicit weights would suggest, meaning a 16MB model could "behave" like a much larger one for long-range patterns.

**Tier 3 — Worth watching:**

6. **RetNet** — Theoretical elegance and chunkwise recurrence are appealing. Less empirical validation at tiny scales than RWKV/xLSTM.

7. **Hybrid (few attention + mostly SSM)** — The Jamba/StripedHyena pattern: use 1–2 attention layers in the model for retrieval expressiveness, fill the rest with SSM layers. At 16MB, even 1 attention layer would represent a significant fraction of parameters, so this needs careful analysis.

### Specific Mechanisms That Help Most

**1. O(N) or O(N log N) training complexity**  
Direct: more training examples per second = more learning in 10 minutes. At 16MB, each forward pass is fast regardless, but at sequence lengths > 512, linear architectures have measurable advantages.

**2. No KV-cache at inference**  
Indirect but important: simplifies the inference pipeline during evaluation/testing loops, meaning you can run evaluations faster during training, which means faster iteration.

**3. Implicit parameterization (Hyena-style)**  
High-leverage for tiny models: a small network generating a large filter effectively expands the model's "capacity" beyond its raw parameter count. This is the most aggressive parameter-efficiency technique available.

**4. Constant hidden state size (RNN mode)**  
For deployment: a 16MB model that handles arbitrary context lengths without growing memory is practical in ways a transformer-based 16MB model is not.

**5. Sample efficiency (Griffin's 6× fewer tokens)**  
If the 10-minute constraint is the binding one: training on less data to reach the same quality = more training passes over higher-quality data = better final model.

---

## Key Papers Summary

| Paper | Authors | Year | Link |
|---|---|---|---|
| S4: Structured State Spaces | Gu, Goel, Ré | 2021 | https://arxiv.org/abs/2111.00396 |
| Mamba: Selective State Spaces | Gu & Dao | 2023 | https://arxiv.org/abs/2312.00752 |
| Hyena Hierarchy | Poli et al. | 2023 | https://arxiv.org/abs/2302.10866 |
| RetNet | Sun, Dong et al. | 2023 | https://arxiv.org/abs/2307.08621 |
| RWKV-4 | Peng et al. | 2023 | https://arxiv.org/abs/2305.13048 |
| Mamba-2 / SSD | Dao & Gu | 2024 | https://arxiv.org/abs/2405.21060 |
| Griffin / Hawk | De, Smith, et al. | 2024 | https://arxiv.org/abs/2402.19427 |
| Eagle & Finch (RWKV-5/6) | Peng et al. | 2024 | https://arxiv.org/abs/2404.05892 |
| xLSTM | Beck, Hochreiter et al. | 2024 | https://arxiv.org/abs/2405.04517 |
| Falcon Mamba 7B | Zuo et al., TII | 2024 | https://arxiv.org/abs/2410.05355 |
| RWKV-7 "Goose" | Peng et al. | 2025 | https://arxiv.org/abs/2503.14456 |
| xLSTM 7B | Beck et al. | 2025 | https://arxiv.org/abs/2503.13427 |
| xLSTM Scaling Laws (ICLR 2026) | Beck et al. | 2025 | https://arxiv.org/abs/2510.02228 |

---

## Key Organizations

| Org | Architecture | Status |
|---|---|---|
| CMU / Princeton (Albert Gu, Tri Dao) | S4, Mamba, Mamba-2 | Academic → Together AI |
| Stanford HazyResearch (Chris Ré, Michael Poli) | S4, Hyena, FlashFFTConv | Academic + startup (Hazy Research) |
| Together AI | StripedHyena, FlashFFTConv | Commercial, open-source releases |
| Microsoft Research | RetNet | Academic/commercial |
| Bo Peng / RWKV Community (Linux Foundation) | RWKV 1–7 | Open-source non-profit |
| NXAI / JKU Linz (Sepp Hochreiter) | xLSTM | Startup + academic |
| Google DeepMind | Griffin, Hawk, RecurrentGemma | Commercial |
| AI21 Labs | Jamba, Jamba 1.5 | Commercial API |
| TII UAE | Falcon Mamba | Open-source (Falcon license) |

---

## Bottom Line for Parameter Golf

**If you can use one insight:** xLSTM's mLSTM variant Pareto-dominates transformers at scales from 80M down (ICLR 2026), trains at the same speed, and has linear inference complexity. At 16MB, the theoretical advantages only compound — transformers' inductive biases (learned QKV projections per layer) add parameter overhead that matters more at small scales.

**If you need the simplest implementation:** RWKV-7 is a pure RNN at inference with transformer-like parallelizable training. No FlashAttention, no special kernels. The 0.19B release provides a reference implementation at near our scale. Apache 2.0 licensed.

**If sample efficiency is the constraint:** Griffin matches Llama-2 on 6× fewer tokens. In a 10-minute training window, this is the most direct win.

**The hybrid pattern:** Use mostly SSM/recurrent layers (RWKV or mLSTM) with 0–2 attention layers. The attention layers provide the recall capability SSMs lack; the recurrent layers handle the bulk of computation efficiently.
