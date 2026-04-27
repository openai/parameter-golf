# Category 8: Sparse Computation & Conditional Compute
## Research Brief for Parameter Golf — 16MB / 10-Minute Training

**Date:** March 2026  
**Focus:** Sparse MoE, Switch Transformers, Hash layers, routing networks — activating only 10–25% of parameters per input

---

## Executive Summary

Sparse computation is the single most powerful architectural lever for breaking the "more parameters = more compute" constraint. The core idea: a model has a **large total parameter count** but **only activates a small fraction per input token**. For Parameter Golf, this is a paradigm shift — instead of 16MB of uniformly active weights, you could have 16MB of *routed* parameters where only 1.6MB (~10%) compute on any given token.

This is not theoretical. DeepSeek-V3 (671B total, ~37B active per token), Mixtral 8x7B (47B total, ~12B active per token), and OLMoE (7B total, 1B active) all prove the architecture at scale. The key question for Parameter Golf: **does sparse routing help at 16MB total, or does the routing overhead eat the budget?**

Short answer: **Yes, if done carefully**. This file explains how.

---

## 1. Core Concept: What Is Conditional Compute?

### The Dense Baseline Problem
In a standard dense transformer, every input token passes through every weight matrix. A 16MB model with 4M parameters (fp32) computes ALL 4M parameters for every token. That's both the ceiling and the floor — you cannot extract more capacity without increasing compute.

### The Conditional Compute Insight
First articulated formally by Yoshua Bengio circa 2013 and operationalized by Shazeer et al. in 2017, conditional compute proposes: **only activate the parts of the network relevant to each input**.

The analogy: a generalist vs. a panel of specialists. Rather than one doctor who knows a bit about everything, have 8 specialists and route each patient to the 1–2 most relevant ones.

In neural network terms:
- Replace FFN (Feed-Forward Network) layers with **Mixture-of-Experts (MoE) layers**
- Each MoE layer has N expert sub-networks (usually mini-FFNs)
- A learned **router/gate network** selects which K experts to activate per token (typically K=1 or K=2)
- Only the selected experts compute; the rest are "off" for that token

**Result:** Total parameters scale by Nx, but compute stays constant (or grows by K/N — the fraction activated).

---

## 2. Landmark Papers & Architectures

### 2.1 Sparsely-Gated MoE (Shazeer et al., 2017) — The Founding Paper
**Paper:** *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*  
**URL:** https://arxiv.org/abs/1701.06538  
**Who:** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (Google Brain)

**What it is:** The first practical large-scale sparse MoE. Applied to 137B parameter LSTM for machine translation. Introduced:
- Trainable gating network: softmax over all experts, take top-K
- Auxiliary load balancing loss to prevent all routing to a single expert
- Noise injection during training to prevent routing collapse

**Why it matters:** Proved 1000x capacity improvement with only minor efficiency losses. Established the two failure modes of MoE: (1) load imbalance — one expert gets everything, (2) routing instability — gates fluctuate wildly.

**16MB relevance:** The load balancing loss is critical even in tiny models. Without it, a 16MB MoE degenerates into a 16MB dense model where one expert handles everything.

---

### 2.2 Switch Transformer (Fedus et al., 2021 / JMLR 2022) — Simplicity at Scale
**Paper:** *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*  
**URL:** https://arxiv.org/abs/2101.03961  
**Who:** William Fedus, Barret Zoph, Noam Shazeer (Google)

**What it is:** Simplified MoE routing to **K=1** (each token routes to exactly ONE expert). This eliminates the combinatorial complexity of multi-expert routing and makes the math tractable for very large N.

Key contributions:
- **Top-1 routing:** Send each token to the single best-matching expert. Cleaner gradients, less communication overhead.
- **Expert capacity buffer:** Each expert has a "token bucket" — if it's full, tokens are passed through unchanged (dropped). Controlled overflow, not chaos.
- **bfloat16 training for sparse models** — first to demonstrate stable MoE training in reduced precision
- **7x speedup** in pre-training vs. T5-Base at matched compute budget
- Scaled to trillion parameters on Colossal Clean Crawled Corpus (C4)

**How routing works mechanically:**
```
Router = Linear(d_model, n_experts)  # tiny linear projection
token_scores = softmax(Router(x))    # probability dist over experts
expert_id = argmax(token_scores)     # pick top-1
route token x to expert[expert_id]
```
The router adds ~0.1% parameter overhead. That's the price for activating ~12.5% of parameters (with 8 experts).

**16MB relevance:**  
If you have 16MB total and split into 8 experts of ~2MB each, plus shared embeddings + attention layers:
- At inference, only 1 expert activates per token = ~2MB compute + shared layers
- During training, all experts eventually get used (load balancing ensures this)
- You get 8x capacity expansion "for free" in terms of inference compute

**Critical constraint for tiny models:** The router itself requires training data to learn good assignments. With limited training data (Parameter Golf may have short training runs), routing quality suffers. K=1 is riskier than K=2 at small scale.

---

### 2.3 GShard / Mesh TensorFlow MoE (Lepikhin et al., 2021) — Distributed Scale
**Paper:** *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*  
**URL:** https://arxiv.org/abs/2006.16668  
**Who:** Google Brain / Google Research

**What it is:** Made MoE tractable across thousands of TPU cores. Key innovation: **every-other-layer** MoE placement (alternating dense and sparse layers). This halves the routing overhead and improves training stability.

**16MB relevance:** The alternating-layer strategy is directly applicable to tiny models. Instead of making every FFN layer sparse, alternate: dense→sparse→dense→sparse. This is more stable and often achieves comparable quality with better training convergence.

---

### 2.4 Mixtral 8x7B (Mistral AI, 2024) — Open-Source Proof of Concept
**Paper:** *Mixtral of Experts*  
**URL:** https://arxiv.org/abs/2401.04088  
**Who:** Albert Q. Jiang et al., Mistral AI

**What it is:** First widely-used open-source sparse MoE that demonstrated superiority over dense models of similar active-parameter count. Architecture:
- 8 experts per MoE layer, top-2 routing (K=2)
- 47B total parameters, ~12B active per token (25% activation rate)
- Outperforms LLaMA 2 70B on most benchmarks while using only 12B active params
- Speed at inference: comparable to a 12B dense model, not a 47B one

**Why it matters:** Established that K=2 routing is robust and can match much larger dense models. The 8 experts with 2 active is the **sweet spot** that has been replicated extensively.

**16MB relevance:** The architectural ratio matters: if 47B total : 12B active = 3.9x, then for 16MB: could store ~63MB of "effective capacity" in 16MB of active compute budget. Though at 16MB, the absolute expert sizes may be too small for meaningful specialization.

---

### 2.5 DeepSeekMoE (DeepSeek, 2024) — Granular Expert Specialization
**Paper:** *Towards Accurate and Lightweight Fully Transparent GPT — DeepSeekMoE*  
**URL:** https://arxiv.org/abs/2401.06066  
**Who:** DeepSeek AI (Damai Dai et al.)

**What it is:** Identified a fundamental flaw in vanilla MoE: with large experts and top-K routing, experts accumulate redundant knowledge. DeepSeekMoE fixes this with:
1. **Fine-grained expert segmentation:** Split N experts into mN smaller experts, activate mK from them. Same total compute, but finer-grained routing = less redundancy.
2. **Shared expert isolation:** Some experts are always active (shared knowledge), freeing routed experts to specialize more.

Result: DeepSeekMoE 2B (total) achieves performance comparable to 7B dense models.

**Later scaling:** DeepSeek-V3 (Dec 2024, arxiv 2412.19437) uses 671B total parameters, 256 routed experts + 1 shared expert per layer, activating ~37B parameters per token (~5.5% of total). Outperforms GPT-4 class models.

**16MB relevance:** The "shared experts" concept is directly applicable. In a 16MB sparse model, maintain 1–2 small "always-active" experts for universal language features, with 4–8 routed specialists. This ensures core linguistic competence even if routing fails during early training.

---

### 2.6 OLMoE (Allen AI / Ai2, 2024) — Fully Open 1B Active / 7B Total
**Paper:** *OLMoE: Open Mixture-of-Experts Language Models*  
**URL:** https://arxiv.org/abs/2409.02060  
**Who:** Niklas Muennighoff, Luca Soldaini, et al. (Allen Institute for AI)

**What it is:** 7B total parameters, 1B active per token (14.3% activation rate). Pretrained on 5 trillion tokens. Completely open: weights, training data, code, and logs all public.

Key findings from 63-page analysis:
- **High specialization confirmed:** OLMoE's routing shows clear expert specialization — different experts activate for different linguistic contexts
- **Token routing is not random:** Experts learn syntax/semantics distinctions, domain specialization, language-structure specialization
- Outperforms all models with similar active parameters, surpasses LLaMA 2 13B Chat with only 1B active params

**Why it matters for research:** This is the most rigorously documented MoE at small active-parameter scale. The 1B active / 7B total ratio means only 14% of parameters activate per token. Real-world proof that sparse routing adds genuine value beyond the larger storage footprint.

**16MB relevance:** If you store 7x the parameters of your target active size, you get OLMoE-equivalent capacity density. At 16MB, that means ~2.3MB active budget with 7B parameter-equivalent capacity. The tradeoff: you need ALL 7x in RAM, even if only 1x computes. For a 16MB model running on-device, this is usually fine (16MB fits in L2/L3 cache easily).

---

### 2.7 JetMoE-8B (MyShell AI, 2024) — Attention + FFN Both Sparse
**Paper:** *Reaching Llama2 Performance with 0.1M Dollars*  
**URL:** https://arxiv.org/abs/2404.07413  
**Who:** Zhen Guo et al., MyShell AI

**What it is:** Extended MoE sparsity to BOTH attention and FFN layers (unlike previous work that only sparsified FFN). Architecture:
- 8B total parameters, 2B active per token (25% activation)
- Attention experts: 8 attention heads route to 2 active heads per token
- FFN experts: standard MoE routing
- Trained on 1.25T tokens for $100K using 30,000 H100 GPU hours
- Outperforms LLaMA 2 7B

**Key innovation:** Sparse attention (also called Mixture-of-Attention) reduces the quadratic attention cost AND adds specialization to the attention mechanism itself.

**16MB relevance:** At 16MB, sparse attention over 2–4 heads (with 8 total) could halve the attention compute without reducing attention capacity. This is a high-leverage technique for tiny models where attention overhead is proportionally larger.

---

### 2.8 Auxiliary-Loss-Free Load Balancing (DeepSeek, 2024)
**Paper:** *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*  
**URL:** https://arxiv.org/abs/2408.15664  
**Who:** Damai Dai et al. (DeepSeek AI)

**What it is:** Traditional MoE uses an auxiliary loss to force balanced routing. Problem: this auxiliary loss interferes with the primary task loss, degrading model quality.

Solution: **Expert-wise bias on routing scores**, dynamically updated based on recent expert utilization. No gradient through the balancing mechanism.
- Validated on MoE models up to 3B parameters, 200B training tokens
- Achieves BOTH better performance AND better load balance vs. auxiliary-loss methods

**16MB relevance:** Critical. In short training runs (10 minutes), the auxiliary loss represents a larger fraction of total training signal. Eliminating it via bias-based balancing could meaningfully improve training quality on tight compute budgets.

---

### 2.9 Hash Layers (Roller et al., 2021) — No Learning Required
**Paper:** *Hash Layers for Large Sparse Models*  
**URL:** https://arxiv.org/abs/2106.04426  
**Who:** Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston (Facebook AI Research)

**What it is:** Replace the learned router with a **deterministic hash function**. Each token's identity (or position) is hashed to a specific expert. No routing parameters, no routing instability, no load imbalance problems.

Types of hashing:
- **Token ID hashing:** Route based on token vocabulary ID
- **Position hashing:** Route based on sequence position (mod N experts)
- **Random hashing:** Fixed random assignment of token IDs to experts

Findings: Hash routing performs **comparably to learned routing** in many settings, especially for smaller models and shorter training runs. The performance gap only appears with very long training (many tokens).

**Why this matters enormously for Parameter Golf:**
1. **Zero routing overhead:** No router parameters, no router compute
2. **Perfect load balance by design:** Hash ensures uniform distribution
3. **Training stability from step 1:** No routing instability in early training
4. **Works with 10-minute training:** Learned routing needs time to converge; hash routing doesn't

**16MB relevance:** For a 10-minute training run, hash routing may actually OUTPERFORM learned routing. The router has no time to learn good assignments, so a deterministic assignment that guarantees load balance and no gradient interference is preferable. This is a strong candidate for Parameter Golf's sparse architecture.

---

### 2.10 Parameter-Efficient MoE (for-ai research, 2023)
**Paper:** *Pushing Mixture of Experts to the Limit: Extremely Parameter-Efficient MoE for Instruction Tuning*  
**URL:** https://arxiv.org/abs/2309.05444  
**Who:** Ted Zadouri et al., for-ai research

**What it is:** Combines MoE with lightweight expert adapters (LoRA-style). Instead of full FFNs as experts, each expert is a small rank-decomposition matrix added to a shared base FFN.
- Update < 1% of an 11B model's parameters
- Outperforms standard PEFT methods
- Generalizes to unseen tasks

**16MB relevance:** This is the clearest path for sparse computation at extreme parameter budgets. Instead of 8 full FFN experts, have 1 shared base FFN + 8 lightweight expert adapters (each ~1/16th the base size). Total parameter cost: ~1.5x base size, with 8x routing granularity.

---

### 2.11 Task-Conditioned Routing (March 2026)
**Paper:** *Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers*  
**URL:** https://arxiv.org/search/cs?query=task+conditioned+routing+sparse&start=0 (March 2026, arXiv 2503.*)  
**Who:** Mynampati Sri Ranganadha Avinash

**What it is:** Shows that routing patterns in MoE are task-specific and consistent — same task types route to same experts. This means:
- Routing is learnable and meaningful
- Expert specialization emerges naturally
- Fine-tuning specific experts for specific tasks is feasible

**16MB relevance:** If your 16MB MoE has consistent routing signatures, you can do domain adaptation by fine-tuning only the relevant experts, not the full model. Huge for post-training specialization.

---

### 2.12 Path-Constrained MoE (March 2026)
**Paper:** *Path-Constrained Mixture-of-Experts*  
**URL:** Submitted March 18, 2026 (arXiv 2503.*)  
**Who:** Zijin Gu, Tatiana Likhomanenko, Vimal Thilak, Jason Ramapuram, Navdeep Jaitly

**What it is:** Recent 2026 work constraining routing paths — instead of free top-K routing, experts form constrained paths through the network. Improves routing consistency and reduces the routing search space.

This is part of a 2025–2026 trend toward **structured routing** — routing that's predictable, interpretable, and hardware-efficient.

---

## 3. How MoE Actually Works: Technical Deep Dive

### The Router Mechanism
For each input token `x` (a d-dimensional vector), the router computes:
```
router_output = softmax(W_router @ x)  # n_experts-dimensional distribution
top_k_indices = argsort(router_output)[-K:]  # top K experts
weights = router_output[top_k_indices]
weights = weights / sum(weights)  # normalize top-K weights

output = sum(weight_i * Expert_i(x) for i in top_k_indices)
```

`W_router` is a `(d_model × n_experts)` matrix. For d_model=256, n_experts=8: that's 256×8=2,048 parameters. Negligible.

### Load Balancing
Without load balancing, the router collapses: one expert gets most tokens (it's better from step 1, gets more gradient, gets better, attracts more tokens...). Two solutions:

**A. Auxiliary loss (original approach):**
```python
# Encourage uniform load across experts
router_probs_mean = mean over batch of router_probs
load_balance_loss = n_experts * sum(f_i * P_i for i in experts)
# where f_i = fraction of tokens dispatched to expert_i
# and P_i = mean router probability for expert_i
total_loss = task_loss + alpha * load_balance_loss
```

**B. Bias-based loss-free balancing (DeepSeek 2024 — recommended):**
```python
# No gradient; just adjust routing scores by a bias
routing_score = softmax(W_router @ x + expert_bias)
# Update expert_bias based on recent utilization:
if expert_i.recent_load > target_load:
    expert_bias[i] -= gamma  # discourage overloaded experts
else:
    expert_bias[i] += gamma  # encourage underloaded experts
```

### Capacity Factor
To handle uneven token distribution during training, each expert has a "capacity" = (tokens_per_batch / n_experts) × capacity_factor. Tokens beyond capacity are dropped (or passed through unchanged in Switch Transformer). Capacity factor of 1.0–1.25 is standard.

---

## 4. Why Sparse Compute Helps Even at 16MB

### Argument 1: More Capacity in Same Size
A 16MB sparse model with 8 experts can represent 8 different "modes" of knowledge in the same byte footprint as one 16MB dense model. Token routing ensures the right mode activates.

**Concrete math:**
- Dense 16MB: 4M params (fp32), all active for every token
- Sparse 16MB: 4M params total, 0.5M active per token (8 experts, top-1 routing)
- Same memory, same inference FLOPs — but 8x the capacity for diverse inputs

### Argument 2: Faster Training via Parallelism
During training, different experts can be updated by different tokens simultaneously. In a batch of 1024 tokens, with 8 experts and balanced routing, each expert receives ~128 tokens and updates independently. This is 8x more parameter-efficient than a dense model that sees all 1024 tokens update the same 4M parameters.

### Argument 3: Specialization = Better Compression
A sparse model can "overfit usefully": each expert specializes in a subset of the input space. This is better compression than a dense model trying to represent all patterns in all parameters. For a 16MB model trying to cover English language, arithmetic, and code — a 3-way split into specialists might work better than a generalist.

### Argument 4: Hash Routing Solves Training Time Problem
For a 10-minute training run, there's no time for a learned router to converge. Hash routing assigns tokens deterministically — computation starts immediately and efficiently. Training loss benefits from specialization from step 1.

---

## 5. Who's Building This (2024–2026)

| Organization | Model | Total Params | Active Params | Notes |
|---|---|---|---|---|
| **Google** | Switch Transformer | ~1T | ~1/N per token | Foundation paper, 2021 |
| **Mistral AI** | Mixtral 8x7B | 47B | ~12B (K=2) | Best open model, 2024 |
| **DeepSeek AI** | DeepSeek-V3 | 671B | ~37B (5.5%) | SOTA frontier, Dec 2024 |
| **Allen AI** | OLMoE-1B-7B | 7B | 1B (14%) | Fully open, 2024 |
| **MyShell AI** | JetMoE-8B | 8B | 2B (25%) | Sparse attention too |
| **Skywork** | Skywork-MoE | 146B | N/A | 16 experts, upcycled |
| **Meta** | FBAI Hash Layers | Research | N/A | Deterministic routing |
| **Mistral AI** | Mixtral 8x22B | 141B | ~39B | Larger variant |
| **for-ai** | PE-MoE | 11B | < 1% updated | Adapter-based |
| **xAI** | Grok (rumored MoE) | Undisclosed | Undisclosed | Industry adoption |

**2026 trend:** Path-constrained MoE (March 2026), structured routing, sparse adapters, and MoE for multimodal tracking are all active research directions. The field is maturing from "does it work?" to "how do we make it reliable and efficient?"

---

## 6. Parameter Golf Specific: 16MB / 10-Minute Application Guide

### Recommended Architecture: Mini-MoE with Hash Routing

```
Total budget: 16MB (≈ 4M parameters at fp32, or ≈ 8M at fp16)

Proposed split:
- Embedding layer (shared): 512 vocab × 64 dims = 32K params (~128KB)
- Attention layers (dense, 4 layers): 4 × 64² × 4 = 65K params (~256KB)  
- MoE FFN layers (4 layers, 8 experts each):
    - Shared base FFN: 64 × 256 × 2 = 32K per layer = 128K total
    - 8 lightweight expert adapters (rank-4 LoRA-style):
      64 × 4 × 2 = 512 params per expert, ×8 = 4K per layer = 16K total
    - Total MoE: ~144K params per layer, ×4 = ~576K params
- LM head (shared with embedding): ~32K params
- Total estimate: ~770K parameters, well within 16MB even at fp32
```

With this design:
- Each token activates: attention layers + 1 shared FFN + 1 expert adapter
- 7/8 expert adapters are "off" per token
- Training: all 8 experts receive gradient via hash routing (balanced from step 1)
- Memory footprint: fits in ~3MB at fp32 — leaves room for larger dimensions

### Scaling up to fill 16MB:
- Increase d_model to 256, FFN dim to 1024
- Use 8 experts with 32-dim adapters each
- Add 6–8 layers
- ~4M total parameters → 16MB at fp32

### Routing Strategy for 10-Minute Training:
1. **First choice: Hash routing** (deterministic, no convergence needed)
   - Token ID mod 8 → expert assignment
   - Perfect load balance immediately
   - Zero router parameters

2. **Second choice: Fixed random routing** 
   - Sample random assignment at init, freeze it
   - Slightly less structured than hash, but very fast

3. **Third choice: Learned routing with loss-free balancing**
   - Use DeepSeek's bias-based balancing
   - Add warm-up phase where routing is random (first 10% of steps)
   - Switch to learned routing after the router has basic training signal

### What NOT to Do (for 16MB / 10 minutes):
- **Don't use auxiliary loss balancing** in short runs — it competes with task loss and hurts quality when both are weak
- **Don't use K=1 routing without load guarantees** — early in training, all tokens may route to the same expert
- **Don't use too many experts** — with 8M parameters, 64 experts means 125K params per expert. Each expert is too small to specialize. Max experts ~8–16 at this scale.
- **Don't apply MoE to attention** unless you have >4M params in attention alone (JetMoE works at 8B; at 16MB, sparse attention adds complexity without enough capacity to specialize)

### Training Tips for MoE at Tiny Scale:
1. **Capacity factor = 1.5** (more generous than standard 1.0-1.25) to handle early training imbalance
2. **Expert dropout** (randomly drop experts with p=0.1 during training) prevents over-reliance on specific experts
3. **Initialize all experts identically** — let training differentiate them, starting from a good base
4. **Use a smaller LR for the router** than for expert weights — the router is a smaller network and can overfit its assignment quickly

---

## 7. Key Insights and Synthesis

### The Fundamental Trade-off at 16MB
Traditional MoE wisdom: "more total params = better, active params control cost." At 16MB, this inverts slightly:
- **Total params ARE the cost** (memory, loading time)
- Active params control inference FLOPs (less important at this scale; FLOPs are tiny anyway)
- The real benefit is **capacity diversity** — routing tokens to specialized subnetworks

**The real question is:** Does a 16MB sparse model beat a 16MB dense model?

Evidence says **yes, with caveats:**
- **Yes** if you train for many tokens (millions+) — experts learn to specialize
- **Probably yes** with hash routing even with short training — deterministic specialization from day 1
- **Maybe** with learned routing and very short training — depends on init and learning rate

### The Densing Law Context (Dec 2024)
*Densing Law of LLMs* (arxiv 2412.04315) showed that model capacity density (quality per parameter) doubles every ~3 months as of late 2024. Sparse models are a major driver of this: they pack more effective capacity per parameter. For Parameter Golf, this means the efficiency ceiling keeps rising and sparse architectures are at the frontier of that improvement.

### Why This Is Bigger Than "Just An Architecture Choice"
Sparse compute changes the training dynamics fundamentally:
- **Different tokens train different experts** → each expert is effectively trained on a semantically coherent subset of data
- **Expert specialization is emergent** → the model self-organizes its knowledge
- **This is a form of implicit data curriculum** → routing = self-supervised specialization

For a model trying to learn language with limited capacity (16MB), automatic self-specialization could be the difference between coherent outputs and averaged-out mediocrity.

---

## 8. Recent 2025–2026 Developments

### 2025
- **OLMoE v2 updates** (March 2025): Allen AI updated weights; analysis showed stable routing even after months of deployment
- **Loss-free balancing** adopted broadly — DeepSeek-V3's auxiliary-loss-free approach is now considered best practice
- **MoE for multimodal:** Sparse-Dense Mixture of Experts adapters for multi-modal tracking (March 2026, arxiv 2503.*)
- **HuggingFace MoE first-class support** (February 2026): MoE models are now fully native in the `transformers` library with dedicated efficient implementations

### 2026
- **Path-Constrained MoE** (March 2026): New structured routing approaches that make routing more predictable and hardware-efficient
- **Task-Conditioned Routing Signatures** (March 2026): Routing fingerprints are now studied for interpretability and targeted fine-tuning
- **Cascaded MoE** (March 2026): Hierarchical MoE where local and global experts cascade — first applied to graph problems, applicability to language unclear but promising

### What's Coming (2026 projections):
- **On-device MoE** — Apple/Qualcomm working on hardware that natively supports sparse routing (rumored in mobile AI chip roadmaps)
- **Hash routing formalization** — FAIR's deterministic routing work being extended to position-aware and semantic hashing
- **Expert merging/pruning** — Post-training compression of MoE by merging underused experts (reduces storage without retraining)

---

## 9. The Critical Question for Parameter Golf

**"Should we build a sparse 16MB model instead of a dense one?"**

### Yes if:
- Training data is diverse (multiple domains/tasks) — experts can specialize
- Training is longer than ~50K steps — enough for routing to learn
- You want a model that can be specialized post-training per expert
- You're using hash routing (no convergence needed)

### No/Maybe if:
- Training is extremely short (< 10K steps) AND using learned routing
- The task is highly uniform (e.g., pure code completion) — no benefit to specialization
- You cannot tolerate the 10–20% parameter overhead for routing infrastructure

### The Asymmetric Bet:
At 16MB, the downside of sparse routing is small (slightly harder to train, slightly more complex code). The upside is potentially 4–8x the effective model capacity. This is a bet worth taking.

**Recommended approach:**
1. Start with 4 experts, hash routing, alternating dense/sparse layers
2. Train and compare against dense baseline at matched param count
3. If routing shows specialization (experts activate non-uniformly for different inputs), scale to 8 experts
4. Add shared experts (1 always-active, 7 routed) for stability

---

## 10. Reference URLs

| Resource | URL |
|---|---|
| Switch Transformer (Google, 2021) | https://arxiv.org/abs/2101.03961 |
| Sparsely-Gated MoE (Shazeer 2017) | https://arxiv.org/abs/1701.06538 |
| Mixtral of Experts (Mistral 2024) | https://arxiv.org/abs/2401.04088 |
| DeepSeekMoE (DeepSeek 2024) | https://arxiv.org/abs/2401.06066 |
| OLMoE (Allen AI 2024) | https://arxiv.org/abs/2409.02060 |
| JetMoE (MyShell 2024) | https://arxiv.org/abs/2404.07413 |
| Auxiliary-Loss-Free Balancing | https://arxiv.org/abs/2408.15664 |
| Hash Layers (FAIR 2021) | https://arxiv.org/abs/2106.04426 |
| Parameter-Efficient MoE | https://arxiv.org/abs/2309.05444 |
| DeepSeek-V3 Technical Report | https://arxiv.org/abs/2412.19437 |
| Densing Law of LLMs | https://arxiv.org/abs/2412.04315 |
| Skywork-MoE Training Techniques | https://arxiv.org/abs/2406.06563 |
| HuggingFace MoE Explainer | https://huggingface.co/blog/moe |
| HuggingFace MoE in Transformers (Feb 2026) | https://huggingface.co/blog/moe-transformers |
| Path-Constrained MoE (March 2026) | https://arxiv.org/search/?query=Path-Constrained+Mixture-of-Experts&searchtype=all |
| Task-Conditioned Routing (March 2026) | https://arxiv.org/search/?query=Task-Conditioned+Routing+Signatures&searchtype=all |

---

## 11. Summary Table: Techniques at a Glance

| Technique | Activation % | Training Complexity | 16MB Suitability | Notes |
|---|---|---|---|---|
| Switch Transformer (Top-1) | 1/N | Medium | ★★★★☆ | Good; risk of routing collapse |
| Top-2 MoE (Mixtral-style) | 2/N | Medium-High | ★★★★★ | Best balance of stability + capacity |
| Hash Routing | 1/N | Low | ★★★★★ | Best for short training runs |
| DeepSeekMoE (fine-grained) | mK/mN | High | ★★★☆☆ | Complex; overkill at 16MB |
| Shared+Routed Experts | 1+K/N | Medium | ★★★★☆ | Stability via shared experts |
| Adapter-based MoE | ~1% | Low | ★★★★★ | Parameter-efficient; works tiny |
| Loss-Free Balancing | N/A | Low | ★★★★★ | Use instead of aux loss always |
| Sparse Attention MoE | 2/8 heads | High | ★★☆☆☆ | Not worth it below 8M params |

---

*Research compiled March 2026 for the Parameter Golf initiative. Focus: 16MB model footprint, 10-minute training, maximum quality per byte.*
