# Parameter Golf — Strategic Intelligence Analysis
**Agent:** Achilles ⚔️ | **Date:** March 24, 2026 | **Classification:** Strategic — Internal Use

---

## Executive Summary

The Parameter Golf competition is moving at breakneck speed — from 1.2244 bpb (baseline) to 1.1228 bpb (leaderboard #1) in **5 days**, with several pending PRs claiming sub-1.11 scores. The real frontier is already at **~1.067 bpb** (GEPA architecture + AdamW TTT), though these submissions may face legality scrutiny.

**Bottom line:** This is not a competition you win by thinking for 37 days. It's a compounding game. Every day we wait, the frontier drops further. Our best shot is to **ship fast, then iterate** — NOT plan a moonshot.

---

## 1. Competitive Gap Analysis

### Leaderboard Progression (March 18–22, 2026)

| Date    | Score (bpb) | Improvement  | Technique                              |
|---------|-------------|--------------|----------------------------------------|
| Mar 18  | 1.2244      | Baseline     | 9L, 512d, tied embeddings              |
| Mar 18  | 1.2197      | -0.0047      | FP16 embedding + LR tuning             |
| Mar 19  | 1.1925      | -0.0272      | Sliding window eval                    |
| Mar 19  | 1.1748      | -0.0177      | Muon WD + 10L + spectral init          |
| Mar 19  | 1.1556      | -0.0192      | SmearGate + OrthoInit + Muon WD        |
| Mar 20  | 1.1428      | -0.0128      | Int5-MLP + BigramHash                  |
| Mar 20  | 1.1271      | -0.0157      | XSA4 + EMA + Int6                      |
| Mar 21  | 1.1248      | -0.0023      | Partial RoPE + LN Scale + XSA4         |
| Mar 22  | 1.1228      | -0.0020      | GPTQ-lite + EMA + warmdown3500         |

### Pending PRs (Not Yet Verified/Leaderboarded)

| PR     | Score (bpb) | Key Innovation                         |
|--------|-------------|----------------------------------------|
| #595   | 1.1100      | Standard AdamW TTT (legal)             |
| #593   | 1.1171      | Full GPTQ + LeakyReLU² (no TTT)        |
| #442   | 1.1027      | AdamW TTT (legal, 3-seed)              |
| #462   | 1.0672      | GEPA arch + AdamW TTT (legal concerns) |
| #573   | 1.0523      | Multi-pass streaming score-first TTT   |
| #518   | 1.0622      | LeakyReLU² + Cosine TTT 50ep           |

### Rate of Improvement

**First 5 days:** -0.1016 bpb (1.2244 → 1.1228)
**Daily average:** -0.0203 bpb/day (decelerating)
**Last 2 days:** -0.0043 bpb (slowing rapidly on training-only techniques)

**Key insight:** Training-level improvements (architecture, quantization, optimizer) are plateauing around **1.12 bpb** on the leaderboard. The jump to sub-1.10 comes from **Test-Time Training (TTT)** — a fundamentally different category. The real question isn't "how low can training go?" but "what's the ceiling of legal TTT?"

### Plateau Projection

- **Without TTT:** ~1.10–1.11 bpb is the likely ceiling for training-only approaches
- **With legal TTT:** ~1.06–1.08 bpb is achievable; possibly ~1.04–1.05 with aggressive innovation
- **Projected winning score by April 30:** **1.02–1.06 bpb**
- **Why:** TTT is still early (AdamW switch happened March 22). 37 days of optimization on TTT techniques + better base models = huge room for improvement

---

## 2. Technique Saturation Analysis

### SATURATED — Everyone Has These (Low Marginal Value)

| Technique                          | Status      | Why                                      |
|------------------------------------|-------------|------------------------------------------|
| 11 layers, 512 dim                 | Standard    | Every top submission uses this            |
| Muon optimizer                     | Standard    | Universal adoption since ~Mar 19          |
| Sliding window eval (stride 64)    | Standard    | ~0.03 free bpb, everyone has it           |
| Tied embeddings                    | Standard    | Baseline, universal                       |
| Int6 quantization                  | Standard    | Every submission uses int6 weights         |
| EMA (decay ~0.997)                 | Standard    | Replaced SWA everywhere                   |
| BigramHash + SmearGate             | Standard    | Appears in all top-10 submissions          |
| U-Net skip connections             | Standard    | Found in all competitive submissions       |
| ReLU² activation                   | Standard    | Universal in top submissions               |

### EMERGING — Gaining Adoption (High Marginal Value Right Now)

| Technique                    | Status    | Expected Gain |
|------------------------------|-----------|---------------|
| Full GPTQ (Hessian-aware)    | Emerging  | 0.004–0.006   |
| LeakyReLU(0.5)²              | Emerging  | 0.001–0.003   |
| Partial RoPE (16/64 dims)    | Emerging  | 0.002–0.003   |
| LN Scale by depth            | Emerging  | 0.001–0.002   |
| GEPA architecture search     | Emerging  | 0.010–0.030   |
| AdamW TTT (legal, score-first)| Emerging | 0.020–0.050   |

### UNEXPLORED — High Opportunity (Competitors Haven't Caught On)

| Technique                          | Expected Gain | Why Unexplored                          |
|------------------------------------|---------------|-----------------------------------------|
| **qTTT** (query-only TTT)         | 0.003–0.010   | Paper dropped recently; enables 2-3× more TTT epochs |
| **LaCT** (Large Chunk TTT)        | 0.005–0.015   | ICLR 2026 Oral; fixes TTT throughput    |
| **Mousse optimizer**               | 0.003–0.008   | arXiv:2603.09697; curvature-aware Muon  |
| **Entropy-coded weights**          | 0.003–0.008   | Saves 1-2 MB → more params              |
| **HybridNorm**                     | 0.002–0.006   | Trivially low complexity, untested      |
| **In-Place TTT w/ NTP objective**  | 0.003–0.010   | ICLR 2026 Oral; may fix TTT ceiling     |
| **NuMuon**                         | 0.002–0.006   | Nuclear-norm Muon; better compression   |
| **Differential Attention**         | 0.005–0.015   | High complexity but high gain           |
| **RWKV architecture**              | Unknown       | Zero recurrent weight overhead; untested |
| **Meta-TTT (FOMAML)**              | Unknown       | Mentioned in comments, untested         |

---

## 3. Risk-Reward Matrix

| Technique                  | Difficulty (1-5) | Expected bpb Gain | Failure Risk | Time to Implement | ROI Score |
|----------------------------|:-----------------:|:------------------:|:------------:|:------------------:|:---------:|
| **AdamW TTT (legal)**      | 2                 | 0.020–0.050        | Low          | 1–2 days           | ⭐⭐⭐⭐⭐ |
| **qTTT**                   | 3                 | 0.003–0.010        | Medium       | 2–3 days           | ⭐⭐⭐⭐  |
| **GEPA arch search**       | 4                 | 0.010–0.030        | Medium       | 3–5 days           | ⭐⭐⭐⭐  |
| **Full GPTQ**              | 3                 | 0.004–0.006        | Low          | 1–2 days           | ⭐⭐⭐⭐  |
| **Mousse optimizer**       | 2                 | 0.003–0.008        | Low          | 1 day              | ⭐⭐⭐⭐  |
| **Entropy-coded weights**  | 4                 | 0.003–0.008        | High         | 3–5 days           | ⭐⭐⭐   |
| **HybridNorm**             | 1                 | 0.002–0.006        | Low          | <1 day             | ⭐⭐⭐⭐  |
| **LaCT TTT**               | 4                 | 0.005–0.015        | High         | 3–5 days           | ⭐⭐⭐   |
| **Differential Attention** | 5                 | 0.005–0.015        | High         | 5–7 days           | ⭐⭐     |
| **RWKV architecture**      | 5                 | Unknown            | Very High    | 7–14 days          | ⭐       |
| **MoE**                    | 5                 | Unknown            | Very High    | 7–14 days          | ⭐       |
| **BitNet**                 | 4                 | Unknown            | High         | 5–10 days          | ⭐⭐     |
| **Depth recurrence**       | 3                 | Unknown            | High         | 2–4 days           | ⭐⭐     |
| **Meta-TTT (FOMAML)**      | 5                 | Unknown            | Very High    | 5–10 days          | ⭐       |
| **NuMuon**                 | 2                 | 0.002–0.006        | Low          | 1 day              | ⭐⭐⭐⭐  |

---

## 4. Our Unique Advantages

### What We Have That PhD ML Researchers Don't

1. **Multi-model AI assistance (7+ models)** — Most competitors are solo developers or small teams. We have a *department*:
   - Codex for rapid code generation and architecture exploration
   - DeepSeek for deep research on papers
   - Gemini for long-context synthesis across dozens of PRs
   - Opus for strategic judgment on what to try next
   - Grok for contrarian analysis of what competitors are getting wrong
   - This means we can explore 3-5× more ideas per day than a solo competitor

2. **Auto-research optimization loops** — We can run systematic experiment pipelines:
   - Research paper → evaluate relevance → spec implementation → code → test → iterate
   - Most competitors are doing this manually. We can automate the exploration
   - The key: we should be exploring *technique combinations*, not just individual techniques

3. **Systems engineering expertise** — The 10-minute constraint is as much a systems problem as an ML problem:
   - FP8 training, FlashAttention 3, custom CUDA kernels, zstd/lzma tuning
   - Parallel Muon (83ms/step) — every ms saved = more steps = better model
   - Most ML researchers ignore systems. We don't

4. **Open source intelligence advantage** — 419 PRs worth of experimentation data is public. We can:
   - Track what's working and what isn't across the entire community
   - Reverse-engineer the best combinations
   - Identify which techniques are synergistic (GEPA + TTT is superadditive)
   - Avoid dead ends that others have already explored

### How to Leverage These Advantages

**Don't compete on ML research depth.** The GEPA architecture (1.0672) was found by evolutionary search, not by a PhD's intuition. Our strength is systems engineering + rapid iteration + multi-model exploration.

**Strategy: "Best of" stacking.** Take the best proven components and stack them:
1. Best base architecture (from GEPA or copy the current frontier)
2. Best quantization (Full GPTQ, not GPTQ-lite)
3. Best TTT protocol (AdamW, legal, score-first)
4. Best systems optimization (Parallel Muon, FP8 if fixable)

This is a *combinatorial* optimization problem, not a research problem. Our multi-model approach is perfectly suited for it.

---

## 5. Speed vs Innovation Tradeoff

### The Optimal Strategy: **Ship Fast, Stack Later**

**This is NOT a moonshot competition.** Here's why:

1. **The rules reward compounding.** Each submission must beat the current best by ≥0.005 bpb. This means you need a *sequence* of improvements, not one big leap.

2. **Information flow is asymmetric.** Every submission teaches the community something. By submitting early, you:
   - Get verified results (pending PRs might be wrong)
   - Establish your name on the leaderboard
   - Get community feedback
   - Signal competence (important for OpenAI hiring pipeline)

3. **TTT is still immature.** AdamW TTT was discovered 2 days ago. The optimization landscape is wide open. Getting a working TTT pipeline NOW means 35+ days to optimize it.

4. **Compute is finite.** 8×H100 at $20/hr = $14,000/month. Moonshots that fail waste compute. Incremental improvements that succeed build on each other.

### Recommended 37-Day Plan

**Phase 1: Ship (Days 1-5) — Target: ~1.11 bpb**
- Fork the best current non-TTT submission (Full GPTQ + LeakyReLU², PR #593 at 1.1171)
- Add legal AdamW TTT (proven 0.020+ gain)
- Ship immediately for leaderboard position
- Expected: ~1.095–1.105 bpb

**Phase 2: Optimize (Days 6-15) — Target: ~1.08 bpb**
- Stack qTTT (query-only TTT) for 2-3× more TTT epochs
- Add HybridNorm (trivial complexity, free gain)
- Add Mousse optimizer (12% more effective training)
- Tune TTT hyperparameters systematically (LR, epochs, layer groups)
- Test NuMuon for compression improvement
- Expected: ~1.07–1.09 bpb

**Phase 3: Push (Days 16-30) — Target: ~1.05 bpb**
- GEPA architecture search (if we can afford the compute)
- Entropy-coded weights to free artifact space for more params
- Test In-Place TTT with NTP objective (ICLR 2026 Oral paper)
- Explore LaCT for TTT throughput improvement
- Systematic hyperparameter sweeps on the full stack
- Expected: ~1.04–1.06 bpb

**Phase 4: Lock-in (Days 31-37) — Final submission**
- Reproduce best result across 3+ seeds
- Clean up code for reproducibility
- Write comprehensive README
- Verify artifact size under 16MB
- Submit final entry

---

## 6. What Would It Take to Win?

### Projected Winning Score: **1.02–1.04 bpb**

Based on:
- Current frontier at 1.0523 (PR #573, pending legality review)
- 37 days of community optimization remaining
- TTT is still in early exploration (AdamW TTT is 2 days old)
- GEPA architecture search is underutilized
- Multiple high-EV techniques remain untested

### Techniques Needed to Win

1. **GEPA-discovered architecture** (or equivalent automated search)
   - SwiGLU FFN, optimized depth, novel skip patterns
   - GEPA architectures show superadditive gains with TTT

2. **Full Hessian GPTQ + QAT** (proven -0.005 bpb over GPTQ-lite)

3. **Legal TTT with AdamW** (proven -0.020 to -0.050 bpb)
   - Score-first backward-looking protocol
   - Optimized epochs, LR schedule, per-layer groups

4. **qTTT or LaCT** (enables 2-3× more TTT adaptation)
   - Query-only TTT reduces per-epoch cost from ~15s to ~4-6s
   - More epochs within the same 10-min eval budget

5. **Entropy-coded weights** (frees 1-2 MB for more parameters)
   - More params = more capacity = lower bpb
   - The 16MB constraint is the real bottleneck

6. **Systems optimization** (FP8, Liger-Kernel, Turbo-Muon)
   - Every 1% speedup = ~7 more training steps
   - At frontier, 7 steps = ~0.001 bpb

### Compute Budget Required

| Phase          | Runs Needed | Cost Estimate |
|----------------|-------------|---------------|
| Exploration    | 50-100 runs | $5,000–10,000 |
| Optimization   | 30-50 runs  | $3,000–5,000  |
| Final testing  | 10-20 runs  | $1,000–2,000  |
| **Total**      | **90-170**  | **$9,000–17,000** |

**Note:** Apply for OpenAI's $1M compute grant. This competition is explicitly designed to scout talent — showing initiative matters.

---

## Key Recommendations (Priority Order)

1. **START WITH TTT.** Fork PR #442 (legal AdamW TTT at 1.1027) or PR #595 (1.1100) and combine with the best quantization (Full GPTQ from PR #593). This alone should get us to ~1.09–1.10.

2. **REQUEST COMPUTE GRANT IMMEDIATELY.** OpenAI is giving away $1M in RunPod credits. Apply now.

3. **USE GEPA or similar automated search.** The winning architecture was found by evolutionary search, not human design. Our multi-model AI advantage means we can run evolutionary search smarter than most.

4. **FOCUS ON LEGAL TTT OPTIMIZATION.** This is the highest-EV unexplored space. qTTT, LaCT, In-Place TTT with NTP objective — all could be breakthroughs.

5. **DON'T WASTE TIME ON EXOTIC ARCHITECTURES.** RWKV, MoE, BitNet are high-risk, high-time, and probably won't beat the transformer + TTT combo in 16MB.

6. **SUBMIT EVERY 3-5 DAYS.** Treat each submission as a checkpoint. Get verified. Build reputation. Learn from community feedback.

7. **MONITOR THE LEGALITY DEBATE.** PR #573 (1.0523) and PR #462 (1.0672) may face legality challenges. If "backward-looking TTT" gets more restricted, the winning score will be higher (~1.08-1.10). If it stays loose, ~1.02-1.04.

---

## Appendix: Competitive Intelligence — Key Players

| Player           | Best Score | Approach                     | Strength          |
|------------------|-----------|------------------------------|-------------------|
| signalrush       | 1.1228    | GPTQ-lite + EMA + warmdown   | Incremental tuning|
| jfprincz         | 1.1248    | Partial RoPE + LN Scale      | Architecture      |
| sjp611           | 1.1027    | AdamW TTT                    | TTT pioneer       |
| JoeProAI         | 1.0672    | GEPA + TTT (OpenClaw agents) | AI-assisted search|
| LoquiAuris       | 1.1100    | Standard TTT                 | TTT optimization  |
| abaybektursun    | 1.1171    | Full GPTQ + LeakyReLU²       | Quantization      |
| saml212          | —         | XSA-all + pruning            | Aggressive XSA    |

**Note:** JoeProAI is using OpenClaw (our own platform!) with AI agents. This validates our approach but also means others are catching up to our tooling advantage.

---

*Achilles ⚔️ — Strike fast, strike smart. The window is closing.*
