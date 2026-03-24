# Depth Recurrence in Parameter-Constrained Transformers: What Works, What Doesn't, and Why

**PR #363 | Non-Record Submission (Research Contribution)**
**Author:** Evangeline Kamin ([@evangelinehelsinki](https://github.com/evangelinehelsinki), itsmeaura/Aura on Discord)
**Base:** PR #325 by Aum08Desai (1.1462 bpb)
**Duration:** 4 days, ~35 runs across 8xH100 SXM bare metal, 2xH100, RTX 3070, and A4500 pods
**Final best (looped):** 1.1787 bpb sliding window | **Flat comparison:** 1.1648 bpb | **Gap:** +0.025 bpb

---

## The Short Version

I spent four days trying to make depth-recurrent transformers competitive in Parameter Golf. They aren't. A flat 11-layer model beats a looped 3x3 model by 0.025 bpb on identical hardware with identical tricks. Three independent researchers (me, Frosty40, and Ciprian-Florin Ifrim) arrived at the same conclusion from different starting points.

But the failure is informative, and two findings survived: **Noisy QAT** (a training technique that collapses quantization error amplification through recurrence from 0.37 bpb to 0.002 bpb) and **the 3x3 > 2x5 loop configuration** (more unique blocks with fewer repeats beats fewer blocks with more repeats, on every metric).

This document covers 250+ hours of experiments, 12 negative results with specific numbers, and an honest post-mortem on why the "save parameters through weight sharing, spend them on more capacity" thesis doesn't work under competition constraints. If you're considering depth recurrence for Parameter Golf, read this first. It will save you days.

---

## Table of Contents

1. [How I Got Here](#how-i-got-here)
2. [The Architecture](#the-architecture)
3. [What Worked](#what-worked)
4. [The Controlled Comparison](#the-controlled-comparison)
5. [Why Recurrence Fails at This Scale](#why-recurrence-fails-at-this-scale)
6. [The Full Experiment Log](#the-full-experiment-log)
7. [Negative Results (All 12)](#negative-results-all-12)
8. [What Might Work With More Compute](#what-might-work-with-more-compute)
9. [Acknowledgments](#acknowledgments)
10. [Reproducing These Results](#reproducing-these-results)

---

## How I Got Here

On Day 0, I deployed 15 research agents to mine papers from labs in 12 countries (Chinese, Japanese, Korean, Israeli, Indian, and others) looking for approaches nobody else in the competition was trying. Depth recurrence kept coming up: Samsung's TRM, Alibaba's Huginn, Relaxed Recursive Transformers, Mixture-of-Recursions. The appeal was obvious for a size-constrained competition. If you share weights across loop iterations, you get more effective depth per byte of artifact. My first looped model on a 3070 hit 1.5630 bpb with only 6.1M params and a 4.1MB artifact. 64% fewer parameters than the baseline. I remember seeing that artifact size and thinking "this is going to crush everyone."

It didn't.

The gap between "this architecture is parameter-efficient" and "this architecture is competitive in a 10-minute training race" turned out to be enormous. But figuring out exactly *why* it's enormous, and documenting every attempt to close it, is (I think) more useful to the community than another 0.001 improvement on the standard 11L stack.

### Background on me

I'm a high school student in Phoenix. I work as a waitress. I have no formal ML background. My compute budget for this competition was about $30 out of pocket plus $170 in Hyperbolic referral credits (thank you to whoever started the referral chain in the Discord, and sorry to Hyperbolic's VCs). My development hardware ranged from an RTX 3070 to bare metal 8xH100 SXM5 nodes rented by the hour. I mention this not for sympathy points but for context: every experiment had a real dollar cost, which shaped which experiments I ran and how carefully I designed them.

### The research pipeline

To compensate for limited compute, I built an aggressive research pipeline:
- **15 parallel research agents** scanning recent papers, filtering for parameter-efficient training techniques relevant to the 16MB/10min constraint
- **A 26-model code review gauntlet** where I ran my training script through GPT-5, Gemini 3.1 Pro, DeepSeek V3.2, O3 Deep Research, Kimi K2.5, Claude Opus, and 20 others. This caught a critical `global _QAT_ACTIVE` bug (QAT may have never been running), env var name mismatches, torch.compile recompilation stalls, and redundant zero_grad calls.
- **Systematic PR mining**: I fetched and analyzed all 600+ competition PRs, spawning subagents to deep-dive the top submissions. This is how I tracked the converging "meta stack" and identified which techniques were worth testing on my architecture.

---

## The Architecture

### The Thesis

Depth recurrence (reusing the same transformer blocks multiple times in a forward pass) has a long lineage: Universal Transformer (Dehghani et al., 2019), Huginn (Alibaba, 2025), Samsung TRM, and several Parameter Golf submissions including PR #325 by Aum08Desai. Share weights across loop iterations, get more effective depth per byte of artifact. In a competition with a 16MB cap, this should be a cheat code.

### Middle-Cycle Layout

PR #325 introduced a "Middle-Cycle" architecture that splits layers into three sections:

```
[Stem blocks]  →  [Core blocks × R repeats]  →  [Tail blocks]
```

- **Stem blocks**: Unique layers processing raw embeddings. Not shared.
- **Core blocks**: Shared layers that execute R times. This is where the parameter savings come from.
- **Tail blocks**: Unique layers producing final representations. Not shared.
- **U-Net skip connections**: Stem outputs added (with learnable weights) to tail block inputs.

I tested two configurations extensively:

| Config | Stem | Core | Repeats | Tail | Effective Depth | Unique Blocks |
|--------|------|------|---------|------|-----------------|---------------|
| **3x3** | 3 | 3 | 3 | 3 | 12 | 9 |
| **2x5** | 2 | 2 | 5 | 2 | 16 | 6 |

The 2x5 was my starting point (forked from PR #325). The 3x3 came from studying Frosty40's Frugendorff architecture (PR #499), which used 6 blocks × 2 repeats. More on why 3x3 won later.

Both configs used 640d model dimension, 8 attention heads with 4 KV heads (GQA), 3x MLP expansion, tied embeddings with vocab 1024, and SmearGate + BigramHash + RoPE from the PR #325 base.

### Where this sits in the competition

The meta as of ~640 PRs is flat 11-12 layer architectures at 512d. For reference:

| PR | Score (bpb) | Approach |
|----|-------------|----------|
| #573 | 1.0523 | Multi-pass streaming legal TTT (overall leader) |
| #609 | 1.1154 | Flat 11L, XSA-all + Full GPTQ, no TTT |
| #593 | 1.1171 | Flat 11L, Parallel Muon + Full GPTQ, no TTT |
| #325 | 1.1462 | Looped 2x5, Middle-Cycle (my starting point) |
| **#363 (this PR)** | **1.1787** | **Looped 3x3, Noisy QAT + EMA + MTP** |

My best looped result is 0.063 bpb behind the best no-TTT flat submission. That gap is the cost of recurrence under these constraints.

---

## What Worked

### 1. Noisy QAT (Original Contribution)

This is the finding I'm most proud of and the reason this PR exists.

**The discovery**: On Day 1, my first 8xH100 run produced a catastrophic result. Pre-quantization bpb was 2.07 (decent for the architecture). Post-quantization bpb was 3.22. A **1.14 bpb gap**. The model was learning fine but quantization was destroying it.

Standard STE (Straight-Through Estimator) quantization-aware training simulates quantization during the forward pass. This works for flat architectures where each weight matrix is used once. But for looped architectures, quantization error compounds: the same weights get quantized once at export, but errors propagate through N repeat iterations. I measured the amplification factor at roughly **900x through 3 recurrence cycles**. Int6 starts with about 4x more error than int8, and that compounds through the loop into something catastrophic.

**The fix**: Instead of STE fake-quantization, inject differentiable uniform noise calibrated to match the magnitude of int8 per-row quantization error:

```python
# In CastedLinear.forward(), for loop core blocks only:
with torch.no_grad():
    amax = self.weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    step_size = amax / 127.0
noise = (torch.rand_like(w) - 0.5) * step_size.to(w.dtype)
w = w + noise
```

Key properties:
- **Differentiable**: Unlike STE, gradients flow through the noise. The model learns weight configurations robust to quantization-scale perturbations.
- **Loop-aware**: Applied only to core (shared) blocks, not stem/tail.
- **Calibrated**: Noise magnitude matches int8 per-row quantization step size. Not arbitrary regularization; matched to the actual export format.

**Result**: Quantization gap collapsed from **0.37 bpb to 0.002 bpb**. That's a 185x reduction. The technique is simple, costs nothing at inference, and should transfer to any depth-recurrent architecture.

(An aside: on the Middle-Cycle architecture with int5 export, Noisy QAT calibrated for int8 actually hurts slightly because the noise magnitude is wrong for int5 step sizes. Matching the noise to the actual export precision is critical. See negative result #10.)

### 2. SWA Inverts the Quantization Gap on Middle-Cycle

This was the weirdest result. Stochastic Weight Averaging (SWA), which periodically averages model checkpoints during training, produces smoother weight distributions. On the Middle-Cycle architecture, post-quantization bpb was sometimes **better** than pre-quantization bpb.

My hypothesis: SWA pushes weights toward flatter minima where the weight distribution is more uniform across rows. Per-row quantization handles uniform distributions well. The smoothing effect of SWA accidentally compensates for quantization noise rather than fighting it.

This might be useful to anyone combining SWA with aggressive quantization schemes.

### 3. 3x3 > 2x5 Loop Configuration

This is the most practically useful finding for anyone working on looped transformers.

I switched from 2x5 to 3x3 after studying Frosty40's Frugendorff (PR #499), which used 6 unique blocks looped only 2x. The intuition: more unique blocks with fewer repeats provides more representational diversity per parameter.

**Controlled comparison (single GPU, identical hyperparameters):**

| Config | Effective Depth | bpb | Artifact Size | ms/step |
|--------|----------------|-----|---------------|---------|
| **3x3** (3 core × 3 repeats) | 12 | **1.3462** | **11.9 MB** | **236** |
| 2x5 (2 core × 5 repeats) | 16 | 1.3519 | 13.2 MB | 260 |

3x3 wins on every axis: **-0.006 bpb, -1.3 MB smaller, -24 ms/step faster**. Two shared blocks repeated 5 times gives the model only 2 distinct computational "programs" to compose. Three shared blocks repeated 3 times gives 3 distinct programs, 50% more diversity, at the cost of only one additional block's worth of parameters.

### 4. The Training Data Shard Lesson

This one cost me hours of debugging and I'm including it as a public service announcement.

Midway through Day 3, I was getting 1.28 bpb on an 8xH100 VM where I'd previously gotten 1.18 on Hyperbolic bare metal. Same code, same config. I ran A/B tests, made LeakyReLU configurable, checked for code regressions. Nothing explained it.

The root cause: **I had only downloaded 1 training shard instead of 80.** The model was memorizing that single shard and generalizing poorly to the validation set. With 80 shards: 1.1914. With 1 shard: ~1.30. A 0.1 bpb difference from training data diversity alone.

Always use all 80 shards. Always.

---

## The Controlled Comparison

This is the definitive experiment. Same hardware (8xH100 SXM bare metal), same quantization (all-int5), same attention config (full MHA, 8 KV heads), same BigramHash (4096), same warmdown (2000), same seed, same eval pipeline (sliding window stride 64, T=0.90).

| | Flat 11L 512d | Looped 3x3 640d | Delta |
|---|---|---|---|
| **bpb (sliding window)** | **1.1648** | 1.1894 | **+0.025** (looped worse) |
| Artifact size | 15.3 MB | 14.5 MB | -0.8 MB (looped smaller) |
| Training steps | 5375 | 4175 | -1200 steps (looped fewer) |
| ms/step | 112 | 144 | +32 ms (looped slower) |

The looped model trains for 1200 fewer steps and each step is 32ms slower. In a 600-second time budget, this is devastating.

Frosty40 shared his own conclusion in the Discord on the same day: *"yeah i did a ton of a/b testing and its not improving anything, it was other modifications. so now im stripping those and running a/b. the recursion in this form is a bust."* He added: *"i kept adding shit to the 'recursive layer' exciting it was getting faster, and those modifications worked anyway, layer was just wasting cycles."*

Ciprian-Florin Ifrim, who ran 250+ experiments for his ternary submission and documented everything in a PDF I wish I'd had on Day 1, found the same. His eval depth recurrence sweep showed a total range of 0.0009 bpb across 5 different repeat counts. Pure noise.

Three independent researchers. Three different architectures. Three different optimization approaches. Same conclusion.

---

## Why Recurrence Fails at This Scale

There are two distinct penalties. I call them the **two taxes of recurrence**.

### Tax 1: Quantization Compounding

Shared weights are stored once and quantized once. But during inference, quantization error propagates through every repeat iteration. For 3x3, each core block's error is seen 3 times. For 2x5, 5 times. And the errors compound nonlinearly because each iteration's output feeds into the next iteration's input.

Noisy QAT partially addresses this (see above), but only for int8 targets. At int5 precision, the interaction between QAT noise and already-aggressive quantization becomes counterproductive.

boreas in the Discord summarized this perfectly: *"so you can't scale the recurrence to take advantage of the smaller size because of the compounding quant tax?"*

Exactly.

### Tax 2: Step Time Overhead

Each loop iteration adds wall-clock time. On 8xH100:

- Flat 11L: 600s / 0.112s = **~5375 steps**
- Looped 3x3: 600s / 0.144s = **~4175 steps**

That's 22% fewer training steps. In a regime where every step matters, this is a brutal penalty.

### Why the Size Advantage Cannot Compensate

The looped model is 0.8 MB smaller (14.5 vs 15.3 MB). Could that headroom fund higher precision to close the 0.025 bpb gap?

No. Moving from int5 to int8 on 0.8 MB of parameters improves roughly 0.005 bpb (based on competition-wide quant deltas). That's an order of magnitude short of the 0.025 gap. The parameter savings from weight sharing are real but insufficient to offset both taxes combined.

---

## The Full Experiment Log

### Day 0: Research + 3070 Prototyping

- Deployed 15 research agents across Chinese, Japanese, Korean, Israeli, Indian labs
- Identified depth recurrence as the unexplored lane
- Built first looped model on 3070: 1.5630 bpb, 6.1M params, 4.1MB artifact
- Ran scaling sweep on 3070: tested wide (3x3 at 768d), deep (5x3 at 512d), balanced (4x4 at 640d)
- All larger configs throughput-limited on 3070; couldn't get enough steps to converge
- Investigated custom compression (entropy analysis showed 2.94 bits/value for int6 vs 5.0-5.5 from zstd)
- Tested bit-packing, delta encoding (delta encoding was a dud), Huffman coding concepts

### Day 1: A4500 Testing, First 8xH100, The Quantization Discovery

- Rented 2x A4500 pods ($0.19/hr spot) for scaling sweeps
- Tested LoRA adapters on recurrence: NoLoRA won at low step counts
- BigramHash stacked well with recurrence
- SmearGate hurt recurrence (gating mechanism incompatible with shared weights)
- MTP broke badly (auxiliary gradients corrupted shared recurrent weights)
- **First 8xH100 run: catastrophic 1.14 bpb quantization gap** (pre-quant 2.07, post-quant 3.22)
- Discovered the ~900x error amplification through recurrence cycles
- **Developed Noisy QAT**: gap collapsed from 0.37 to 0.002 bpb
- Submitted PR #363 as non-record research contribution

### Day 2: Forking PR #325, Code Review Gauntlet, Sweeps

- Forked Node's PR #325 (looped 2x5 Middle-Cycle architecture)
- Applied batch fixes: Muon 0.99, warmdown adjustment, Partial RoPE 16/64, LN Scale, XSA last 4, Late QAT
- Discovered SWA gap inversion (post-quant sometimes better than pre-quant on Middle-Cycle)
- **26-model code review gauntlet** found the `global _QAT_ACTIVE` bug and 5 other issues
- Ran parallel hyperparameter sweeps on two 2xH100 rigs while at work
- Confirmed: EMA(0.997) ≈ SWA, warmdown 1500 > 3000 > 1000, MTP 4 heads / weight 0.3, Muon WD 0.02
- GPTQ-lite: -0.0027 bpb (free, post-training)
- Value Residual: catastrophically incompatible with loops (+0.14 worse)
- TTT with AdamW: catastrophically overfit at lr=0.0005 (1.5636 bpb)

### Day 3: 3x3 Beats 2x5, The Shard Lesson, Architecture Switch

- Tested 3x3 vs 2x5 after studying Frosty40's Frugendorff: **3x3 won on every dimension**
- Lost hours debugging 1.28 bpb on 8xH100 VM; root cause was 1 training shard instead of 80
- With 80 shards: 1.1914. With 1 shard: ~1.30.
- Best 8xH100 looped result: **1.1787 bpb** sliding window (3x3 + EMA + MTP + int5 + GPTQ-lite + T=0.90)
- Tried FIIZiK_'s techniques: stride 16 eval (-0.015 bpb, huge), T=0.90 (confirmed optimal)
- Factored embeddings at 192d: catastrophic (+0.053 regression). At 256d: still bad (+0.063)
- FIIZiK_ told me his optimal was 256 on 768d, but it doesn't transfer to our int5 setup

### Day 4: Flat Comparison, Accepting the Data

- Frosty40 DMs me: recursion is a bust, he's stripping it out after days of DGX Spark A/B testing
- FIIZiK_ asks if I'm on the recurrent transformer; I tell him yes, factored dims didn't work, 1.1787
- He says: *"Well 1.18 to 1.17 is nice"* and *"I mean that's not the point of this challenge imo"*
- **Ran the controlled flat vs looped comparison**: flat 1.1600 (int6, over budget), flat 1.1648 (all-int5, fits), looped 1.1894 (same tuned config)
- Flat wins by 0.025. The loop adds ~32ms/step overhead = 1200 fewer training steps.
- Tried adding the loop back to the tuned flat config just to be sure: confirmed +0.025 penalty
- Compared against Frosty40's PR #499: his MLP 4x and 6×2 loop gave 1.1478, better than our 3×3 with 3x MLP, but his own A/B testing showed the gains came from MLP width, not the loop

### 8xH100 Results Summary

| Config | Sliding bpb | Steps | ms/step | Artifact | Fits? |
|--------|------------|-------|---------|----------|-------|
| Flat 11L tuned (fullMHA+bg4096+wd2000, all-int5) | **1.1648** | 5375 | 112 | 15.3MB | YES |
| Flat 11L baseline (GQA, bg2048, wd1500, all-int5) | 1.1671 | 5550 | 108 | 15.0MB | YES |
| Flat 11L (int6, over budget) | 1.1600 | 5550 | 108 | 17.2MB | NO |
| Looped 3x3 best (EMA+MTP+int5+GPTQ-lite) | 1.1787 | 4200 | 143 | 15.6MB | YES |
| Looped 3x3 tuned (same config as flat winner) | 1.1894 | 4175 | 144 | 14.5MB | YES |
| Looped 2x5 (original PR #325 fork, 3-seed mean) | 1.1834 | 4200 | 143 | 15.6MB | YES |

### Hyperparameter Sweeps (2xH100)

All sweeps on 2xH100 with 1 data shard. Directionally reliable but absolute numbers are higher than 8xH100.

**EMA x Warmdown** (20 combinations, most corrupted by torch.compile recompilation):
- Best surviving: EMA 0.996, Warmdown 2000 = 1.2910 bpb

**MTP (Multi-Token Prediction)**:

| MTP Heads | Loss Weight | bpb |
|-----------|-------------|-----|
| **4** | **0.3** | **1.2974** |
| 6 | 0.3 | 1.3010 |
| 2 | 0.3 | 1.3045 |

**Muon Weight Decay** (lower is better for looped, opposite to flat convention):

| WD | bpb | Delta |
|----|-----|-------|
| **0.02** | **1.2955** | baseline |
| 0.04 | 1.2983 | +0.003 |
| 0.06 | 1.3060 | +0.011 |

Hypothesis: weight decay on shared parameters has an outsized effect because those weights are used in every loop iteration. Aggressive decay compounds through the loop just like quantization error.

---

## Negative Results (All 12)

Every failed experiment, with specific numbers. This section may be the most useful part of this writeup.

### 1. XSA on All Layers (Looped)

XSA applied to all blocks including loop core on every repeat: **+0.001 worse** (1.1953 vs 1.1940). On a looped architecture, "all layers" means the shared core blocks get XSA on every repeat. Too aggressive. The standard 11L stack benefits because its "all 11 layers" means 11 *unique* computations. Our "all layers" means 3 unique computations, each repeated 3 times. Very different.

### 2. Cyclic Muon Momentum (0.85-0.95, period 50)

Reported as -0.0045 bpb on flat architectures (PR #623). Combined with XSA and QuadgramHash: **+0.058 worse** (catastrophic). The momentum drops below the warmup target (0.85), destabilizing looped convergence. Looped architectures amplify optimizer instability because perturbations compound through repeat iterations.

### 3. QuadgramHash (1024 buckets, dim 32)

Tested alongside cyclic momentum and XSA. Could not isolate. When the combined test came back +0.058 worse, there wasn't compute budget to test each independently. Inconclusive.

### 4. Factored Embeddings (EMBED_DIM 192 and 256)

FIIZiK_ used EMBED_DIM=254 on his 768d ternary model and called it "very small loss." But his architecture is fundamentally different (ternary weights, 8192 vocab). On our int5 setup with vocab 1024:

| EMBED_DIM | Ratio | bpb | Delta | Artifact |
|-----------|-------|-----|-------|----------|
| 640 (none) | 100% | 1.1787 | baseline | 15.6MB |
| 256 | 40% | 1.2416 | **+0.063** | 14.8MB |
| 192 | 30% | 1.2316 | **+0.053** | 16.4MB (OVER) |

Both terrible. With a 1024-token vocabulary, the embedding table is already small (1024 × 512 = 0.5M params). Compressing it further saves negligible parameters while destroying representation quality. Factored embeddings only make sense with large vocabularies (FIIZiK_ uses 8192).

### 5. Value Residual (ResFormer)

Reported as -0.015 bpb on flat architectures (PRs #486/#490). On looped: **+0.14 worse** (1.4378 bpb). Catastrophic. Even with initialization fix (lambda init at -4.0, so sigmoid(-4.0) ≈ 0.018 = almost no mixing initially).

In a looped architecture, the "first layer V" is from the stem, but the loop core sees it on every iteration. The V residual creates an increasingly stale reference as depth increases, and the shared weights cannot learn different mixing ratios for different repeat iterations. Value Residual assumes each layer has a unique position in the network; shared layers violate that assumption.

### 6. Progressive Loop Unrolling (2 → 5 repeats)

Start training with 2 loop repeats, linearly increase to 5. Broke DDP. Dynamic control flow is incompatible with torch.compile + DistributedDataParallel. Single-GPU test: **2172 ms/step** (9x slower than baseline 236 ms/step). The compile graph breaks on every repeat-count change, triggering full recompilation.

### 7. Sawtooth LR Schedule

Caused torch.compile recompilation **every step** because the LR change triggers a guard check. Step time went from 248 ms to **987 ms** (4x slowdown). Only 607 steps completed. Results were garbage.

Same root cause as #6: anything that changes a value torch.compile traces through causes recompilation. LR schedules must be implemented outside the compiled region.

### 8. Test-Time Training (Full-Weight)

829 steps of AdamW on validation data: **1.56 bpb** vs 1.38 baseline. Massive overfitting. GPTQ-quantized weights sit in narrow curvature-aligned minima that AdamW's adaptive learning rates destroy. TTT and aggressive quantization are fundamentally at odds unless using SGD or carefully constrained LoRA.

(Per-document LoRA TTT was implemented but DDP crashes prevented proper multi-GPU testing. Still on the to-do list.)

### 9. LeakyReLU(0.5)²

Reported as -0.003 on flat architectures. Showed **-0.003 improvement on 2xH100** (1-shard) but **negligible on 8xH100** (80-shard). The benefit may be data-regime-dependent: with 1 shard the model sees less diversity, and leaky activation's gradient flow through negative values helps; with 80 shards the model learns to route around dead ReLU regions naturally.

**Always validate single-GPU findings on the target hardware.**

### 10. Late QAT + int5

Enable QAT in the final 10% of steps, combined with int5 export: **+0.006 worse**. QAT calibrated for int8 noise is the wrong magnitude for int5 export. The model gets trained to be robust to int8-scale perturbations but actually faces int5-scale perturbations at export. Matching QAT noise to export precision is critical.

### 11. BigramHash(10240)

Reported as -0.070 bpb on flat 11L (PR #450). On looped: **no improvement** (1.2980 vs 1.2963 on 2xH100). Hypothesis: the looped architecture already gets some n-gram-like pattern recognition from seeing data multiple times through the loop. The additional bigram capacity is redundant with what the loop provides.

### 12. 704d Model Dimension

Increase from 640d to 704d for more capacity per block: **worse** on 2xH100. Fewer steps at higher ms/step. The wider model doesn't train enough in 10 minutes to compensate for increased per-step cost.

---

## What Might Work With More Compute

Honest speculation, clearly labeled.

### Longer Training Budgets

The fundamental issue is that looped models trade step count for effective depth. In 10 minutes, this trade is unfavorable. At 30+ minutes (or unlimited track), the step-count penalty shrinks while the parameter-efficiency advantage grows. PR #612 achieves 1.1079 bpb on the unlimited (100-min) track with a GEPA architecture. Looped architectures may be competitive at longer time horizons where the "Tax 2" (step time overhead) becomes less dominant.

### Adaptive Depth at Inference

If the model could choose how many loop iterations per token, easy tokens could exit early and hard tokens could iterate longer. This is the Universal Transformer's original proposal. The challenge: making this compatible with torch.compile and batched inference, both of which demand static computation graphs.

### Noisy QAT Matched to Export Precision

Our Noisy QAT was calibrated for int8 (step_size = amax / 127.0) but we exported at int5. A version calibrated for int5 noise (step_size = amax / 15.0) might close the gap. We ran out of compute to test this.

### Better Loop Designs

The 3x3 > 2x5 finding suggests the optimal configuration isn't obvious. Asymmetric loops (more stem than tail), heterogeneous repeat counts (repeat block 1 more than block 2), or attention on first and last repeat only with MLP-only middle repeats are all unexplored.

---

## Acknowledgments

- **Aum08Desai** (PR #325): The Middle-Cycle architecture and original 1.1462 bpb looped submission.
- **Frosty40** (PR #499, "The Frugendorff"): For sharing his negative results on recursion openly, both in DMs and in the public Discord. His honest assessment ("the recursion in this form is a bust... I kept adding [] to the 'recursive layer' exciting it was getting faster, and those modifications worked anyway, layer was just wasting cycles") saved me and others significant compute.
- **Ciprian-Florin Ifrim** (PRs #640/#641): The most thorough experiment documentation in the competition (250+ experiments). His suggestions on eval stride 16, temperature scaling T=0.90, factored embeddings, and z-loss directly shaped my experiments. His 250-experiment PDF is a masterclass in systematic ML research.
- **boreas**: For summarizing the core tension better than I could ("so you can't scale the recurrence to take advantage of the smaller size because of the compounding quant tax?"). Exactly.
- **Node / capitlism** (PR #325): For open-sourcing the looped transformer that started this whole investigation and telling people to "feel free to optimize."
- **The flat no-TTT SOTA authors** (PRs #609, #593, #606): The reference points that define what the standard stack can achieve, and indirectly, the ceiling that recurrence has to beat to be worth using.
- **OpenAI / Will DePue**: For sponsoring compute credits, actively answering questions in Discord, and creating a competition that explicitly rewards honest research alongside leaderboard performance. Will's comment that "people aren't being nearly ambitious enough" is what pushed me to continue working on the looped architecture in the first place.
- **Hyperbolic**: For the referral credits that made this possible. Sorry to your VCs.
- **The entire Parameter Golf community** (~640 PRs of shared knowledge): This competition's culture of open experimentation made this work possible. Seeing fbe_dev share his results in real-time, watching the referral credit meta-game unfold, and getting direct coaching from top competitors is not something I expected from an ML competition.

---

## Reproducing These Results

Training script: `pr325_train_gpt.py`

Key environment variables for the controlled comparison:

```bash
# Flat 11L 512d (best submittable: 1.1648 bpb)
NUM_LAYERS=11 MODEL_DIM=512 LOOP_CORE_LAYERS=0 LOOP_REPEATS=1 \
MLP_INT5=1 ATTN_INT5=1 NUM_HEADS=8 NUM_KV_HEADS=8 \
BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=2000 \
EVAL_TEMPERATURE=0.90 EVAL_STRIDE=64 SEED=42

# Looped 3x3 640d (1.1894 bpb on same config)
NUM_LAYERS=9 MODEL_DIM=640 LOOP_CORE_LAYERS=3 LOOP_REPEATS=3 \
MLP_INT5=1 ATTN_INT5=1 NUM_HEADS=8 NUM_KV_HEADS=8 \
BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=2000 \
EVAL_TEMPERATURE=0.90 EVAL_STRIDE=64 SEED=42
```

Both use `MAX_WALLCLOCK_SECONDS=600` on 8xH100 SXM with 80 training shards.

---

## Final Thoughts

I set out to prove that depth recurrence could be competitive in Parameter Golf. I failed. But I think the failure is worth more than another 0.001 improvement on the standard stack.

The two taxes, quantization compounding and step-time overhead, are structural. They are not hyperparameter problems or implementation bugs. They are consequences of the competition's constraints: a fixed time budget that penalizes slower steps, and an artifact size limit that forces aggressive quantization where shared weights compound errors.

Noisy QAT is, to my knowledge, a novel contribution. The idea that loop-core weights should be trained with noise calibrated to quantization error is simple, effective for int8 targets, and should transfer to any depth-recurrent architecture. The 0.37 → 0.002 bpb gap collapse is the strongest single result in this work.

The 3x3 > 2x5 finding is immediately actionable: prefer more unique blocks with fewer repeats.

Everything else is a negative result. I believe documenting these honestly is more valuable than cherry-picking the one configuration where looped models look competitive. When boreas asked "what sort of things did you try?" in the Discord, and Frosty40 warned "DO NOT FRUGENDORFF it just wastes cycles," I realized that the most useful thing I could do was write all of this down so the next person doesn't have to spend 4 days and $200 learning the same lessons.

If someone finds a way to make recurrence work under these constraints, these failures will save them time. If the gap turns out to be fundamental at this scale, this document explains why.

---

*Best looped: 1.1787 bpb (3x3, 8xH100, sliding window) | Best flat: 1.1648 bpb (11L, same hardware) | Controlled gap: +0.025 bpb (looped worse)*
