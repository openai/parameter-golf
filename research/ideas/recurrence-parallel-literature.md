# Recurrence research thread (active) + literature

**Status:** 🟢 ACTIVE — recurrence is the active research focus as of 2026-04-21. Kicking off this session.
**Goal:** find one recurrence-class experiment worth running within the remaining ~10-day deadline window.
**Budget envelope:** ~$30 for exploratory runs, reserve ~$60–80 for a 3-seed final submission.

## What we're doing today (2026-04-21)

1. ✅ Literature gathered on recurrence, parallel residuals, XSA, recurrence-schedule research.
2. ✅ Reviewed **PG-internal recurrence history** (#8 → #363 → #1285 → #1204 → #1334 → #1394 → Pavel's 857de47 → #1663 → #1697 → #1726 → #1736 → #1714 → #1678 → #1739).
3. ✅ **Confirmed actual baseline config:** #1736 uses `NUM_LOOPS=2, LOOP_START=3, LOOP_END=5, ENABLE_LOOPING_AT=0.35` — i.e., **Loop345 (3 layers × 3 passes = 17 virtual layers)**, not "Loop45" as directory name suggests.
4. ⏳ **Decision pending:** which recurrence experiment to actually run (see "Candidate experiments, DISPROVEN and SURVIVING" below).

## PG-internal recurrence findings (takes priority over outside literature for OUR stack)

| Experiment | PR | Result |
|---|---|---|
| Naive loop on unquantized (flat vs looped) | #363 | Looped +0.025 worse; quantization error compounds superlinearly through repeated weights |
| Noisy QAT fixes quant-recurrence interaction | #363 | Quant gap 0.37 → 0.002 bpb |
| WD raise 0.085→0.090 enables all-int6 | #1285 (dexhunter) | First working record; WD-quant synergy |
| Step-3000 onset first used | #1204 | Prior to this, no explicit step-based onset |
| Loop345 (3-layer loop) introduced | 857de47 (Pavel L.) | Merged as SOTA 1.0810 |
| Hard-onset = smooth homotopy | #1663 | **No difference; homotopy wasted complexity** |
| Onset sweep {1600,...,3000} on this arch | #1697 | 2600 optimal (single seed) |
| ENABLE_LOOPING_AT=0.15 vs 0.35 | #1726 | **0.15 is +0.050 worse** |
| Range expansion to layers 2-7 | #1726 | **+0.163 worse** |
| Peripheral shift to 5-6 only | #1726 | **+0.006 worse** |
| Step-0 activation | #1739 | **Catastrophic: 1.3936 bpb** |
| Recur-Alpha (learnable scalar, init 0) | #1714 | 1.0857 pre-TTT; **grant ran out before TTT eval** — composition with #1736 never measured |
| 4-layer loop Loop3-6 | #1678 | Pending 8×H100 eval |

## Candidate experiments: DISPROVEN on this stack — DO NOT RUN

| # | Experiment | Why killed |
|---|---|---|
| ~~1~~ | ~~Timing sweep (earlier activation)~~ | #1726 already: 0.15 → +0.050 worse. #1739: step-0 catastrophic. |
| ~~2~~ | ~~Progressive ramp / smooth activation schedule~~ | #1663: hard-onset = smooth, no difference. |
| ~~3~~ | ~~Position shift to layers 0-1 or 5-6~~ | #1726: layer 2-7 +0.163 worse; layer 5-6 +0.006 worse. Layer 3-5 is sweet spot. |

**Key learning:** ILR paper's "early-layer recurrence wins" finding does NOT transfer to this stack. It's been directly tested on PG and failed.

## Candidate experiments: STILL VIABLE

| # | Experiment | Isolation | Cost | Reasoning |
|---|---|---|---|---|
| **A** | **Port Recur-Alpha (#1714) onto #1736** | Clean | ~$25 | Learnable scalar per looped block, init 0 = identity. 6 params. Never composed with #1736's phased TTT (author's grant ran out). **Strongest remaining candidate.** |
| B | **Cross-pass XSA** (novel) | Clean | ~$25 | Subtract pass-N-1 component from pass-N during recurrence. Genuinely novel; not in any prior PR or paper found. Higher code risk. |
| C | **4-layer loop (Loop3-6)** | Clean env-var | ~$20 | #1678 running it in parallel; we could race or wait. Given #1726 showed expansion to 2-7 is worse, 3-6 sitting between 3-5 (good) and 2-7 (bad) is speculative. |
| D | **Wait for #1678 result** | N/A | $0 | Free info. Downside: no control over schedule, result may be inconclusive. |

## Updated recommendation

**Port Recur-Alpha from #1714 onto #1736 as spec 015.** Reasoning:

1. **Identity-at-init** — zero regression risk at worst. Unlike BigramHash's RNG drift or BPB-weighted's optimizer fight, this is safe-by-default.
2. **Composition pattern matches dexhunter's playbook** — take a proven-elsewhere lever, compose it with our base, be the first to measure it with phased TTT.
3. **Nobody else has this data.** #1714's grant ran out before TTT eval. We're uniquely positioned.
4. **Cheap.** ~30 LOC (scalar per loop iteration, init 0, fold into block-forward). Uses existing loop machinery. No encoder/decoder plumbing changes.
5. **Directly addresses the recurrence question.** The Recur-Alpha scalar is literally "how much of the recurrence output to use per pass" — learnable.

**Why not cross-pass XSA first:** more novel but higher risk. We haven't even measured whether pass-to-pass redundancy exists on our stack (the prerequisite). Cross-pass XSA is the follow-up if Recur-Alpha lands and redundancy analysis suggests stationarity.

## What running Recur-Alpha would look like

- **Code:** per `_parallel_block` and sequential block paths during the decoder loop, attach a learnable scalar `recur_alpha[k]` for each pass index k in 0..num_loops. At init all zero → block output blends α=0% of recurrence carry, α=100% of first-pass output (identity).
- **Branch:** `exp/recur-alpha` from research.
- **Ladder:** 2H smoke ($1) → 8H full ($20) = $21.
- **Seed:** 42, screening mode.
- **Expected Δ:** if α learns to move off 0, there's something real; if α stays near 0, recurrence isn't carrying useful information in later passes (consistent with stationarity hypothesis).

---

**Purpose (below):** Background reading list for the "training-side architecture" research thread. Not meant to be read cover-to-cover; use as a reference when designing experiments.
**Date compiled:** 2026-04-21.

## XSA — what's actually in our code

**Paper:** Exclusive Self Attention (XSA). Apple Machine Learning Research. arXiv: [2603.09078](https://arxiv.org/abs/2603.09078).

**Mechanism:** after computing the standard attention output `y = softmax(QKᵀ/√d) V`, XSA subtracts `y`'s projection onto its own V direction:
```
y' = y - (y · V̂) V̂      where V̂ = V / ||V||
```
The result is guaranteed orthogonal to V. This is exactly what #1736's `_xsa_efficient` does (lines 779–786).

**Stated motivation:** in standard attention, a token can attend to itself, and the self-attention contribution ends up parallel to its own V. This mixes "context information from other tokens" with "pointer to own value," creating unnecessary competition between modeling contextual vs. point-wise features. XSA removes this ambiguity.

**Claims:**
- Consistent improvement across model sizes up to 2.7B
- Gains grow with sequence length
- Replaces the need for "attention sink" tokens — each token becomes its own sink naturally

**Why this matters for us:** XSA is a specific case of a general class — "remove the redundant self-component from an accumulation." The generalization to cross-pass redundancy in depth recurrence is a natural extension.

---

## Recurrence / looped transformers — the core references

### Universal Transformer (foundational)

**Paper:** Dehghani et al., "Universal Transformers." ICLR 2019. arXiv: [1807.03819](https://arxiv.org/abs/1807.03819).

**Mechanism:**
- Tie parameters across layers — one block's weights are reused for all depths
- Recurrent in depth: iteratively refines representations by applying the same self-attention + transformation multiple times
- Adds ACT (Adaptive Computation Time) for dynamic per-position halting

**Theoretical claim:** computationally universal (can simulate any Turing machine with enough recurrent steps).

**Empirical claim:** outperforms standard Transformer on algorithmic + language understanding tasks.

**Relevance to us:** canonical "weights-shared across depths" design. #1736's Loop45 is a limited/structured version — only blocks 4-5 are shared across multiple passes, not the whole network.

### Adaptive Computation Time (ACT)

**Paper:** Graves, "Adaptive Computation Time for Recurrent Neural Networks." arXiv: [1603.08983](https://arxiv.org/abs/1603.08983).

**Mechanism:** per-position halting units — the model decides for each token how many steps of computation to do. Cost penalty ("ponder cost") prevents the model from always pondering forever.

**Relevance:** the other "more compute selectively" lever, orthogonal to fixed recurrence depth. Could be relevant if we want to make Loop45 adaptive per-token.

### Looped Transformers (more recent, 2023–2025)

**Theoretical:** Looped transformers can simulate many iterative algorithms (gradient descent, Newton's method, dynamic programming) exactly, with depth matching algorithmic step counts but parameter count constant. Turing complete at constant state width.

**Practical papers from 2025:**

1. **Intra-Layer Recurrence (ILR)** — Nguyen & Lin, CanadianAI 2025. arXiv: [2505.01855](https://arxiv.org/abs/2505.01855). **KEY FINDING: allocating more iterations to EARLIER layers yields optimal results.** They apply recurrence selectively per-layer. Code: [GitHub](https://github.com/ant-8/Layer-Recurrent-Transformers).
   - **For us:** #1736 loops blocks 4-5 (middle). ILR's finding says loops should be earlier. If this transfers, it suggests Loop23 or Loop12 might outperform Loop45.

2. **Retrofitted Recurrence** — arXiv [2511.07384](https://arxiv.org/html/2511.07384). Teaching pretrained LMs to think deeper via added recurrence.

3. **LoopFormer** — arXiv [2602.11451](https://arxiv.org/html/2602.11451). Elastic-depth looped transformers via shortcut modulation.

4. **Depth-recurrent structure** — some 2024-25 papers use "prelude + recurrent block + coda" pattern, matching #1736's encoder/loop/decoder structure.

### ALBERT-style parameter sharing

**Paper:** Takase & Kiyono, "Lessons on Parameter Sharing across Layers in Transformers." arXiv: [2104.06022](https://arxiv.org/abs/2104.06022).

**Mechanism:** share weights across ALL layers (like ALBERT) but with small per-layer adaptations.

**Relevance:** this is the maximally-shared version. #1736's Loop45 is a middle ground: full-sharing only for blocks 4-5. The paper discusses when sharing helps vs hurts — generally more sharing hurts less as models get larger.

---

## Parallel residuals / multi-stream

### GPT-J / PaLM parallel blocks

No single "canonical paper" — introduced in GPT-J codebase, refined in PaLM paper.

**Paper (PaLM):** Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." arXiv: [2204.02311](https://arxiv.org/abs/2204.02311).

**Mechanism:**
```
Serial (standard):  y = x + MLP(norm(x + Attention(norm(x))))
Parallel (GPT-J):   y = x + Attention(norm(x)) + MLP(norm(x))
```

**Claims:**
- ~15% faster training at large scale (attn + MLP input matmuls can be fused)
- Small quality degradation at 8B, no quality degradation at 62B
- Attention and MLP don't see each other's updates — less coupling but more compute parallelism

**Relevance:** #1736 uses a 2-LANE variant of this (lanes after `parallel_start_layer=8`). See next section.

### Multi-lane / multi-stream transformers

Most multi-stream transformer literature is from **vision** (DualFormer, DuoFormer, two-stream spatio-temporal). NLP applications are thinner.

**Branchformer:** Peng et al., ICML 2022. [proceedings.mlr.press/v162/peng22a](https://proceedings.mlr.press/v162/peng22a/peng22a.pdf). Parallel MLP-attention branches with learned merge — close to #1529's two-lane residual idea, in a speech setting.

**Relevance:** the academic literature on 2-lane-in-last-half-of-model variants (what #1736 actually does) is thin. Most lane-variants are "split from input, merge at output" in vision. #1529's "split starting at layer 8 of the decoder" is an engineering choice without clear precedent — meaning it's less well-studied and potentially less well-tuned.

### Subformer (another weight-sharing variant)

**Paper:** Reid et al., "Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers." arXiv: [2101.00234](https://arxiv.org/abs/2101.00234). Proposes "sandwich-style" parameter sharing — different sharing in different parts of the network.

---

## Synthesis — what the literature suggests for our research thread

### For recurrence specifically

1. **Early-layer recurrence > middle-layer recurrence.** ILR (2025) directly says this. #1736's Loop45 (middle) may be sub-optimally positioned.

2. **Variable-pass depth is standard practice in papers.** ACT and its descendants. But: it adds complexity; gains depend heavily on task.

3. **Turing-completeness results** are academic interest — don't think they give practical bpb gains, but mean the architecture has representational headroom.

4. **Cross-pass redundancy (our XSA extension idea) is not in the literature I found.** Genuinely novel direction.

5. **Prelude + recurrent + coda structure** is a known pattern. #1736 is already this shape. The research question is the DETAILS: which block positions, how many iterations, whether the recurrent block is heterogeneous.

### For parallel residuals

1. **Parallel vs serial is a well-studied quality-speed tradeoff.** Small models lose quality; large models don't. #1736 is small (~36M params) → probably losing some quality from parallel structure. Unlikely to be a quick win.

2. **Multi-lane literature is thin.** #1529's two-lane setup is relatively novel and therefore under-explored.

3. **Branchformer-style learned-merge** might be a cleaner design than 0.5+0.5 average. Worth considering.

---

## Variants worth considering (from literature + our stack)

Prioritized roughly by "novelty × expected EV × feasibility":

### High priority (literature-supported, high novelty for PG)

1. **Loop position sweep.** Move `LOOP_START/LOOP_END` earlier per ILR finding. Cheap (env var). Specifically try `LOOP_START=2, LOOP_END=3` and `LOOP_START=1, LOOP_END=2`. Single runs each.

2. **Cross-pass XSA.** Apply XSA-style subtraction across recurrence passes. Subtract the component of pass-N's output parallel to pass-(N-1)'s output. Novel — not in literature. Identity-at-init if done right (init subtraction coefficient to 0).

3. **Asymmetric recurrence.** Loop only attention (not MLP). Rationale: attention is cheaper than MLP per pass, so more iterations at lower compute cost. Untested.

### Medium priority (literature-supported, moderate novelty)

4. **Variable recurrence during training.** Randomly sample NUM_LOOPS per step, so weights are robust across depths. Then evaluate at max depth. Compute: more expensive, more variance.

5. **Additional MLP output gate.** (Already proposed earlier) — zero-init gate on MLP output, same pattern as AttnOutGate. Not really recurrence-related but training-time arch.

6. **Branchformer-style learned merge** for parallel residuals. Replace fixed 0.5+0.5 lane merge with a learned merge. Small parameter addition.

### Low priority (speculative / hard to implement)

7. **ACT-style adaptive halting** per-token. Complex to implement; unclear win on BPB.

8. **ALBERT-style full sharing.** Share ALL layer weights, not just 4-5. Would dramatically reduce parameters but harder to tune.

---

## The open research question for us

**"Is cross-pass redundancy a real phenomenon in #1736's Loop45, and can we exploit it with an XSA-style correction?"**

Concrete steps to answer:
1. Instrument #1736 to log the cosine similarity between pass-N and pass-N+1 outputs during training. If similarity is very high (>0.95), passes are mostly stationary — subtraction would help. If it's already diverse (<0.7), XSA's contribution is small.
2. Based on instrumentation, decide: cross-pass XSA, loop-position change (ILR-inspired), or asymmetric recurrence.

This is ~2-3 hours of instrumentation work (no training, just adding logging + reading logs), then we know whether the expensive experiment is worth running.

---

## Reading priority if you want to dip in

If you read ONE paper: **Intra-Layer Recurrence in Transformers** (2505.01855). Shortest, most directly relevant ("earlier layers > middle layers for recurrence").

If you read TWO: add **Exclusive Self-Attention** (2603.09078) — the XSA paper. It's the inspiration for cross-pass XSA.

If you read THREE: add **Universal Transformer** (1807.03819) — the foundational recurrence paper.

Everything else is reference material.

---

## Recurrence schedule / when to activate (added 2026-04-21)

### What the literature says

1. **ProRes — Progressive Residual Warmup** — arXiv [2603.05369](https://arxiv.org/abs/2603.05369). Residual branches activate sequentially from shallow to deep with a coefficient warming 0→1 linearly. Shallow layers stabilize first.

2. **Sparse Growing Transformer (SGT)** — arXiv [2603.23998](https://arxiv.org/abs/2603.23998). Recurrence activated in DEEPER layers FIRST, then extended to shallower. Claims deeper layers differentiate earlier in training.

3. **Staged Training for Transformer LMs** — Shen et al, ICML 2022 ([PDF](https://proceedings.mlr.press/v162/shen22f/shen22f.pdf)). Progressive stacking with heuristic schedules (50K/70K/280K steps for 3/6/12 layer models).

4. **Learning to Grow (LiGO)** — reuse pretrained smaller models as initialization for deeper ones. Saves ~44% FLOPs on BERT-Base pretraining.

5. **Curriculum-Guided Layer Scaling (CGLS)** — arXiv [2506.11389](https://arxiv.org/abs/2506.11389). Couples progressive model expansion with data-complexity curriculum.

### Key tensions in the literature

- **ProRes / FreezeOut:** shallow layers converge first → freeze them early.
- **SGT:** deeper layers differentiate first → activate recurrence there first.
- **ILR:** early-layer recurrence beats late-layer recurrence (in their test).

These are contradictory claims about WHERE convergence happens first. Likely architecture- and task-dependent. **Our stack's dynamics haven't been measured.**

### Consistent claim across papers

**Progressive/curriculum schedules beat hard switches** for training stability. #1736's current `enable_looping_at=0.35` is a hard switch — the literature would predict this is suboptimal but not catastrophic.

### Candidate variants for "better timing"

Ranked by cost:

1. **Earlier activation:** `enable_looping_at=0.15` or `0.20`. Env var only. Cheap to test.
2. **Later activation:** `enable_looping_at=0.50` or `0.70`. Env var only. Cheap to test.
3. **Smooth ramp of NUM_LOOPS:** 0 → 1 at 35% → 2 at 60%. Code change, ~30-50 LOC.
4. **Mixing-coefficient warmup:** blend single-forward and looped output via α that ramps from 0 to 1. Code change, moderate.
5. **Progressive layer expansion (SGT-style):** layer-specific activation schedule. Complex; probably not worth the budget.

### Honest assessment for our decision

- The literature is NOT conclusive. Progressive > hard is consistent, but specific schedules vary.
- Variants 1 and 2 are env-var-only and could be tested in a 3-run sweep (~$15 total in screening mode).
- Variants 3 and 4 need code changes; higher risk of bugs per our Edit-split-block lesson.
- Running a 3-point sweep of `enable_looping_at` on spec 008 would generate empirically-grounded data about OUR stack's dynamics, which neither the ILR nor SGT papers provide.
