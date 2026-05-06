# The Crawler: A 23-Day Architecture Research Campaign

<img width="812" height="522" alt="crawler_2" src="https://github.com/user-attachments/assets/9fd8add0-ba3c-46ab-a2ca-e50eb311d1cc" />


This document is a chronological research corpus for rapid iteration on an architecture concept. It is a hybrid flat-plus-crawler transformer with a shared depth-recurrent bottleneck. It is not quite a Universal Transformer, but I ran it for four hours with the assumption that it was close enough.

The tables below are the fast read. I iterated on the concept of a transformer with recursive properties and enjoyed developing an original system.

The end state is straightforward. The legal 10-minute crawler runner is a depth-recurrent hybrid transformer `7F+3C, loops=3, loop-aware GPTQ, prune+pyminify` at `1.13536063` BPB and `15,844,157` bytes. The long-run confirmation is `7F+3C, loops=3, loop-aware GPTQ, 4-hour full run` at `1.07424983` BPB and `14,184,849` bytes. The final four-hour run establishes stable wind-down behavior and handling of the recursive elements.

---

## Core Tables

### Table 1: Final Outcome And Major Checkpoints

| Date | Technical stack | Alias | int6_sw BPB | Size | Status | Why it matters |
|------|-----------------|-------|-------------|------|--------|----------------|
| 2026-03-22 | 6×2 shared, 640d, MLP4 | Frugendorff Squared | 1.1478 | 15.15MB | first production result | Proved the original recursive line could survive H100 production scale before quantization reality hit |
| 2026-03-27 | 4F+1C, loops=4, DeltaNet heads=4, loop-aware GPTQ, late-start EMA | Medusa IV | 0.9984 mean (0.8104 best) | 9.96MB | volatile failure | Most explosive crawler-side score signal in the campaign, but too unstable and bug-prone to promote |
| 2026-03-30 | 4F+1C, loops=3, mlp=6.0 | Crawler Leg 3 | 1.18720 | 8.84MB | stable base | First durable hybrid foundation after the original line was closed |
| 2026-04-01 | 5F+1C+5F | Nightcrawler / BW11 | 1.17651 | 10.05MB | promoted | Made the program's real architecture thesis explicit: depth around the crawler bottleneck beats decorative complexity |
| 2026-04-02 | 9F+1C | BWX 9F | 1.13868 | 15.24MB | former legal leader | Best in-tree crawler leader before the late multi-crawler line opened |
| 2026-04-09 | 9F+2C | BW_9F2C | 1.13190 | 16.86MB | quality pass, size fail | Best recovered oversize descendant; proved the late line was better on quality but illegal on bytes |
| 2026-04-10 | 8F+3C, loops=3, loop-aware GPTQ, prune+pyminify | Trapper Keeper 1 (s444) | 1.13541288 | 15.90MB | legal full run | First full legal seed of the late winner line |
| 2026-04-10 | 7F+3C, loops=3, loop-aware GPTQ, prune+pyminify | Trapper Keeper 1 (s4) | 1.13536063 | 15.84MB | current legal leader | Best legal 10-minute crawler run in the corpus |
| 2026-04-11 | 7F+3C, loops=3, loop-aware GPTQ, 4-hour full run | Nightcrawler Cubed (s4) | 1.07424983 | 14.18MB | 4-hour confirmed | Strongest confirmation that the crawler thesis held beyond the 10-minute constraint |

### Table 2: Naming Translation

| Internal name | Technical stack meaning | Role in the program |
|---------------|-------------------------|---------------------|
| Frugendorff | shared recursive transformer, `K` unique blocks applied `N` times | original proof-of-concept line |
| Medusa | `4F+1C`, 4 loops, DeltaNet heads, loop-aware GPTQ | volatile DeltaNet continuation between closure and resurrection |
| ClownCar | early flat-plus-crawler hybrid submission line | resurrection bridge after formal closure |
| Crawler Legs | early stable hybrid ablation line | foundation search |
| Bandit Wagon / BW | post-foundation stacking and interaction program | scaling and interaction testing |
| Nightcrawler | `5F+1C+5F` | first clean statement of the final architecture philosophy |
| BWX 9F | `9F+1C` | former legal in-tree leader |
| BW_9F2C | `9F+2C` | best recovered oversize descendant |
| Trapper Keeper 1 | `7F+3C` | legal 10-minute runner |
| Nightcrawler Cubed | `7F+3C` over 4 hours | long-run confirmation of the thesis |

`F` means unique flat transformer blocks. `C` means shared crawler blocks. Unless stated otherwise, the late crawler line uses 3 loops, loop-aware GPTQ, and int6 export.

### Table 3: Decisive Tests

| Date | Test | Technical stack | Result | Why it changed the program |
|------|------|-----------------|--------|----------------------------|
| 2026-03-23 | Quantization catastrophe | shared-loop line after GPTQ | 1.3766 pre-quant → 5.716 post-quant at 3 loops | Killed the naive recursive thesis and forced quantization discipline to become central |
| 2026-03-27 to 2026-03-29 | Medusa volatility test | `4F+1C`, 4 loops, DeltaNet heads, loop-aware GPTQ, late-start EMA | 0.8104 best seed, 0.9984 mean, std 0.1724 | Showed explosive upside inside the crawler line, but forced DeltaNet quarantine and a causality-fixed reset |
| 2026-03-29 to 2026-03-30 | Foundation lock | loop-aware GPTQ, EMA off, compile fullgraph, loops=3, mlp≥5 | durable positives across ablations and CL1/CL2 | Converted the crawler from an idea into a stable production recipe |
| 2026-04-01 | Nightcrawler depth test | `5F+1C+5F` vs earlier BW arms | 1.17651, promoted | Showed the crawler works best as a bottleneck inside a deeper flat scaffold |
| 2026-04-04 to 2026-04-06 | Helix and Ouroboros transfer test | aggressive side programs | strong micro signals failed to transfer or compose | Prevented the program from chasing decorative architecture novelty |
| 2026-04-08 to 2026-04-09 | Multi-crawler grid and symmetry | `8F+3C`, `7F+4C`, `9F+2C`, `9F+3C` | multi-crawler line beats BWX on quality | Established that the real late problem was legality, not model quality |
| 2026-04-10 | Legalization run | `7F+3C` with prune and aggressive pyminify | 1.13536063 at 15,844,157 bytes | Proved the late crawler line could be made legal |
| 2026-04-11 | Long-run confirmation | `7F+3C` over 4 hours | 1.07424983 at 14,184,849 bytes | Confirmed the Nightcrawler thesis under a much longer compute budget |

---

## Day-By-Day Research Journal

---

## Day 1 — March 18, 2026: Proof of Concept

> **Goal:** Validate whether recursive weight sharing can beat unique layers at matched parameter count.

The Frugendorff was born on a DGX Spark with 300 training steps beside a lot of other novel concepts in a rapid ablation sweep.

The idea was simple: instead of 9 unique transformer layers at 512 dimensions, use 3 shared blocks applied 3 times each at 864 dimensions. Same parameter count, wider layers, deeper effective network.

| Config | val_bpb | Delta | Params | Dim |
|--------|---------|-------|--------|-----|
| Baseline (9 unique, 512d) | 2.7927 | — | 17.05M | 512 |
| **Fractal only (3×3, 864d)** | **2.5953** | **−0.1975 (−7.1%)** | 16.57M | 864 |
| Fractal + Gravity (3×3, 864d) | 2.6149 | −0.1779 | 16.57M | 864 |

A 7.1% improvement with fewer total parameters. Gravity — auxiliary losses on early loops — hurt. The model learned to suppress early-loop contributions on its own, converging gravity weights to [0.13, 0.13, 0.70]. The mechanism was width, not recursion. That insight would take another week to prove rigorously.

---

## Days 2–4 — March 19–21, with Frug Squared landing on March 22: The Automated Sweep

>
> **Goal:** Find the optimal Frugendorff configuration through automated search and scale to H100. Develop a non-standard architecture concept.
> **Problem:** Recursion couldn't repeat back to back. It created a bandsaw pattern in my first tests, so I started playing with cadence to give the recursive layers time to hold data.

An overnight Qwen-guided architecture search ran 141 configurations and found the sweet spots: 2×4 block layout, cadence 3 (F/N/N pattern), learning rate 2e-3, MLP multiplier 3×. The winner reached 2.3332 BPB — 12% better than baseline.

A second extended sweep of 227 runs pushed further: 4×2 at cadence 4 hit 2.155 BPB. Cadence 4 beat cadence 3 beat cadence 1 beat cadence 2. A clear ordering.

**Discovery: Cadence Training.** Running all loops every step creates a "bandsaw" loss oscillation — shared weights receive contradictory gradient signals from different loop positions on consecutive steps. Alternating fractal steps (all loops) with normalize steps (single clean pass) in an F/N/N/N pattern gives the shared weights recovery time. The 227-run sweep confirmed cadence 4 is optimal. The bandsaw disappears at cadence ≥ 3.

**Discovery: TTT Leverage Multiplier.** Test-time training on shared weights updates all N loop iterations simultaneously — an N× leverage multiplier per gradient step. The v3+TTT variant peaked at 1.1901 BPB, a 0.032 improvement and roughly 3× the typical TTT gain. Aggressive TTT drifted after window 1400. (Note: TTT itself was never banned — only causality-violating implementations that score and train on the same tokens simultaneously. Legal score-first TTT, where chunks are evaluated before adaptation, remains permitted and is used by the #1 leaderboard submission. The Frugendorff's N× leverage is an architectural property, not a causality issue — its legality depends on whether the implementation followed score-first protocol.)

The full-scale H100 runs followed:

| Run | Config | Sliding BPB | Artifact | Steps | ms/step |
|-----|--------|-------------|----------|-------|---------|
| **Frugendorff Squared** | **6×2, 640d, MLP 4×** | **1.1478** | **15.15MB** | 4,390 | 137 |
| v2 | 3×4, 960d, MLP 3.0 | 1.2113 | 14.2MB | 5,738 | 105 |
| v3+TTT | 3×4, 960d, MLP 3.3 | ~1.1901 | 14.3MB | 5,590 | 107 |
| v1 (2-block) | 2×4, 1024d, MLP 3.0 | 1.2715 | 11.3MB | 7,625 | 79 |

Frugendorff Squared landed at 1.1478 BPB — 0.025 from the posted world lead. The research line looked promising. It was submitted as PR to openai/parameter-golf, framed as a compression technique.

---

## Days 5–6 — March 22–23: The Quantization Catastrophe

> **Goal:** Submit the Frugendorff as a competitive entry. Instead, discover that quantization destroys it.

GPTQ destroys weight-shared models. Rounding error compounds multiplicatively across loop iterations:

| Config | Pre-quant BPB | Post-quant BPB |
|--------|--------------|----------------|
| 11L, share_start=4, 3 loops | 1.3766 | **5.716** |
| 11L, share_start=4, 4 loops | 1.4058 | **6.313** |
| 11L, share_start=4, 5 loops | 1.4138 | **6.246** |

More loops meant worse survival. The gap between pre-quant and post-quant performance was not 0.01 or 0.05 — it was 4+ BPB. Catastrophic. This single finding would drive the entire subsequent quantization research program: loop-aware GPTQ, QAT, int8 training.

---

## Day 7 — March 24, 2026: The H4 Bridge

> **Goal:** Keep the chronology honest between catastrophe and closure. Capture the bridge-day work that tested post-Frug reformulations before the line was formally declared dead.

| Run | BPB | Notes |
|-----|-----|-------|
| H4A_gsv7_control | **1.21451** | Best recovered bridge-day row |
| H4B_gsv7_crawler_bank | 1.23711 | Early crawler-bank variant |
| H4_A — 6 flat, 0 crawler | 1.35077 | Flat-side bridge comparator |
| H4_B — 5 flat + 1 crawler ×2 at bottleneck | 1.35134 | Bottleneck crawler bridge arm |

Both scored 1.35, but the crawler addition used fewer MB. This kept interest in the line.

---

## Days 8–10 — March 25–27: Signal Analysis and Death

> **Goal:** Determine whether the Frugendorff can be saved. Prove what the mechanism actually is. Make a go/no-go decision.

A rigorous 7-test statistical analysis across 8 micro-crawler TSVs, 175 Frugendorff sweep configs, and production cadence ablations delivered a verdict that was hard to argue with:

1. Width is the primary lever (~0.033 BPB from 22% wider dim at fixed params).
2. Recursion provides **zero per-step benefit** — crawler-step and normalize-step losses are identical.
3. More looping produces an early boost that **decays** over training.
4. Post-processing (SWA, quantization) is hostile to shared weights.

Steps after crawler-steps are slightly *worse* than steps after normalize-steps. There is no momentum effect from iterative refinement. The Frugendorff's advantage was almost entirely explained by wider layers at fixed parameter count, not by the recursive refinement that was supposed to be the whole point.

An H-FRUG remediation sweep on RTX A6000 tested three approaches — KL distillation, loop bottleneck gate, reinvest to width — and all failed. Root cause: shared-weight gradient conflict. The same weights must simultaneously map early→mid and mid→late representations, and at short training budgets the optimizer cannot find a viable compromise.

The conclusion was written formally: **"Frugendorff / Crawler research line: CLOSED."** Personal Neural SOTA at time of closure: 1.1129 BPB (Rat Rod Green v1).

At almost the same moment, one continuation branch produced the most exciting unstable result in the whole crawler story: **Medusa**. It took the crawler into a `4F+1C` topology with 4 loops, `DELTA_NET_HEADS=4`, loop-aware GPTQ, and late-start EMA.

| Seed | Sliding-window BPB | Size | Post-EMA BPB | Steps |
|------|--------------------|------|--------------|-------|
| 42 | **0.8104** | 9.96MB | 0.2519 | 4,872 |
| 300 | 0.9578 | 9.97MB | 0.3882 | 4,880 |
| 1337 | 1.2269 | 9.96MB | 0.7126 | 4,876 |
| mean | **0.9984** | 9.96MB | — | — |

Medusa mattered because it proved there was still real energy in the crawler line even after the formal death verdict. It also proved that raw breakout scores were not enough. The branch was too volatile to trust, and the failure mode was instructive: a DeltaNet state-dtype bug plus quantization unravel through 4 crawler loops. That combination is why DeltaNet remained fascinating but eventually became quarantined from the stable crawler line.

---

## Days 11–12 — March 28–29: The Junkyard Era and Resurrection

> **Goal:** Ship what we can (ClownCar hybrid submission), then figure out if the crawler concept has any life left in a different form. N-grams were popular, the Frug was small, and combining them looked like a low-BPB, low-MB option. My due diligence on N-gram building failed me, and my personal "legal" version had a leak even though it was listed as legal on the leaderboard. Who cares. N-grams are what they are. Let's keep going.

At this point I knew I could get decent numbers from the recursion element, but it would unwind on me during compression. I had to find a way to stabilize the fractal compaction idea.

**The ClownCar series** evolved in a parallel submission-oriented track, combining the crawler architecture with an X-WING ngram oracle. ClownCar (base) established the 4F+1C×4L template at 1.1996 BPB. Raw crawler-track results: broken config 1.18119 (seed 1337), 1.18150 (seed 42); correct neural config 1.12299 (seed 1337). FX_Wing_Delta scored 1.18092 (seed 1337). ClownCar_II added DeltaNet heads with canonical kernel and achieved 1.0427 BPB sliding window — but exposed severe EMA degradation during warmdown (0.47 → 0.73 BPB). ClownCar_III tried trigram preprocessing with no improvement. ClownCar_VI stripped both EMA and GPTQ (`SKIP_EMA=1, SKIP_GPTQ=1`), capturing live weights instead. ClownCar_VII re-enabled GPTQ with loop-aware 2-phase calibration. These experiments established patterns that became permanent: `SKIP_EMA=1`, `LOOP_AWARE_GPTQ=1`, and `DELTA_NET_HEADS=0` (quarantined for creating causality violations via cross-loop state carry).

The Medusa cleanup line belongs here too. `Medusa_VII` was the causality-fixed crawler base that fed into the March 29 submission path. In other words, Medusa did not disappear when the unstable DeltaNet version failed. The useful parts survived: loop-aware GPTQ, no-EMA discipline, and a cleaned crawler path with `DELTA_NET_HEADS=0`.

**Bandit_ClownCar_X + Cubric Ngram9** was submitted on March 29 as a production hybrid:

| Seed | Full eval BPB | Sliding window BPB | Size |
|------|--------------|-------------------|------|
| 444 | **0.4957** | **1.1860** | 9.21MB |
| 300 | 0.4961 | 1.1868 | 9.52MB |
| 4 | 0.4964 | 1.1874 | 9.27MB |
| **mean** | **0.4961** | **1.1867** | |

The architecture was 4F+1C×4L with a full X-WING ngram oracle (orders 2–9, 8M hash buckets per order, 3D Cubric with 54 adaptive multipliers, entropy-adaptive alpha 0.20–0.75). 

The **Cobra** harness was designed as a 9-config multi-armed bandit for base model quality (COMPLEMENT_ALPHA sweep, SWA_EVERY, weight decay, late-QAT thresholds). The BW series' rapid progress made it obsolete before it could be scheduled — the answers it would have found were already being discovered faster through targeted ablations.

The decisive architectural pivot happened here. Rather than treating recurrence as the whole model, the crawler was recast as a **hybrid component** within a flat transformer stack. That shift, more than any single hyperparameter, created the modern crawler lineage.

---

## Days 12–13 — March 29–30: Foundation

> **Goal:** Map the entire hybrid crawler parameter space, separate viable regions from non-viable ones, and lock a production config.
>
> The important thing discovered is that recursion requires a loop-aware GPTQ path to compress properly. (*If not GPTQ, then some other loop-aware compression path.*)

**Crawler_Ablations_v1** (1×H100, 600s/arm, seed=1337) was the first systematic sweep of crawler infrastructure:

| Arm | Override | int6 SW BPB | Delta vs A | Verdict |
|-----|----------|-------------|------------|---------|
| A_baseline | (none) | 1.60513 | — | baseline |
| **B_loop_aware_gptq** | LOOP_AWARE_GPTQ=1 | **1.56511** | **−0.0400** | **WIN** |
| **E_compile_fullgraph** | COMPILE_FULLGRAPH=1 | **1.57930** | **−0.0258** | **WIN** |
| D_int8_off | CRAWLER_QUANT_INT8=0 | 1.60273 | −0.0024 | wash |
| C_ema_on | SKIP_EMA=0 | 1.67479 | +0.0697 | **LOSER** |
| F_gptq_and_ema | both | 1.70575 | +0.1006 | **WORST** |

Loop-aware GPTQ won by −0.040. EMA was actively harmful (+0.070 to +0.101) — it smooths weights in ways that destroy GPTQ calibration. EMA and GPTQ are antagonistic; combined is worse than either alone. Fullgraph compile saved 0.026 BPB simply by being faster (more steps in the same wallclock).

**Crawler_Leg_1** (1×H100, 600s/arm, seed=1337) was the 11-arm foundational parameter map:

| Arm | Config | int6 SW BPB | Delta | Verdict |
|-----|--------|-------------|-------|---------|
| **CL1-07** | **mlp_mult=5.0** | **1.64868** | **−0.0977** | **BEST** |
| **CL1-01** | **loops=3** | **1.65890** | **−0.0875** | **WIN** |
| CL1-00 | baseline (loops=4, mlp=4.0) | 1.74636 | — | baseline |
| CL1-05 | inst_dim=64 | 1.75201 | +0.0057 | wash |
| CL1-04 | inst_dim=16 | 1.75600 | +0.0096 | wash |
| CL1-03 | inst_dim=0 (off) | 1.78019 | +0.0338 | LOSER |
| CL1-09 | 5F+1C | 1.79416 | +0.0478 | LOSER |
| CL1-02 | loops=5 | 1.81547 | +0.0691 | LOSER |
| CL1-06 | mlp_mult=3.0 | 1.86261 | +0.1163 | LOSER |
| CL1-10 | 3F+2C | 1.86610 | +0.1197 | LOSER |
| CL1-08 | crawler_quant_int8=0 | 1.94389 | +0.1975 | WORST |

MLP width was the largest lever: mlp=5.0 was −0.098 BPB AND faster (655ms vs 735ms). Fewer loops was better — each loop adds ~0.085 BPB quant gap. INST_DIM (tested 0, 16, 32, 64) had only 0.034 BPB total impact, nearly irrelevant. CRAWLER_QUANT_INT8=1 was mandatory (+0.198 BPB catastrophic if disabled). More crawler blocks destroyed quality at this scale.

A quick **Bandit Wagon width/depth proxy** (500 steps, seed=444) confirmed depth beats width at every tested point. Width expansions to 576 and 640 both trailed the depth arms.

**Crawler_Leg_2** (8×H100, 350s/arm, seed=1337) confirmed CL1 winners at production scale:

| ID | Config | int6 SW BPB | Delta | Verdict |
|----|--------|-------------|-------|---------|
| **CL2-02** | loops=3 + mlp=5.0 + GPTQ + COMPILE | **1.19593** | **−0.0069** | **BEST** |
| CL2-04 | loops=3 + mlp=6.0 | 1.19828 | −0.0046 | good |
| CL2-01 | loops=3 + mlp=5.0 | 1.20211 | −0.0007 | wash |
| CL2-00 | baseline (loops=4, mlp=4.0) | 1.20285 | — | baseline |
| CL2-03 | loops=2 + mlp=5.0 | 1.20667 | +0.0038 | LOSER |

Architecture wins compressed at scale: CL1's −0.098 became CL2's −0.007. But the direction held. Production config was locked: **loops=3, mlp=5.0+, COMPILE_FULLGRAPH=1, LOOP_AWARE_GPTQ=1.**

---

## Day 14 — March 31: Scaling and Signal Stacking

> **Goal:** Stack confirmed wins (battery, compile, TAP, anchor) onto the locked config and push the 4F+1C base as far as it goes.

This phase tested whether the newly locked 4F+1C base could support additional signals without losing its quantization discipline. Most proposed additions did not survive escalation.

**BW3** (pyramid-512 choke + battery): 1.20684 BPB (+0.01964 vs Leg 3). Pyramid adds 1.57M cold params that do not converge in a 600-second budget. It did not promote.

**BW4** (battery only, ROPE_SCALES=9,1,1): 1.18731 BPB (−0.00015 vs Leg 3). Promoted. Zero extra params, tighter quant gap.

**BW5** (BW4 + COMPILE_FULLGRAPH=1): 1.18672 BPB seed 444, 1.18758 seed 300. Promoted on 2-seed mean (1.18715 vs 1.18743).

**BW5_Cannon** (scalar cannon): Gate signal (−0.00016) reversed at full run (+0.00020). Cross-run variance swamped the apparent gain, so the idea did not promote.

**BW5_PyramidCannon**: +0.03440 at gate. This was a decisive negative result; the 1.57M cold choke parameters compounded over time.

**BW5_Pyramid** (1GPU, 500 steps): −0.00987 proxy signal, but the 8GPU PyramidCannon showed it was proxy inflation. Deferred.

**BW6_Skipgram** (trigram hash, 4×GPU, 2000 steps, seed=444) was explicitly null:

| Arm | Change | int6_sw BPB | Delta | step_ms | Size |
|-----|--------|-------------|-------|---------|------|
| BW6SK-00 | bigram-only control | 1.28952 | — | 74.53 | 9.48MB |
| BW6SK-01 | +TRIGRAM | 1.28966 | +0.00014 | 74.47 | 9.34MB |

Crawler recurrence already approximates trigram context. The only secondary effect was a compression side benefit: the trigram artifact shrank by ~140KB while quality remained slightly worse.

**BW7 MegaGate** (8-arm, 4×GPU, 2000 steps) was the day's most important run:

| Arm | Change | int6_sw BPB | Delta | Verdict |
|-----|--------|-------------|-------|---------|
| CTRL-00 | baseline | 1.28913 | — | control |
| **TAP-03** | **TAP_DIM=32 shared** | **1.28560** | **−0.00352** | **WIN** |
| **ANC-05** | **ANCHOR_DIM=32** | **1.28578** | **−0.00334** | **strong** |
| TAP-04 | TAP_DIM=16 per-loop | 1.28646 | −0.00266 | positive |
| ANC-06 | ANCHOR_DIM=64 | 1.28750 | −0.00163 | weaker |
| TAP-02 | TAP_DIM=32 per-loop | 1.28773 | −0.00140 | positive |
| SMEAR-01 | CRAWLER_LOOP_SMEAR=1 | 1.28910 | −0.00003 | NULL |
| FLAT-07 | FLAT_WEIGHT_SHARE=1 | 1.32607 | +0.03694 | HARD FAIL |

TAP shared clearly outperformed TAP per-loop. Anchor dim=32 clearly outperformed dim=64. SMEAR was null, while SharedFlat failed decisively. The full arm table matters: TAP-04 and TAP-02 were both real but weaker than TAP-03, and ANC-06 confirmed the anchor signal narrows sharply as the state grows wider. The day produced two clear winners and eliminated the remaining alternatives.

---

## Day 15 — April 1: The Path to Nightcrawler

> **Goal:** Scale from 4F to 5F+, add GPTQ production calibration, and establish a new champion.

**BW8** (TAP shared dim=32) and **BW9** (Anchor dim=32) were designed as gate-only experiments using MegaGate evidence. Neither got its own full run — both were absorbed into the BW12 interaction matrix.

**BW10** ran loop-aware GPTQ as a standalone production test on the BW5+TAP base:

| | Gate (4×GPU, 2k) | Full Run (8×H100) |
|--|---|---|
| Control (no GPTQ) | 1.28889 | — |
| GPTQ int6_sw | 1.28403 (−0.00486) | **1.18293** |
| Delta vs BW5 | | **−0.00380** |
| Steps | 2,000 | 7,893 |
| Size | | 9.96MB |

Promoted. New champion. Proxy inflation was 1.3× (gate predicted −0.00486, production delivered −0.00380).

**BW11** added a 5th flat layer (5F+1C vs 4F+1C):

| Seed | int6_sw BPB | Quant Gap | Steps | Size |
|------|-------------|-----------|-------|------|
| 444 | **1.17651** | 0.01109 | 7,074 | 10.05MB |
| 300 | **1.17490** | 0.01395 | 7,077 | 10.34MB |
| 4 | 1.17676 | — | 7,074 | 10.27MB |
| **mean** | **1.1761** | | | |

Promoted. −0.01021 vs BW5, −0.00641 vs BW10. This was the **Nightcrawler** — a 5F+1C+5F bridge architecture. The champion lineage was now BW5 → BW10 → BW11/Nightcrawler. This was also the point where the architecture stopped being framed as "recursion for free depth" and started being framed as a compression/thinking engine: compress stored structure with a crawler bottleneck, then spend the saved budget on deeper flat scaffolding around it.

---

## Days 15–16 — April 1–2: Interaction Gates and the Depth Sprint

> **Goal:** Find the depth ceiling, test every remaining interaction knob, and select the best architecture for production. Ship BWX.

**BW12** (4×GPU, 2000 steps) tested interaction dynamics on the Nightcrawler baseline. Tap-off (−0.00199) and GPTQ (−0.00204) were both positive. Standard and loop-aware GPTQ tied exactly.

Full BW12 gate table:

| Arm | Change | int6_sw BPB | Delta |
|-----|--------|-------------|-------|
| BW12INT-00 | control (Nightcrawler 5F+TAP shared) | 1.27438 | — |
| BW12INT-01 | tap off | 1.27239 | −0.00199 |
| BW12INT-02 | anchor dim=32 | 1.27337 | −0.00102 |
| BW12INT-Q1 | standard GPTQ | 1.27234 | −0.00204 |
| BW12INT-Q2 | loop-aware GPTQ | 1.27234 | −0.00204 |

**BW13** tested the same axes on a tap-off baseline. Anchor dim=32 regressed (+0.00276). Anchor dim=64 regressed harder (+0.00292). GPTQ helped (−0.00193). GPTQ-lite was nearly as good (−0.00177) with less calibration cost. The key finding: **anchor and tap-off are antagonistic.**

Full BW13 gate table:

| Arm | Change | int6_sw BPB | Delta |
|-----|--------|-------------|-------|
| BW13INT-00 | control (tap-off baseline) | 1.27152 | — |
| BW13INT-01 | tap-off + anchor dim=32 | 1.27428 | +0.00276 |
| BW13INT-02 | tap-off + anchor dim=64 | 1.27444 | +0.00292 |
| BW13INT-Q1 | standard GPTQ | 1.26958 | −0.00193 |
| BW13INT-Q1L | GPTQ-lite | 1.26975 | −0.00177 |

**BW14** went for big swings:

| Arm | Description | int6_sw BPB | Delta |
|-----|-------------|-------------|-------|
| BW14BS-00 | Control (tap-off) | 1.27383 | — |
| **BW14BS-01** | **NUM_FLAT_LAYERS=6** | **1.26684** | **−0.00700** |
| BW14BS-02 | Crawler choke flat-128 | 1.28552 | +0.01169 |
| BW14BS-03 | Crawler choke flat-512 | 1.27378 | −0.00005 |

Six-flat depth was the biggest single-arm gain at −0.007. Choke flat-128 regressed strongly, while choke flat-512 was effectively neutral. Depth, not routing, was the relevant lever.

This launched a rapid architecture sprint. **BW15** consolidated the 41 arms from BW12–BW14 into a single decision matrix. **BW16** swept flat depth 6–11F; the isolated 10F run confirmed that more depth could improve quality (`1.24295` BPB) but not within the legal artifact budget (`17.83MB`), while 9F emerged as the practical ceiling. Width was explored through **micro crawler runs**. The best micro crawler (`run8_pd_cad2` at `1.13554`) actually beats BWX 9F on quality but exceeds 16MB:

| Run | int6 SW BPB | Notes |
|-----|-------------|-------|
| run8_pd_cad2 (dim=624) | **1.13554** | Best quality — beats BWX 9F, oversize |
| run6_best_delib (dim=624) | 1.13746 | oversize |
| H100 base (dim=624) | 1.13838 | oversize |
| run3_selfref (dim=624) | 1.14148 | oversize |

Every width increase blew the size budget, confirming depth as the only viable scaling axis. **BW17** tested cadence interactions on the 9F base at rapid scale — loops=2 showed a strong directional signal (−0.054) but was unconfirmed at production. **BW18** and **BW19** mapped the remaining knob space (49 architecture/cadence arms, 18 crawler balance arms) — the individual knobs were tested across BW7, BW12–BW14, and bridge ablations rather than as single monolithic campaigns. **BW20** templated the brotli compression swap that became part of the production pipeline. The collective signal from this sprint converged on one clear answer.

**BWX_Latest** was the contender selection tournament. Using everything learned from BW12–BW16, the winner was clear: 9 flat layers, 1 crawler, tap-off, no anchor.

**BWX 9F** production run (8×H100, 600s, seed=444):

| Metric | Value |
|--------|-------|
| **int6_sw_bpb** | **1.13867894** |
| model_params | 26,270,292 |
| step_avg | 110.19ms |
| steps | 5,446 |
| bytes | 15,239,617 (15.24MB) |

Config: 9F+1C, loops=3, ROPE=9,1,1, INST_DIM=32, TAP=0, ANCHOR=0, COMPILE_FULLGRAPH=1, SKIP_GPTQ=1.

This became the in-tree leader and remains the reference crawler submission in this repository. In 15 days, the program moved from formal closure of the original recursive line to a new hybrid architecture at `1.13867894`, the best result the crawler lineage produced under the campaign's legal constraints.

---

## Day 17 — April 3: Helix — The Boldest Idea

> **Goal:** Try a fundamentally different recurrence paradigm — co-firing instead of sequential loops — to break through the BWX 9F ceiling.

The Helix concept was radical, too radical: instead of the sequential encoder→crawler→decoder pipeline, fire the crawler alongside every flat layer with bidirectional cross-injection. The crawler fires 9 times (once per flat layer) instead of 2–3 in loop mode. Cross-injection via gated 32-dim projections (~65K new params), zero-initialized for warm start. Final merge: `x = x_flat + sigmoid(gate) * x_crawl`.

**Helix_DepthRecur** (4×H100, 500 steps, seed=444) produced the most illuminating result of the entire Helix campaign:

| Arm | Config | int6_sw BPB | vs ctrl |
|-----|--------|-------------|---------|
| R0 | 5F ctrl (no helix, no recur) | 1.41130 | — |
| R1 | 5F recur-only L2,3 | **1.50073** | **+0.089** |
| **R2** | **5F helix-only dim=64** | **1.40347** | **−0.008** |
| S0 | 5F helix + recur L2,3 | 1.40426 | −0.007 |

Depth recurrence without Helix incurred a 0.095 BPB quantization penalty. Helix reduced that penalty to 0.004. But recurrence added nothing on top of Helix — S0 was barely different from R2. Helix was therefore confirmed at proxy scale, but only in its helix-only form.

**Helix_FlatServe** proposed 5 modifications to flat layer behavior (residual scaling, noise injection, progressive delegation, crawler-aligned output projection, asymmetric skip connections). Deprioritized — the SplitHead results redirected all Helix resources toward cross-attention experiments instead.

**DarkHorse** attempted to port Helix onto an external codebase (PR #1296 by aryanbhosale, 1.0897 BPB). Their depth recurrence at layers 4,5 created a 0.095 BPB quant gap that Helix might have shielded. Both `train_gpt_base.py` and `train_gpt_helix.py` were written, but after `Helix_ab_3` showed that the SplitHead concept did not scale, the port was no longer worth the integration risk, and we got distracted and never ran it. I still kind of want to run it. The Helix line would be fun to work on. Bidirectional injection deserves affection.

---

## Day 18 — April 4: Helix SplitHead — The Largest Signal and Its Failure

> **Goal:** Find the optimal Helix configuration through a wide micro sweep, then scale it to production.

**Helix_SplitHead** ran a 30-arm micro ablation (4×H100, 200 steps, dim=256, seed=444). The concept: split the crawler's attention heads between self-attend and cross-attend, with position-agnostic keys on the cross side.

| Arm | Config | BPB | vs S0 ctrl | step_ms |
|-----|--------|-----|-----------|---------|
| **B6** | **cross=4, dim=192** | **1.8333** | **−0.0272** | 191.59 |
| B5 | cross=2, dim=192 | 1.8374 | −0.0231 | 210.98 |
| W2 | cross=2, WD=0.12 | 1.8482 | −0.0123 | 228.87 |
| D4 | cross=4, dim=128 | 1.8511 | −0.0094 | 191.23 |
| H4 | cross=4 (full cross) | 1.8549 | −0.0056 | 192.05 |
| S0 | helix ctrl (no split) | 1.8605 | — | 206.76 |
| S1 | no helix at all | 1.9112 | +0.0507 | 128.90 |

The headline: no helix → helix → best SplitHead gave a total swing of **−0.078 BPB** — the largest architectural signal in the entire crawler program. The split-ratio ladder was clean: H4 full cross landed 1.8549, H2 50% cross 1.8575, H1 25% cross 1.8586, and H3 75% cross 1.8597. Full cross-attention beat every split ratio. The crawler didn't want self-attention at all. Weight decay 0.12 was the strongest single hyperparameter. QK gain 5.0/6.0 both hurt — the crawler needs broad attention, not sharpened focus. 7F depth hurt all arms versus 5F.

**BWXII_Helix_SplitHead** attempted the same at production scale (9F, seed=300): 1.25031 int6_sw, 177.65ms/step, 11.04MB. It regressed versus BWX 9F (1.13868). The micro signal did not transfer.

This was the first lesson about scale and being cheap in this competition. It would not be the last.

---

## Days 18–20 — April 4–6: Quant Fix and Ouroboros

> **Goal:** Attack the quant gap from multiple angles (QAT, int8, contractive loss) and see if individually-positive signals can compose into a production win.

**BWXII_QuantFix** explored quantization fixes at 1GPU and 2×GPU scale. The best 1GPU result was `T4_smart_wd012_Q2_gptq_loop_int8` at 1.27427 BPB with a quant gap of 0.0109 — smart_skip was the strongest intervention. The 2×GPU movie test showed R4_full_fix improved raw BPB (1.2420) but the quant gap widened to 0.0383. Quantization erased most gains.

**Ouroboros Ablation** (4×GPU, 600s, seed=300) tested three quantization-focused improvements on the BWX 9F base:

| Arm | int6_sw BPB | Delta | Verdict |
|-----|-------------|-------|---------|
| control | 1.16409 | — | baseline |
| noisy_qat | **1.16113** | **−0.00296** | STRONG |
| crawler_int8 | **1.16094** | **−0.00315** | BEST |
| contractive | **1.16183** | **−0.00227** | positive |

All three beat control. All trained faster (176–178ms vs 186ms). But would they compose?

**Ouroboros_stacked** (2000 steps, seed=444) answered definitively: **no.** The stacked result was 1.24514 BPB — dramatically worse than any individual arm's ~1.161 — and the artifact exceeded 16MB by 172KB. The three signals conflicted when composed.

**Ouroboros_II** designed a mixed-bit variant (attn=5, mlp=6, embed=8) with all three improvements plus brotli on loops=2. It was deprioritized after Ouroboros_stacked showed the signals conflict when composed; a fourth stacking variant was not a good use of compute.

**Ouroboros_III** tried the stack on the exact BWX 9F base at full production scale (8×H100, 600s, seed=444): 1.14462 BPB. **+0.00594 worse than BWX 9F.** The individually-positive signals did not compose. Stacking failed.

**Helix_ab_3** (8×H100, 2000 steps, seed=444) scaled the SplitHead concept to full model size (4F, dim=512):

| Arm | int6_sw BPB | step_ms | Delta |
|-----|-------------|---------|-------|
| HAB3-00_ctrl | 1.28905 | 72.77 | — |
| HAB3-01_helix | **1.42860** | 86.35 | **+0.13955** |

At scale the signal reversed decisively. The `−0.078` micro result became `+0.140`, meaning the largest architectural signal in the program failed to transfer **twice** — once in `BWXII`, once in `ab_3`.

Also designed during this period: **BW17** cadence testing on DGX Spark (loops=2 showed a `−0.054` directional signal). **BWXI** proposed a 5-signal stack (brotli + GPTQ + QK4 + loops=2 + warmdown), but Ouroboros III's `+0.006` regression from only three stacked signals made that risk unjustifiable. **BW21 NoisyQAT** had the strongest isolated signal (`−0.00296` from Ouroboros ablation) but was deprioritized once the multi-crawler breakthrough appeared.

---

## Days 21–22 — April 7–8: The Multi-Crawler Breakthrough

> **Goal:** Explore loop depth and multi-crawler configurations. Find out if more than one crawler layer can beat the 9F+1C ceiling.

**BW22_LoopDepth** (8×GPU, 2000 steps gate, seed=444) swept loop counts on the 9F base:

| Arm | Loops | ROPE_SCALES | int6_sw BPB | step_ms | Delta |
|-----|-------|-------------|-------------|---------|-------|
| A0_ctrl | 3 | 9,1,1 | 1.24352 | 110.28 | — |
| A1_loop4_naive | 4 | 9,1,1,1 | 1.24255 | 119.82 | −0.00097 |
| A2_loop4_battery | 4 | 9,3,1,1 | 1.24260 | 119.58 | −0.00093 |
| **A3_loop5_battery** | **5** | **9,3,1,1,1** | **1.24091** | **128.88** | **−0.00261** |
| A4_loop5_prog | 5 | 9,5,3,1,1 | 1.24176 | 128.99 | −0.00176 |

Quality scaled with depth, but so did throughput cost: loop4 was ~9% slower, loop5 ~17%. A3 promoted as a quality-priority candidate.

**Corpus Ablations v1** (4×GPU, 1500 steps, seed=444) was a 16-arm screen on the BWX 9F base. It produced the single most important finding of Phase 5:

| Arm | Change | int6_sw BPB | Delta | Verdict |
|-----|--------|-------------|-------|---------|
| A00 | Control (BWX 9F) | 1.32999 | — | baseline |
| **A07** | **NUM_CRAWLER_LAYERS=2** | **1.31766** | **−0.01234** | **BREAKTHROUGH** |
| A04 | 4 loops diff (ROPE=9,3,1,1) | 1.32533 | −0.00467 | strong |
| A03 | 4 loops naive | 1.32604 | −0.00396 | strong |
| A05 | 5 loops prog | 1.32595 | −0.00405 | strong |
| A02 | ANCHOR_DIM=32 | 1.32771 | −0.00229 | positive |
| A06 | INST_DIM=64 | 1.33096 | +0.00097 | DEAD |

Two crawler layers — **−0.01234 BPB.** The breakthrough signal. The late QAT lane only yielded directional outcomes: A01 TAP shared was mildly positive (`~−0.0010`), A08 crawler-int8 was near-null (`~−0.0006`), A10 softclamp improved relative to legacy QAT A09 (`~−0.0047`), and A11 sigmoidste was unstable (`+0.041`). Exact BPBs for A01, A08, A10, and A11 are unavailable, so they are not entered as hard values here, but the old blanket claim that `A09–A11` simply "crashed" was too coarse.

---

## Day 23 — April 9: The Size Wall

> **Goal:** Confirm multi-crawler at production scale. Fit 8F+3C or 9F+2C within the 16MB artifact cap.

Everything converged toward confirming the multi-crawler breakthrough. A **Layer Relationship Grid** (2×GPU, 1000 steps, seed=444) mapped the full 5×4 surface of flat layers × crawler layers:

| Config | int6_sw | Delta vs 9F+1C | step_ms |
|--------|---------|-----------------|---------|
| **8F+3C** | **1.39529** | **−0.01727** | 181 |
| 7F+4C | 1.39547 | −0.01709 | 206 |
| 9F+3C | 1.39599 | −0.01657 | 198 |
| 7F+3C | 1.39910 | −0.01346 | 176 |
| 6F+4C | 1.40000 | −0.01256 | 207 |
| 9F+1C (ctrl) | 1.41256 | — | — |

8F+3C was the quality peak. Three crawler layers matched three loops — a potential symmetry law. Unfortunately, 8F+3C broke the bank by 180KB, and the unmodified 7F+3C missed the MB limit by only 70KB. We cooked on the 7F+3C, and that became the final 4-hour (14.2MB) run candidate and the best 10-minute (15.9MB) runner.

The complete grid, including sizes and step times, was:

| Config | int6_sw | step_ms | Size |
|--------|---------|---------|------|
| 9F+1C | 1.41256 | 135.12 | 12.67MB |
| 9F+2C | 1.39778 | 160.13 | 14.12MB |
| 9F+3C | 1.39599 | 197.68 | 15.33MB |
| 9F+4C | 1.40689 | 229.37 | 17.06MB (OVER) |
| 8F+1C | 1.41419 | 133.83 | 11.76MB |
| 8F+2C | 1.40593 | 170.09 | 13.05MB |
| **8F+3C** | **1.39529** | **181.43** | **14.58MB** |
| 8F+4C | 1.39823 | 220.71 | 15.96MB |
| 7F+1C | 1.42120 | 111.88 | 10.73MB |
| 7F+2C | 1.40816 | 154.98 | 12.28MB |
| 7F+3C | 1.39910 | 175.87 | 13.79MB |
| 7F+4C | 1.39547 | 206.39 | 15.01MB |
| 6F+1C | 1.42609 | 105.41 | 9.75MB |
| 6F+2C | 1.41689 | 144.72 | 11.13MB |
| 6F+3C | 1.40834 | 177.49 | 12.54MB |
| 6F+4C | 1.40000 | 207.06 | 13.90MB |
| 5F+1C | 1.43142 | 91.72 | 8.54MB |
| 5F+2C | 1.41603 | 124.19 | 9.86MB |

An isolated 4×GPU confirmation of the grid winner landed **8F+3C at 1.34632 int6_sw and 15.00MB**, validating the surface outside the batch sweep.

A **Symmetry Ablation** (4×GPU, 1000 steps, seed=444) tested higher orders directly:

| Arm | Config | step_ms | val_bpb (300s gate) | Verdict |
|-----|--------|---------|---------------------|---------|
| **A0** | **8F+3C, 3 loops** | **547** | **3.2425** | **WINNER** |
| A1 | 8F+4C, 4 loops | 806 | 3.3202 | loses both axes |
| A2 | 8F+6C, 6 loops | — | FAILED | disk full cascade |
| A3 | 8F+8C, 8 loops | — | FAILED | disk full cascade |

Available evaluation tails preserved the same ordering: A0 finished at `3.29162042` int6 sliding-window, while A1 at least reached `3.35664470` int6 roundtrip before the log cut off. The margin was decisive. The 4th loop costs roughly 0.8s per step and produces worse quality. The separate 4×GPU symmetry gate for 4×4 logged raw_bpb `1.3696`, int6 roundtrip `1.37830138`, and artifact `16,152,138` bytes (over cap by 152KB); the sliding-window eval for 4×4 did not complete. C=3 is the practical ceiling.

Then the production runs:

**BW_9F2C** (8×H100, 600s, seed=444):

| Metric | BWX 9F | BW_9F2C |
|--------|--------|---------|
| int6_sw BPB | 1.13868 | **1.13190** |
| Delta | — | **−0.00678** |
| Bytes | 15,239,617 | **16,857,961** |
| Legal? | YES | **NO (+860KB)** |

**Trapper_Keeper_1** (8F+3C, 8×H100, 600s, seed=444, no FA3):

| Metric | BWX 9F | TK1 |
|--------|--------|-----|
| int6_sw BPB | 1.13868 | **1.13527** |
| int6_sw BPB (GPTQ 60L) | — | **1.13418** (best TK1) |
| Delta (best) | — | **−0.00450** |
| Bytes | 15,239,617 | **17,948,983** |
| Legal? | YES | **NO (+1.95MB)** |

Both configurations beat the leader on quality. Both exceeded the 16MB artifact cap. TK1's pod lacked FA3 (157ms/step → only 3,811 steps); with FA3 it would have been ~110ms → ~5,450 steps → likely better quality. Brotli recompression might save 10–15% on size, making TK1 near-legal, but near-legal is still non-compliant.

The **Ratio Sweep v2** attempted a 9-arm comparison of flat:crawler ratios but was interrupted by an OOM kill after only B00–B02:

| Arm | Config | int8 BPB | Size |
|-----|--------|----------|------|
| B00 | 10F+0C (pure transformer) | 1.31961 | 15.43MB |
| B01 | 9F+1C (baseline) | 1.32794 | 14.63MB |
| B02 | 10F+1C | 1.32484 | 16.17MB |

B01 and B02 completed but are **invalid** — the test harness was rebuilt from scratch and missed 9 critical crawler env vars (LOOP_AWARE_GPTQ, MLP_LEAKY_SLOPE, CRAWLER_MLP_CHOKE_DIM, etc.), so any crawler-vs-flat comparison from that batch is non-interpretable. Only B00 (10F+0C, pure transformer — no crawler code affected) is possibly clean: 1.31961 int8 BPB at 15.43MB. The remaining 6 arms — including the critical 8F+2C and 7F+3C configurations — OOM killed before completing.

**midnight_GPTQ** tested the GPTQ bank fix on the neural-track Midnight 12L leader. The structural fix worked correctly (0 → 60 tensors quantized), but quality regressed by +0.00312 BPB. The model had apparently adapted around the broken GPTQ, so the fix did not promote.

**Midnight_Black** ran three times on 8×H100 (seed=444). Run 1 was misconfigured (wrong quant bits, sequential loader) and thrown out. Run 2a (cache=1): 1.10899 BPB, 16.44MB (over cap). Run 2b (cache=4): **1.10831 BPB**, 15.74MB. Both worse than champion (1.10568). **DOES NOT PROMOTE** — the 3-variable stack failed (+0.00263 BPB).

**Crawler_Katta** ran on 8×H100 (2000 steps, seed=444). The Euler control hit 1.24486 int6_sw at 110ms/step. RK2 fast (2 loops) was 7.4% faster (102ms) but regressed +0.00186 BPB — the speed gain did not offset the quality loss. RK4 and hybrid solvers crashed due to an implementation bug in the forward pass. Euler remained the preferred solver. A quick test into Runge-Kutta variants, nothing to see here.

**BW23_EcoConcept** (QAT surrogate variants) got a DGX Spark smoke test (`4.73` BPB roundtrip), but the run was too short to support a sliding-window conclusion. Its core concept was also tested through corpus arms A09 (QAT legacy STE: `+0.0085`) and A10 (softclamp: `+0.004`). QAT surrogates did not help on this architecture. **Crawler_Symmetry** (C=LOOPS design law, testing 4×4 through 8×8) and **TTT_Ablation** (end-to-end test-time training) remained unfinished as the campaign's final days concentrated on fitting the multi-crawler winners within the size constraint.

---

## Finish

The program ended with a stable crawler line built by rapid exploration, repeated falsification, and aggressive simplification. The original fully recursive thesis did not survive intact; the hybrid thesis did. What held up was a flat-depth scaffold around a small shared crawler bottleneck, with loops kept to three, EMA removed, loop-aware GPTQ treated as mandatory, and bytes treated as a first-class constraint rather than an afterthought. The legal `7F+3C` runner closed the competition problem, and the `7F+3C` 4-hour confirmation showed that the final architecture was not just a trick for the 10-minute harness. It was a stable compression-and-thinking system discovered under pressure through competition constraints and rapid iteration.

---

## Appendices

The supporting records below follow the campaign in time. The dated score ledger comes first. The keep/kill and disposition ledgers are also ordered by campaign date rather than by abstract category. Records recovered from dated logs but lacking trustworthy original run-day placement are separated at the end.

---

## Appendix A: Chronological Score Ledger

This appendix is the compact dated record of the major production, bridge, and late-sprint results.

| Date | Family / branch | Record | int6_sw BPB | Size | Status | Notes |
|------|-----------------|--------|-------------|------|--------|-------|
| 2026-03-22 | Frugendorff | Frug Squared | 1.1478 | 15.15MB | promoted | First H100 production result from the original recursive line |
| 2026-03-24 | H4 bridge | H4 bridge cluster best | 1.21451 | — | bridge evidence | Source-dated bridge-day work kept explicit so the chronology does not jump directly from catastrophe to closure |
| 2026-03-27 | Medusa | Medusa IV | 0.9984 mean / 0.8104 best | 9.96MB | volatile failure | DeltaNet crawler breakout with extreme cross-seed variance; important but non-promotable |
| 2026-03-29 | ClownCar | ClownCar X Cubric | 1.1860 sw / 0.4957 full (s444) | 9.21MB | side submission | Parallel hybrid submission track; informative but not the main crawler promotion line |
| 2026-03-30 | Crawler Legs | Crawler Leg 3 | 1.18720 (s1337) | 8.84MB | promoted | 3-seed mean 1.18743; loops=3, mlp=6.0 |
| 2026-03-30 | Bandit Wagon bridge | BW-00 Anchor | 1.18616 | 9.10MB | bridge run | 4F+1C bridge into BW series |
| 2026-04-01 | Bandit Wagon | BW10_GPTQ | 1.18293 (s444) | 9.96MB | promoted | Loop-aware GPTQ standalone production win |
| 2026-04-01 | Bandit Wagon | Nightcrawler / BW11 | 1.17651 (s444) | 10.05MB | promoted | 3-seed mean 1.1761; validates flat-depth scaling |
| 2026-04-02 | BWX | **BWX 9F** | **1.13868** | **15.24MB** | former legal leader | Former in-tree crawler leader before the 2026-04-10 legalization result |
| 2026-04-02 | BWX side branch | Micro crawler run8 | 1.13554 (s444) | ~16.6MB | quality pass, size fail | Beats BWX on quality, oversize |
| 2026-04-03 | Ouroboros | Ouroboros | 1.13727 (s444) | 15.03MB | research submission | 3-seed mean 1.1364; interesting branch, not a promotion over BWX |
| 2026-04-09 | BWX late branch | BW_9F2C | 1.13190 (s444) | 16.86MB | quality pass, size fail | Best recovered oversize descendant (+860KB) |
| 2026-04-09 | TK1 | TK1 (8F+3C) | 1.13527 / 1.13418 GPTQ | 17.95MB | quality pass, size fail | Strong late descendant, but far over cap |
| 2026-04-10 | TK1 | Trapper Keeper 1 (8F+3C, s444) | 1.13541288 | 15.90MB | legal full run | First full legal seed of the final TK1 line |
| 2026-04-10 | TK1 | Trapper Keeper 1 (8F+3C, s300) | 1.13853446 | 15.85MB | legal full run | Second legal seed; slower quality than the best legal seed |
| 2026-04-10 | TK1 | Trapper Keeper 1 (8F+3C, s4) | 1.13536063 | 15.84MB | current legal leader | Best legal full run at 15,844,157 bytes |
| 2026-04-10 | TK1 | TK1 7F+3C | 1.13678 | 16.07MB | near-miss size fail | Smaller config, still over by about 65KB |
| 2026-04-10 | TK1 | TK1 7F+3C + GPTQ + pyminify | 1.13418 | 16.03MB | near-miss size fail | Best recovered pre-legalization late variant, over by about 32.9KB |
| 2026-04-10 | Final sprint | Trapper Keeper 1 (8F+3C) | 1.13536063 | 15.84MB | current legal leader | Legalized runner confirmed across three full seeds |
| 2026-04-11 | Final confirmation | Nightcrawler Cubed (7F+3C, s4) | 1.07424983 | 14.18MB | 4-hour confirmed | Organizer-facing long-run confirmation completed with a legal artifact |
| 2026-04-10 | Neural side track | Midnight_Black | 1.10831 (s444) | 15.74MB | side comparator | Legal side track, but not a crawler promotion |

---

## Appendix B: Campaign Decision Ledger

This appendix preserves the keep/kill record in campaign order rather than collapsing it into a short summary.

**Rejected, invalid, or non-promoted directions (campaign order):**

| Date | Hypothesis / direction | Result | Evidence |
|------|------------------------|--------|----------|
| 2026-03-27 | DeltaNet crawler line (`Medusa`) | best seed 0.8104, mean 0.9984, std 0.1724; unstable | Medusa IV record + Medusa_VII cleanup |
| 2026-03-29 | DeltaNet (`DELTA_NET_HEADS=4`) | EMA degradation, causality bugs | ClownCar_II |
| 2026-03-29 | EMA (`SKIP_EMA=0`) | +0.070 to +0.101 BPB | Crawler_Ablations_v1 |
| 2026-03-29 | inst_dim=0 (off) | +0.034 BPB | CL1-03 |
| 2026-03-29 | mlp_mult=3.0 | +0.116 BPB | CL1-06 |
| 2026-03-29 | loops=5 | +0.069 BPB | CL1-02 |
| 2026-03-29 | `CRAWLER_QUANT_INT8=0` | +0.198 BPB | CL1-08 |
| 2026-03-29 | 3F+2C split | +0.120 BPB | CL1-10 |
| 2026-03-30 | loops=2 | +0.004 BPB | CL2-03 |
| 2026-03-31 | Pyramid-512 choke (full run) | +0.020 BPB | BW3 |
| 2026-03-31 | PyramidCannon combined | +0.034 BPB | BW5_PyramidCannon |
| 2026-03-31 | SharedFlat (`FLAT_WEIGHT_SHARE=1`) | +0.037 BPB | BW7 FLAT-07 |
| 2026-03-31 | SMEAR | −0.00003 (null) | BW7 SMEAR-01 |
| 2026-04-01 | Width expansion (dim=576,640) | +0.017 to +0.049 BPB | BW-01, BW-02 |
| 2026-04-01 | Trigram hash | +0.00014 (null) | BW6_Skipgram |
| 2026-04-01 | Scalar cannon at scale | +0.00020 (reversed) | BW5_Cannon |
| 2026-04-02 | Anchor on tap-off | +0.003 (regression) | BW13 |
| 2026-04-02 | Crawler choke flat-128 | +0.012 BPB | BW14BS-02 |
| 2026-04-02 | 10F (isolated) | worse + over 16MB | BW16 |
| 2026-04-04 | Helix SplitHead at full scale | +0.140 BPB | Helix_ab_3 |
| 2026-04-04 | BWXII Helix SplitHead (9F) | 1.25031 (regressed) | BWXII production |
| 2026-04-04 | Depth recurrence without Helix | +0.089 BPB | Helix_DepthRecur R1 |
| 2026-04-04 | QK gain 5.0/6.0 on crawler | WORSE | Helix_SplitHead micro |
| 2026-04-04 | 7F with Helix | All arms worse than 5F | Helix_SplitHead |
| 2026-04-05 | Ouroboros stacked (all 3 arms) | 1.245 (vs 1.161 individual) | Ouroboros_stacked |
| 2026-04-06 | Ouroboros III (stacked) | +0.006 BPB | Ouroboros_III |
| 2026-04-07 | Crawler_Katta RK2 (2 loops) | +0.00186 BPB, 7% faster | Throughput doesn't compensate |
| 2026-04-07 | Crawler_Katta RK4/hybrid | CRASHED | Implementation bug in solver |
| 2026-04-08 | INST_DIM=64 | +0.001 BPB | Corpus A06 |
| 2026-04-08 | QAT legacy STE | +0.0085 vs control | Corpus A09 |
| 2026-04-08 | QAT sigmoidste | +0.041 BPB | Corpus A11 |
| 2026-04-09 | Width expansion (dim=624,640) | 1.1375 to 1.1377 BPB, both 16.6 to 16.9MB oversize | Micro crawler runs |
| 2026-04-09 | Symmetry 4×4 (8F+4C, 4 loops) | 3.3202 vs 3.2425 control, 806ms/step | Crawler_Symmetry |
| 2026-04-09 | GPTQ bank fix (Midnight 12L) | +0.00312 BPB | midnight_GPTQ |
| 2026-04-09 | Midnight_Black (3-signal stack) | +0.00263 BPB | 1.10831 vs 1.10568 champ |

**Confirmed positive or durable findings (campaign order):**

| Date | Finding | Delta / result | Evidence |
|------|---------|----------------|----------|
| 2026-03-29 | Loop-aware GPTQ | −0.040 | Crawler_Ablations_v1 + BW10 (−0.00380 production) |
| 2026-03-29 | COMPILE_FULLGRAPH=1 | −0.026 | Crawler_Ablations_v1 |
| 2026-03-29 → 2026-03-30 | loops=3 (from 4) | −0.088 | CL1 + CL2 confirmed |
| 2026-03-29 → 2026-03-30 | mlp=5.0+ | −0.098 | CL1 + CL2 confirmed |
| 2026-03-31 | Battery ROPE 9,1,1 | −0.00015 | BW4 |
| 2026-04-09 | C=3 symmetry (3 crawlers × 3 loops) | optimal | Grid + symmetry ablation confirmed |

**Quality passes blocked by the 16MB size cap:**

| Date | Config | int6 SW BPB | Size | Delta vs BWX |
|------|--------|-------------|------|-------------|
| 2026-04-02 | Micro crawler run8_pd_cad2 | **1.13554** | ~16.6MB | **−0.00314** |
| 2026-04-09 | BW_9F2C | **1.13190** | 16.86MB | **−0.00678** |
| 2026-04-09 | TK1 8F+3C (no GPTQ) | **1.13527** | 17.95MB | **−0.00341** |
| 2026-04-09 | TK1 8F+3C + GPTQ 60L | **1.13418** | ~17.9MB | **−0.00450** |
| 2026-04-10 | TK1 7F+3C | **1.13678** | 16.07MB | **−0.00190** |
| 2026-04-10 | TK1 7F+3C + GPTQ + pyminify | **1.13418** | 16.03MB | **−0.00450** |

**Open leads (untested at production scale):**

| Date | Finding | Delta | Evidence | Why open |
|------|---------|-------|---------|----------|
| 2026-03-22 | Legal TTT N× leverage | −0.032 | Frugendorff v3+TTT (score-first legal) | Discarded prematurely during an overcorrection. Preserved at `vault/preserved_ttt/` |
| 2026-04-04 | Weight decay 0.12 on crawler | −0.012 | Helix_SplitHead micro | Untested at production scale |
| 2026-04-06 | loops=2 (RAPID proxy) | −0.054 directional | BW17 DGX Spark | Unconfirmed at full scale |
| 2026-04-08 | 4 loops diff battery | −0.0046 | Corpus A04 | Untested on production 8F+3C base |

---

## Appendix C: Experiment Disposition Ledger

This appendix records what ran, what was invalid, what was shelved, and what was absorbed into later work, in campaign order.

| Date | Experiment | Concept | Actual disposition |
|------|-----------|---------|--------------------|
| 2026-03-27 → 2026-03-29 | Medusa series | DeltaNet crawler continuation of Frugendorff | Ran and produced one of the most volatile positive signals in the corpus: 0.8104 best seed, 0.9984 mean. A state-dtype bug plus quantization unravel made the branch non-promotable. The cleaned `Medusa_VII` line fed the March 29 submission path, but DeltaNet itself was later quarantined. |
| 2026-03-28 | Cobra | 9-config base-quality racecar | Never ran. BW-series progress made it obsolete before scheduling; its target questions were answered faster by targeted crawler ablations. |
| 2026-04-02 | BW16_WidthSweep | MODEL_DIM beyond 512 | Never ran as a formal sweep. Micro crawler later tested dim=624 (1.1375, 16.65MB oversize) and dim=640 (1.1377, 16.86MB oversize). Width killed size budget; depth won instead. |
| 2026-04-02 | BW18_DeltaMatrix | 49-arm knob sweep | Never ran as a standalone campaign. The knobs were effectively absorbed into BW7, BW12-BW14, and bridge ablations. |
| 2026-04-02 | BW19_CrawlerSystem | 18-arm balance study | Never ran. The intended questions were folded into BW12-BW14 interaction gates and later corpus ablations. |
| 2026-04-02 | BWXI_Brotli_GPTQ | 5-signal stack on BWX 9F | Never ran. Shelved after Ouroboros III showed 3-signal stacking already regressed; a 5-signal stack was unjustifiable risk. |
| 2026-04-03 | Helix_FlatServe | Flat layers optimized for crawler | Never ran. SplitHead results redirected Helix effort to cross-attention and then away from Helix entirely. |
| 2026-04-03 | DarkHorse | Helix on PR #1296 codebase | Never ran. Code existed, but once Helix_ab_3 showed poor transfer, the port was no longer worth the integration risk. |
| 2026-04-06 | TTT_ablation | End-to-end test-time training | Sweep launched on the parameter-golf-lab side (11 configs). Shape mismatch killed the first pass. Second attempt captured two partial gate BPBs, 1.24373 and 1.24398 (seed 444). Full production eval never completed. |
| 2026-04-06 | BW21_NoisyQAT | Noise-injection QAT on 9F | Never ran on 9F. The only supporting signal was a different-base Ouroboros ablation (−0.00296), and the campaign deprioritized it once the late multi-crawler line opened. |
| 2026-04-07 | Crawler_Katta | RK solver variants | Ran on 8×H100 and 1×GPU. Euler control: 1.24486 int6_sw at 110ms. RK2: 1.24672 (+0.00186) at 102ms. RK4/hybrid crashed. Did not promote. |
| 2026-04-08 | BW23_EcoConcept | QAT surrogate + mixed-bit | DGX Spark smoke only (4.73 BPB roundtrip, no sliding-window eval). Related corpus arms A09 (+0.0085) and A10 (+0.004) also failed to justify promotion. |
| 2026-04-09 | Ratio sweep B03-B08 | Remaining flat:crawler ratios | B03 launched (1.34920 int6), but every crawler arm in the rebuilt harness was invalid because 9 crawler env vars were missing. B04-B08 OOM-killed. Only B00 pure transformer is possibly clean. |
| 2026-04-09 | Crawler_Symmetry | C=LOOPS design-law test | Ran on 4×GPU. A0 (3×3 control) hit 3.2425 BPB at 547ms. A1 (4×4) hit 3.3202 at 806ms, worse on both axes. A2/A3 (6×6, 8×8) ended in a disk-full cascade. |
| 2026-04-09 | Midnight_Black | 3-signal aggressive stack | Ran three times on 8×H100. Best clean result: 1.10831 BPB, +0.00263 vs champion. Did not promote. |

---

## Appendix D: 2026-03-31 Proxy Ablation Battery (500-step, seed=444)

46 arms covering every crawler infrastructure knob. This is the foundational dataset that informed BW7 MegaGate and all subsequent architecture decisions. Best results first.

**Choke Shape (BWC):** BWC-04 choke=512 (1.42887 BEST), BWC-02 choke=128 (1.43674), BWC-03 choke=256 (1.44071), BWC-01 choke=32 (1.45004 WORST).

**Tap Configuration (BWT):** BWT-05 dim=32 per-loop deep (1.43004), BWT-01 dim=32 shared all (1.43227), BWT-03 dim=16 per-loop (1.43268), BWT-06 dim=32 per-loop shallow (1.43322), BWT-02 dim=32 per-loop all (1.44133), BWT-04 dim=64 per-loop all (1.44346).

**Battery Schedule (BWB):** BWB-01 1,2,4 gentle asc (1.43769), BWB-04 9,3,1 desc (1.44156), BWB-05 1,9,1 middle (1.44237), BWB-03 1,5,25 aggressive (1.44283), BWB-07 9,1,1 first wide (1.44355), BWB-02 1,3,9 moderate asc (1.44470), BWB-06 1,1,9 final wide (1.44797).

**Pyramid + RoPE (BWCD):** BWCD-02 rope 9,1,1 (1.43531), BWCD-01 rope 4,2,1 (1.43749), BWCD-00 rope 9,3,1 desc (1.43779), BWCD-03 rope 9,3,9 wide-med-wide (1.44248).

**Choke + Battery (BWCB):** BWCB-00 rope 1,2,4 (1.44850), BWCB-02 rope 1,5,25 (1.44864), BWCB-01 rope 1,3,9 (1.44874).

**Choke Shape Variants (BWCS):** BWCS-02 pyramid dim=512 (1.44724), BWCS-06 residual dim=128 (1.45260), BWCS-03 pyramid_res dim=128 (1.45419), BWCS-01 pyramid dim=128 (1.45711), BWCS-05 grouped dim=512 groups=4 (1.45748), BWCS-00 control flat (1.45761), BWCS-04 grouped dim=512 groups=8 (1.46247).

**Cannon Variants (BWE/BWVC):** BWE-02 channel 1.5K (1.43590), BWE-00 control (1.44166), BWVC-00 control (1.44236), BWVC-01 scalar 3 params (1.44261), BWVC-02 channel 1.5K (1.44296), BWE-01 scalar 3 params (1.44337), BWVC-03 rmsnorm 1.5K (1.44428).

**Other:** BWS-01 loop smear (1.44628), CTRL-00 all disabled (1.44185).

**XSA Coverage:** BWXSA-02 XSA=15 100% (1.51431), BWXSA-01 XSA=13 87% (1.51982), baseline XSA=11 73% (1.52365).

**MLP Slope:** 0.75 (1.55637 best, −0.00065), 0.5 control (1.55702), worst (1.56116, +0.00413). All near-identical.

**Depth:** 4F+1C (1.52365), 5F+1C (1.54404), 6F+1C (1.56887). Depth beats width at every point.

---

## Appendix E: 2026-04-03 BWXII QuantFix Full Sweep

Weight decay sweep on 1×GPU with GPTQ layer variants:

| Config | 54 GPTQ | 60 GPTQ |
|--------|---------|---------|
| **T4 smart_wd012** | **1.27427** (BEST) | 1.27670 / 1.27730 |
| T0 wd=0.12 | 1.27719 | 1.28000 / 1.28015 |
| T1 wd=0.15 | — | 1.29058 / 1.29080 |
| T2 wd=0.20 | 1.28900 | 1.29231 / 1.29239 |
| T3 wd=0.25 | 1.30689 | 1.31026 / 1.31032 |

Movie test: R1_fire_embed 1.28121 (gap 0.0225), R4_full_fix 1.28033 (gap 0.0383). Full_fix improved raw BPB but widened quant gap.

---

## Appendix F: 2026-03-18 to 2026-03-22 Cadence Characterization (H1/H2 series)

| Config | cad=1 | cad=2 | cad=3 | cad=4 |
|--------|-------|-------|-------|-------|
| H1 (4F+2C×2) | 1.50919 | 1.42221 | 1.39411 | 1.38358 |
| H2 (3F+3C×2) | 1.60068 | 1.45872 | 1.42107 | 1.40304 |

H1_cad0_FULLSCALE (no cadence, every step fractal): 1.16029 — best quality when throughput cost is ignored.
H2_2f4cx2_cad4: 4.30781 — DIVERGED. Too much recursion.

---

## Appendix G: Source-Dated Historical Recoveries

These runs were recovered from dated session logs, records, and archive inventories. The source date is known in every row below. What still varies is how tightly each run can be placed on the main crawler spine: some are direct Frugendorff antecedents, some are bridge experiments, and some are adjacent side branches.

**Frugendorff antecedents and adjacent recovered records:**

| Source date | Campaign relation | BPB | Experiment | Evidence |
|-------------|-------------------|-----|-----------|----------|
| 2026-03-22 | pre-crawler Frugendorff line | 1.24533560 | train_gpt_min.py / fractal_h100 | dated run transcript capture (`fa168c83-cdfb-4c09-9032-2761810286fe.txt`) |
| 2026-03-22 | pre-crawler Frugendorff line | 1.27147320 | train_gpt_fractal_h100.py | dated run transcript capture (`d31921e8-8344-417d-9a93-f54180d5a21d.txt`) |
| 2026-03-22 | pre-crawler Frugendorff line | 1.32133176 | train_gpt_fractal_h100.py | dated run transcript capture (`c1539c60-b928-4a45-a69b-aa7209f61ff4.txt`) |
| 2026-03-22 | pre-crawler Frugendorff line | 1.21184021 | train_gpt_fractal_h100_v4.py | dated run transcript capture (`e6bc18b0-0818-460e-9321-5a864a919e3e.txt`) |
| 2026-03-22 | pre-crawler Frugendorff line | 1.17566115 | train_gpt_fractal_h100_v5.py | dated run transcript capture (`edb680e7-5a4c-4552-9c22-7b9886b6f7fa.txt`) |
| 2026-03-23 | Frugendorff submission line | 1.14782318 | Frugendorff Squared (6x2, 640d, MLP4) | `records/track_10min_16mb/2026-03-23_Frugendorff_Squared_6x2_640d_MLP4/submission.json` |
| 2026-03-23 | Frugendorff archive inventory | 1.14292439 | GPTQ_42layers (Frugendorff) | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-23 | Frugendorff archive inventory | 1.14780933 | Frugendorff (pr374 branch) | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-23 | Frugendorff archive inventory | 1.21111811 | train_gpt_fractal_h100.py | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-23 | Frugendorff archive inventory | 1.21131577 | train_gpt_fractal_h100_v2/v3_ttt.py | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-23 | Frugendorff archive inventory | 1.21221793 | Frugendorff | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-23 | Frugendorff archive inventory | 1.21857814 | train_gpt_fractal_h100_v5.py | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-23 | Frugendorff archive inventory | 1.27489099 | GPTQ_35layers (Frugendorff) | listed in a dated 2026-03-23 pod-archive inventory extraction |
| 2026-03-26 | adjacent side branch, not crawler line | 1.87501360 | A-WING GREEN_3 Width 640 | dated 2026-03-26 run transcript (`A-WING GREEN_3 — Width 640`) |

**H4 bridge runs recovered from dated logs:**

| Source date | Campaign relation | BPB | Experiment | Evidence |
|-------------|-------------------|-----|-----------|----------|
| 2026-03-24 | bridge work before later multi-crawler grids | 1.21450611 | H4A_gsv7_control | `H4A_gsv7_control_20260324_201010.txt` |
| 2026-03-24 | bridge work before later multi-crawler grids | 1.23711457 | H4B_gsv7_crawler_bank | `H4B_gsv7_crawler_bank_20260324_201746.txt` |
| 2026-03-24 | bridge work before later multi-crawler grids | 1.35077091 | H4_A — 6 flat, 0 crawler | `H4_A_6flat_20260324_234630.txt` |
| 2026-03-24 | bridge work before later multi-crawler grids | 1.35134167 | H4_B — 5 flat + 1 crawler x2 at bottleneck | `H4_B_5f1cx2_btn_20260324_235155.txt` |

---

*Research journal compiled 2026-04-10. Source: parameter-golf-lab (crawler/, junkyard/, records/, legs/), sota_nueral (legs/, records/), git history, pod logs.*


TO WRAP EVERYTHING UP - THANKS FOR READING FELLOW HUMANS!

*Hope: to work on AI research or visualizations with an amazing team. 
