# Structured Experiment Summary

This is kind of a summary of (maybe not all, lol) the experimentation I did — lots of learning. Thanks to OpenAI for the $525 + a few hundred dollars of my own credits! I’d love to try more ideas I had but couldn’t due to a shortage of credits.

In this journey, I tried not to get bogged down by leaderboard approaches as much as possible. In a few places, though, when I got stuck, I did take help from the community. My general approach was: train a model → analyze it → try to solve the issues observed in the analysis. This ended up costing me many experiments and dollars.

GIT REPO TO FIND ALL EXPERIMENTS:
https://github.com/SPThole/parameter-golf-experimentations

I have also made a cool mind map of all the experimentation — basically the path of what I did and why. I’ve also attached lineages that are relevant from community discussions and leaderboard files.

I am planning to build on this:
https://github.com/SPThole/bpb_wtf or visit: https://bpb-wtf.vercel.app/

I’m also building a broader direction around this (mind map + experiments). If this resonates with anyone or you’d like to collaborate, feel free to reach out — I’d love to explore this further together.

> **Competition**: OpenAI Parameter Golf
> **Objective**: Minimize validation loss (bits-per-byte, bpb) under a 16MB artifact constraint within 10-minute training on 8×H100
> **Total experiments**: 119+
> **Date range**: Early 2026 — 2026-04-13
> **Best result**: **1.0744 legal_ttt bpb** (ImprovedParallelResiduals, community PR #1523, 8×H100)

---

# Table of Contents

1. [Global Leaderboard](#1-global-leaderboard)
2. [Phase 3a: Baseline Experiments (exp00–exp18)](#2-phase-3a)
3. [Phase 3b-Part1: Systematic Ablations (exp27b–exp33b)](#3-phase-3b-part1)
4. [Phase 3b-Part2: LR Fix Era (exp34b–exp48b)](#4-phase-3b-part2)
5. [Phase 3b-Part3: Simplification + XSA (exp53b–clean_54b)](#5-phase-3b-part3)
6. [Phase 3.5: 8×H100 Simulation (exp60–exp80)](#6-phase-35)
7. [Phase 3.6: Diagnostic-Driven Era (exp83–exp87)](#7-phase-36)
8. [Phase 3b-Muon: Parallel Muon Optimizer (exp70_parallel_muon–exp91)](#8-phase-3b-muon)
9. [Phase 3c: Architecture Rewrite + Meta-TTT (exp92–exp109)](#9-phase-3c)
10. [Phase 3c-Community: Community SOTA (SP8192+)](#10-community-sota)
11. [Phase 3c-Frontier: Pushing Past Community (exp110–exp119)](#11-frontier)
12. [Misc: Co-occurrence QK Init](#12-misc)
13. [Known Issues](#13-known-issues)
14. [Key Learnings by Phase](#14-key-learnings)
15. [TLDR: Top 20 Learnings](#15-top-20)
16. [Appendix: Full Lineage Trees](#16-appendix)

---

# 1. Global Leaderboard

## 1.1 All-Time Best (by legal_ttt bpb)

| Rank | Experiment | Date | legal_ttt | val_bpb | int6_bpb | Artifact | Hardware | Source |
|:----:|-----------|:----:|:---------:|:-------:|:--------:|:--------:|:--------:|:------:|
| **1** | ImprovedParallelResiduals | 2026-04-11 | **1.0744** | — | — | 15.96 MB | 8×H100 | Community PR #1523 |
| 2 | WiderEmb_TapInV6_TTT | 2026-04-10 | 1.0788 | 1.0813 | 1.0980 | ~16 MB | 8×H100 | Community |
| 3 | SP8192_3LayerRecur | 2026-04-09 | 1.0808 | 1.0873 | 1.0997 | ~16 MB | 8×H100 | Community |
| 4 | exp101 | 2026-04 | 1.11588 | 1.1352 | 1.13930 | ~16 MB | 8×H100 | Our work |
| 5 | exp95 | 2026-03 | 1.1169 | 1.1363 | — | ~16 MB | 8×H100 | Our work |
| 6 | exp74 | 2026-03 | — | 1.1539 | 1.1685 | 15.86 MB | 1×H100 (sim) | Our work |
| 7 | exp54b | 2026-03 | — | 1.2642 | 1.2708 | 15.54 MB | 1×H100 | Our work |

## 1.2 Milestone Timeline

| Date | Best legal_ttt | Experiment | Key Innovation |
|------|:--------------:|-----------|----------------|
| Early | 1.3389 (quant) | exp00 baseline | Starting point |
| Early | 1.3145 (quant) | exp09 | Step count + loss masking |
| Phase 3b | 1.2708 (quant) | exp54b | LR fix + simplification |
| 2026-03 | 1.1456 (sliding) | exp74 | Partial RoPE + diagnostics |
| 2026-03 | 1.1169 | exp95 | Meta-TTT + size optimization |
| 2026-04-04 | 1.11588 | exp101 | Position-conditional bigram |
| 2026-04-09 | 1.0808 | SP8192_3LayerRecur (community) | SP8192 + depth recurrence |
| 2026-04-10 | 1.0788 | WiderEmb_TapInV6 (community) | Wider loop + Tap-In V6 |
| **2026-04-11** | **1.0744** | **ImprovedParallelResiduals (community, PR #1523)** | **Cross-lane parallel residuals** |

---

# 2. Phase 3a: Baseline Experiments (exp00–exp18)

> **Hardware**: 1×H100 (or A100), 600s wallclock
> **Base**: exp27 (modded-nanogpt reference)
> **Best result**: exp09/exp13 — quant bpb **1.3145**, artifact 14.5 MB

## 2.1 Config Constants

| Parameter | Value |
|-----------|-------|
| Model dim | 512 |
| Num layers | 11 (10 unique, layer sharing) |
| Attention | GQA, 8 Q-heads, 4 KV-heads, head_dim=64 |
| MLP | LeakyReLU², mlp_mult=3.0 (hidden=1536) |
| Vocab size | 1024 (SentencePiece BPE) |
| Seq length | 2048 |
| Softcap | 30 |
| Optimizer | Muon (matrices) + Adam (scalars) |
| Momentum | Cyclic 0.85–0.95, period=50 |
| Grad accum | 2 |
| SWA | Start at 20% training, every 100 steps |
| AWQ alpha | 0.6 |
| Quantization | int6 + zstd |
| Wallclock | 600s (10 min) |
| Total params | ~25.5M |

## 2.2 Leaderboard

| Rank | Exp | Name | Quant BPB | Raw BPB | Artifact | Under 16MB? |
|:----:|:---:|------|:---------:|:-------:|:--------:|:-----------:|
| **1** | **54b** | xsa-zstd-ckfix | **1.2708** | 1.2642 | 15.54 MB | **Yes** |
| 2 | 53b | lean-combo (v5) | 1.2720 | 1.2640 | 15.19 MB | Yes |
| 3 | community | SOTA leaderboard (1xH100) | 1.2825 | 1.2501 | 13.06 MB | Yes |
| 4 | 48b | 10blocks-depth | 1.2930 | 1.2870 | 14.59 MB | Yes |
| 5 | 42b | revive-block9 | 1.2969 | 1.2867 | 14.01 MB | Yes |
| 6 | 34b | lr-schedule-fix | 1.2990 | 1.2891 | 15.13 MB | Yes |
| 7 | 39b | swa-tuning | 1.2942 | 1.2875 | 14.55 MB | Yes |
| 8 | 30b | combo | 1.3156 | 1.2983 | 15.05 MB | Yes |
| 9 | 29b | lossweight-typemb | 1.3176 | 1.3007 | 15.75 MB | Yes |
| 10 | 27b | resid-norm | 1.3197 | 1.3000 | 15.30 MB | Yes |
| 11 | 09 | padignore-wordboost | 1.3145 | 1.2974 | 14.5 MB | Yes |
| 11 | 13 | multihead-gate-bigram | 1.3145 | 1.2974 | 14.5 MB | Yes |
| 13 | 10 | trigram-unigram | 1.3151 | 1.2956 | 15.6 MB | Yes |
| 14 | 06 | swa-awq-accum2 | 1.3161 | 1.2982 | 15.7 MB | Yes |
| 15 | 07 | tighter-swa-awq | 1.3164 | 1.2978 | 15.5 MB | Yes |
| 16 | 05 | grad-accum4 | 1.3181 | 1.3001 | 15.8 MB | Yes |
| 17 | 12 | trigram64-awq06 | 1.3222 | 1.2969 | 15.1 MB | Yes |
| 18 | 08 | ctx-freq-bias | 1.3225 | 1.3014 | 15.0 MB | Yes |
| 19 | 18 | separate-trigram64 | 1.3247 | 1.2995 | 15.0 MB | Yes |
| 20 | 11 | trigram-slim-awq07 | 1.3259 | 1.2994 | 14.6 MB | Yes |
| 21 | 15 | engram-3order | 1.3260 | 1.2995 | 14.6 MB | Yes |
| 22 | 14 | engram-multiorder | 1.3338 | 1.3056 | 15.0 MB | Yes |
| 23 | 00 | baseline-rerun | 1.3389 | 1.3166 | 14.7 MB | Yes |
| 24 | 02 | speed-bigramfp16-awq | 1.3429 | 1.3200 | 17.3 MB | **No** |
| 25 | 04 | no-cyclic-momentum | 1.3489 | 1.3230 | 15.8 MB | Yes |
| 26 | 16 | jepa-aux | 1.3526 | 1.3194 | 14.5 MB | Yes |
| 27 | 17 | byte-engram | 1.3527 | 1.3201 | 14.6 MB | Yes |
| 28 | 01d | xsa-only | 1.3568 | 1.3294 | 14.8 MB | Yes |
| 29 | 03 | qat-ste | 1.3684 | 1.3365 | 15.7 MB | Yes |
| — | 01b | ln-scale-only | — | — | — | Not run |
| — | 01c | ema-only | — | — | — | Not run |

## 2.3 Detailed Experiment Cards

### Exp00 — Baseline Rerun (exp27)

| Field | Value |
|-------|-------|
| **Folder** | `exp00_baseline-rerun_exp27/` |
| **Based on** | exp27 (modded-nanogpt reference) |
| **Status** | Done |
| **Steps** | 1,084 |
| **Raw BPB** | 1.3166 |
| **Quant BPB** | 1.3389 |
| **Artifact** | 14.7 MB |

**Details**: Control experiment. Runs the exp27 reference config on A100 with 600s wallclock, grad_accum=8, GQA 8Q/4KV heads, 25.5M params, int6+zstd quantization.

**Observations**: ~553ms/step. Quant gap = 1.7%. Bigram.proj has worst quantization error. Still improving at end — more steps would help.

---

### Exp01b — LN Scale Only (ablation)

| Field | Value |
|-------|-------|
| **Folder** | `exp01b_ln-scale-only_from-exp27/` |
| **Based on** | exp27 |
| **Status** | Not run (empty log) |

**Details**: Isolated test of `1/√(layer+1)` layer-norm damping without partial RoPE.

---

### Exp01c — EMA Only (ablation)

| Field | Value |
|-------|-------|
| **Folder** | `exp01c_ema-only_from-exp27/` |
| **Based on** | exp27 |
| **Status** | Not run (empty log) |

**Details**: Isolated test of EMA weight averaging without other changes.

---

### Exp01d — XSA Only (ablation)

| Field | Value |
|-------|-------|
| **Folder** | `exp01d_xsa-only_from-exp27/` |
| **Based on** | exp27 |
| **Status** | Done |
| **Steps** | 1,017 |
| **Raw BPB** | 1.3294 |
| **Quant BPB** | 1.3568 |
| **Artifact** | 14.8 MB |

**Details**: Cross-sequence attention on last 4 layers, keeping SWA and full RoPE. Slower per step (~590ms vs 553ms baseline), fewer steps completed.

**Observations**: XSA alone hurts — slower steps + no benefit = regression. The speed cost outweighs any representational gain.

---

### Exp02 — Speed + Bigram FP16 + Better AWQ

| Field | Value |
|-------|-------|
| **Folder** | `exp02_speed-bigramfp16-awq_from-exp00/` |
| **Based on** | exp00 |
| **Status** | Done |
| **Steps** | 1,080 |
| **Raw BPB** | 1.3200 |
| **Quant BPB** | 1.3429 |
| **Artifact** | 17.3 MB |

**Details**: Three changes combined: (1) muon_backend_steps=4 (was 5), val_loss_every=200 for speed; (2) bigram.proj kept in FP16 instead of quantized; (3) per-category AWQ alphas (bigram=0.75, attn=0.6, mlp=0.5) with 16 calibration batches.

**Observations**: FP16 bigram blows artifact to 17.3MB — over budget. Per-category AWQ alpha interesting but gains washed out.

---

### Exp03 — QAT-STE (Quantization-Aware Training)

| Field | Value |
|-------|-------|
| **Folder** | `exp03_qat-ste_from-exp00/` |
| **Based on** | exp00 |
| **Status** | Done |
| **Steps** | 1,035 |
| **Raw BPB** | 1.3365 |
| **Quant BPB** | 1.3684 |
| **Artifact** | 15.7 MB |

**Details**: Fake-quantize weights during warmdown phase using Straight-Through Estimator.

**Observations**: Worst result of the batch. QAT-STE adds overhead (fewer steps) and destabilizes training. SWA+AWQ already handles quant gap reasonably.

---

### Exp04 — No Cyclic Momentum

| Field | Value |
|-------|-------|
| **Folder** | `exp04_no-cyclic-momentum_from-exp00/` |
| **Based on** | exp00 |
| **Status** | Done |
| **Steps** | 1,084 |
| **Raw BPB** | 1.3230 |
| **Quant BPB** | 1.3489 |
| **Artifact** | 15.8 MB |

**Details**: Fixed momentum=0.95 instead of cycling 0.85-0.95.

**Observations**: Marginal degradation. Cyclic momentum is slightly helpful — oscillation may act as implicit regularization.

---

### Exp05 — Grad Accum 4 (More Steps)

| Field | Value |
|-------|-------|
| **Folder** | `exp05_grad-accum4_from-exp00/` |
| **Based on** | exp00 |
| **Status** | Done |
| **Steps** | 1,206 |
| **Raw BPB** | 1.3001 |
| **Quant BPB** | 1.3181 |
| **Artifact** | 15.8 MB |

**Details**: Reduced grad_accum 8→4, doubling step count within wallclock. Effective batch halved from 524K→262K tokens.

**Observations**: **Major breakthrough** — first sub-1.32 quant bpb. +122 more steps than baseline.

---

### Exp06 — SWA + AWQ + Accum 2

| Field | Value |
|-------|-------|
| **Folder** | `exp06_swa-awq-accum2_from-exp05/` |
| **Based on** | exp05 |
| **Status** | Done |
| **Steps** | 1,219 |
| **Raw BPB** | 1.2982 |
| **Quant BPB** | 1.3161 |
| **Artifact** | 15.7 MB |

**Details**: Pushed accum 4→2. SWA_EVERY=100, AWQ_ALPHA=0.6, WARMUP_STEPS=80.

**Observations**: Continued improvement. Raw bpb breaks below 1.30 for first time.

---

### Exp07 — Tighter SWA + AWQ

| Field | Value |
|-------|-------|
| **Folder** | `exp07_tighter-swa-awq_from-exp06/` |
| **Based on** | exp06 |
| **Status** | Done |
| **Steps** | 1,220 |
| **Raw BPB** | 1.2978 |
| **Quant BPB** | 1.3164 |
| **Artifact** | 15.5 MB |

**Details**: SWA_EVERY=150, AWQ_ALPHA=0.7.

**Observations**: Reduced artifact ~200KB vs exp06. Quant bpb nearly identical. Sweet spot is SWA_EVERY=100.

---

### Exp08 — Context Frequency Bias

| Field | Value |
|-------|-------|
| **Folder** | `exp08_ctx-freq-bias_from-exp05/` |
| **Based on** | exp05 |
| **Status** | Done |
| **Steps** | 1,196 |
| **Raw BPB** | 1.3014 |
| **Quant BPB** | 1.3225 |
| **Artifact** | 15.0 MB |

**Details**: Learned scalar bias `ctx_freq_bias * log(1 + count_in_context)` on logits. Exploits 77.7% token burstiness. +1 parameter.

**Observations**: Small improvement over exp05 but redundant with what attention learns. Smallest artifact at 15.0MB.

---

### Exp09 — Pad Ignore + Word-Start Boost ⭐ BEST

| Field | Value |
|-------|-------|
| **Folder** | `exp09_padignore-wordboost_from-exp06/` |
| **Based on** | exp06 |
| **Status** | Done |
| **Steps** | 1,203 |
| **Raw BPB** | 1.2974 |
| **Quant BPB** | **1.3145** |
| **Artifact** | **14.5 MB** |

**Details**: (1) `ignore_index=0` — skip pad tokens that waste gradient; (2) learned `word_start_boost` scalar that scales bigram at word boundaries (`▁` tokens). +1 parameter.

**Observations**: **Best result.** Pad-ignore removes ~5-10% wasted compute. Word-start boost helps bigram focus on hardest prediction (word-initial at 5.1 bpb vs 2.9 bpb for repeats).

---

### Exp10 — Trigram + Unigram Bias

| Field | Value |
|-------|-------|
| **Folder** | `exp10_trigram-unigram_from-exp09/` |
| **Based on** | exp09 |
| **Status** | Done |
| **Steps** | 1,199 |
| **Raw BPB** | 1.2956 |
| **Quant BPB** | 1.3151 |
| **Artifact** | 15.6 MB |

**Details**: Trigram hash table (10240×128) + learned `unigram_bias * log(freq)`. ~1.3M extra params.

**Observations**: **Best raw bpb** (1.2956) but quant bpb slightly worse than exp09. Extra params compress less efficiently.

---

### Exp11 — Trigram Slim (dim=48) + AWQ 0.7

| Field | Value |
|-------|-------|
| **Folder** | `exp11_trigram-slim-awq07_from-exp10/` |
| **Based on** | exp10 |
| **Status** | Done |
| **Steps** | 1,198 |
| **Raw BPB** | 1.2994 |
| **Quant BPB** | 1.3259 |
| **Artifact** | 14.6 MB |

**Details**: Trigram embed dim 128→48 with `tri_proj` (48→512). AWQ alpha 0.6→0.7.

**Observations**: Smaller artifact but significantly worse quant bpb. dim=48 too small, AWQ 0.7 too aggressive. Double regression.

---

### Exp12 — Trigram 64-dim + AWQ 0.6

| Field | Value |
|-------|-------|
| **Folder** | `exp12_trigram64-awq06_from-exp10/` |
| **Based on** | exp10 |
| **Status** | Done |
| **Steps** | 1,195 |
| **Raw BPB** | 1.2969 |
| **Quant BPB** | 1.3222 |
| **Artifact** | 15.1 MB |

**Details**: Middle-ground trigram dim=64, AWQ alpha=0.6.

**Observations**: Better than exp11 but worse than exp09. Hash collisions in 10240-entry table too frequent for trigrams (1024³ combinations).

---

### Exp13 — Multi-Head Gated Bigram ⭐ TIED BEST

| Field | Value |
|-------|-------|
| **Folder** | `exp13_multihead-gate-bigram_from-exp09/` |
| **Based on** | exp09 |
| **Status** | Done |
| **Steps** | 1,201 |
| **Raw BPB** | 1.2974 |
| **Quant BPB** | **1.3145** |
| **Artifact** | **14.5 MB** |

**Details**: K=2 independent hash functions (averaged, reduces collisions). Context gate: `sigmoid(gate_proj(tok_emb) + gate_bias)`. +513 extra params.

**Observations**: **Tied with exp09 for best quant bpb**. Multi-head reduces collision 10%→1%, but impact neutralized. Base for exp14-26.

---

### Exp14 — Engram Multi-Order (1-5gram)

| Field | Value |
|-------|-------|
| **Folder** | `exp14_engram-multiorder_from-exp13/` |
| **Based on** | exp13 |
| **Status** | Done |
| **Steps** | 1,196 |
| **Raw BPB** | 1.3056 |
| **Quant BPB** | 1.3338 |
| **Artifact** | 15.0 MB |

**Details**: 1-5gram with 2 hash heads each = 10 lookups/position. Shared 10240×128 table. 0 new params.

**Observations**: **Significant regression**. Shared embeddings across n-gram orders = destructive interference.

---

### Exp15 — Engram 3-Order (Orthogonal Subspaces)

| Field | Value |
|-------|-------|
| **Folder** | `exp15_engram-3order_from-exp14/` |
| **Based on** | exp14 |
| **Status** | Done |
| **Steps** | 1,196 |
| **Raw BPB** | 1.2995 |
| **Quant BPB** | 1.3260 |
| **Artifact** | 14.6 MB |

**Details**: 1-3gram × 2 heads = 6 lookups. Orthogonal subspace: unigram [0:42], bigram [42:84], trigram [84:128].

**Observations**: Better than exp14 but worse than exp09. Each subspace too small (~42 dims).

---

### Exp16 — JEPA Auxiliary Loss

| Field | Value |
|-------|-------|
| **Folder** | `exp16_jepa-aux_from-exp15/` |
| **Based on** | exp15 |
| **Status** | Done |
| **Steps** | 1,146 |
| **Raw BPB** | 1.3194 |
| **Quant BPB** | 1.3526 |
| **Artifact** | 14.5 MB |

**Details**: Predictor MLP (512→128) predicts next position's engram embedding. MSE loss λ=0.1. ~65K extra training params.

**Observations**: **Major regression**. Not true JEPA — uses fixed hash targets. MSE against hash embeddings provides adversarial gradient. ~524ms/step (slower).

---

### Exp17 — Byte-Level Engram

| Field | Value |
|-------|-------|
| **Folder** | `exp17_byte-engram_from-exp16/` |
| **Based on** | exp16 |
| **Status** | Done |
| **Steps** | 1,146 |
| **Raw BPB** | 1.3201 |
| **Quant BPB** | 1.3527 |
| **Artifact** | 14.6 MB |

**Details**: ByteBoundaryEmbedding — cross-token byte bigram/trigram from 4-byte window. ~49K extra params.

**Observations**: No improvement over exp16. Base (exp15+JEPA) too weak.

---

### Exp18 — Separate Trigram Table (64-dim)

| Field | Value |
|-------|-------|
| **Folder** | `exp18_separate-trigram64_from-exp13/` |
| **Based on** | exp13 |
| **Status** | Done |
| **Steps** | 1,194 |
| **Raw BPB** | 1.2995 |
| **Quant BPB** | 1.3247 |
| **Artifact** | 15.0 MB |

**Details**: Separate 64-dim trigram table (10240×64) + 2-head hashing + projection, on top of exp13's 128-dim bigram. ~688K extra params.

**Observations**: Marginal raw improvement but worse quant bpb (1.3247 vs 1.3145). Extra params don't compress well.

## 2.4 Evolution Tree

```
exp27 (reference)
 ├── exp00 (baseline rerun) ──────────────────────── quant 1.3389
 │    ├── exp01b (LN scale ablation)                  not run
 │    ├── exp01c (EMA ablation)                        not run
 │    ├── exp01d (XSA ablation)                        quant 1.3568  ✗ worse
 │    ├── exp02 (speed+FP16 bigram)                    quant 1.3429  ✗ over 16MB
 │    ├── exp03 (QAT-STE)                              quant 1.3684  ✗ worst
 │    ├── exp04 (no cyclic momentum)                   quant 1.3489  ✗ worse
 │    ├── exp05 (grad_accum=4) ────────────────────── quant 1.3181  ✓ breakthrough
 │    │    ├── exp08 (context freq bias)                quant 1.3225
 │    │    └── exp06 (SWA+AWQ+accum=2) ───────────── quant 1.3161  ✓ improved
 │    │         ├── exp07 (tighter SWA/AWQ)            quant 1.3164
 │    │         └── exp09 (pad ignore+word boost) ──── quant 1.3145  ⭐ BEST
 │    │              ├── exp10 (trigram+unigram) ───── quant 1.3151
 │    │              │    ├── exp11 (trigram slim)      quant 1.3259  ✗
 │    │              │    └── exp12 (trigram 64d)       quant 1.3222
 │    │              └── exp13 (multihead gate bigram)  quant 1.3145  ⭐ TIED BEST
 │    │                   ├── exp14 (engram 1-5gram)    quant 1.3338  ✗
 │    │                   │    └── exp15 (engram 3order) quant 1.3260
 │    │                   │         └── exp16 (JEPA aux) quant 1.3526  ✗✗
 │    │                   │              └── exp17 (byte engram) quant 1.3527  ✗✗
 │    │                   ├── exp18 (separate trigram)  quant 1.3247
 │    │                   └── exp19-26 (phase 3b) ──── in progress
```

## 2.5 Lessons Learned (Phase 3a)

1. **Step count is king**: Reducing grad_accum (2× more optimizer updates) > any architectural change.
2. **Don't fight the quantizer**: QAT-STE and FP16 bigram tried to address quantization directly — both failed. Better convergence naturally produces better-quantizing weights.
3. **N-gram tables have diminishing returns**: Bigram valuable (+0.02 bpb). Trigram marginal. Higher-order actively hurt.
4. **Hash collision reduction matters less than expected**: Multi-head hashing (exp13) reduces collisions 10%→1%, but quant bpb unchanged from exp09.
5. **Auxiliary losses are dangerous**: JEPA (exp16) caused biggest single regression.
6. **Simple targeted fixes beat complex architectures**: Pad-ignore (0 params) + word-start boost (1 param) > 688K trigram params.
7. **Quant gap is the real metric**: exp10 had best raw bpb but exp09 had best quant bpb.

## 2.6 TL;DR (Phase 3a)

**What worked**:
- Cutting `grad_accum` 8→4→2 = **2× more steps** = biggest single win (exp05→06)
- `ignore_index=0` to skip pad tokens in loss = free improvement (exp09)
- `word_start_boost` scalar for bigram at `▁` boundaries = +1 param, measurable gain (exp09)
- SWA every 100 steps + AWQ alpha=0.6 = good quant compression without hurting quality

**What didn't work**:
- QAT-STE, FP16 bigram, XSA, fixed momentum (exp01d–04) = all worse than baseline
- Trigram/n-gram tables (exp10–12, 14–15, 18) = raw bpb improves but quant bpb regresses
- JEPA auxiliary loss with fixed hash targets (exp16–17) = worst regression of all

**One-liner**: *More optimizer steps + smarter loss masking > fancy architecture, every time.*

---

# 3. Phase 3b-Part1: Systematic Ablations (exp27b–exp33b)

> **Hardware**: 1×H100, 600s wallclock
> **Base**: exp09 (chosen over exp13 — multi-head bigram added complexity without improving quant bpb)
> **Best result**: exp30b — quant bpb **1.3156**, artifact 15.05 MB

## 3.1 Leaderboard

| Rank | Exp | Name | Base | Quant BPB | Raw BPB | Size | Under 16MB? |
|:----:|:---:|------|:----:|:---------:|:-------:|:----:|:-----------:|
| **1** | **30b** | combo (resid-norm + loss-wt + type-emb) | exp09 | **1.3156** | 1.2983 | 15.05 MB | **Yes** |
| 2 | 33b | alternating RoPE + NTK | exp30b | 1.3145 | 1.2971 | 14.94 MB | Yes |
| 3 | 29b | loss-weight + token-type-emb | exp09 | 1.3176 | 1.3005 | 15.75 MB | Yes |
| 4 | 27b | resid-norm | exp09 | 1.3197 | ~1.300 | ~15.3 MB | Yes |
| 5 | 31b | RoPE base 50k | exp30b | 1.3206 | 1.2953 | 15.01 MB | Yes |
| 6 | 09 | padignore-wordboost (baseline) | exp06 | 1.3282 | 1.2974 | 14.5 MB | Yes |
| 7 | 32b | aux word-boundary loss | exp30b | 1.3424 | 1.3153 | 15.68 MB | Yes |
| — | 28b | perlayer quant | exp09 | N/A (analysis only) | — | 16.24 MB | **No** |

## 3.2 Detailed Experiment Cards

### Exp27b — Residual Stream Normalization (from exp09)

| Field | Value |
|-------|-------|
| **Folder** | `exp27b_resid-norm_from-exp09/` |
| **Based on** | exp09 |
| **Steps** | ~1200 |
| **Raw BPB** | ~1.300 |
| **Quant BPB** | **1.3197** |
| **Artifact** | ~15.3 MB |
| **Extra params** | 0 |

**Change**: Parameterless `F.rms_norm` after each decoder skip-connection.

**Observations**: Residual norm growth (19.7→89.5) was root cause of poor quantization in later layers. RMSNorm keeps norms bounded → flatter weight distributions → lower quant error.

**Verdict**: ✅ Validated.

---

### Exp28b — Per-Layer Quantization Bitwidth (from exp09)

| Field | Value |
|-------|-------|
| **Folder** | `exp28b_perlayer-quant_from-exp09/` |
| **Based on** | exp09 |
| **Status** | Analysis only |
| **MSE improvement** | -16.13% |
| **Artifact** | 16.24 MB |

**Change**: Higher bitwidths for boundary layers (0,1,8,9): int7 attn, int6 MLP.

**Observations**: Cuts boundary quant error in half but blows 16MB budget.

**Verdict**: ❌ Not viable. Size kills it.

---

### Exp29b — Loss Weighting + Token-Type Embedding (from exp09)

| Field | Value |
|-------|-------|
| **Folder** | `exp29b_lossweight-typemb_from-exp09/` |
| **Based on** | exp09 |
| **Steps** | 1207 |
| **Raw BPB** | 1.3005 |
| **Quant BPB** | **1.3176** |
| **Artifact** | 15.75 MB |
| **Extra params** | +8,305 |

**Change**: (1) Per-token loss weighting: 1.5x word-start, 0.8x easy suffixes. (2) 7-category token-type embedding: 7×16 + 16×512 proj + learned scale. Zero-initialized.

**Observations**: Strong win. Loss weighting redistributes gradient to high-opportunity tokens. Token-type gives explicit structural signal.

**Verdict**: ✅ Validated.

---

### Exp30b — Combo: Resid-Norm + Loss-Weight + Token-Type (from exp09) ⭐ BEST

| Field | Value |
|-------|-------|
| **Folder** | `exp30b_combo_from-exp09/` |
| **Based on** | exp09 |
| **Steps** | 1200 |
| **Raw BPB** | 1.2983 |
| **Quant BPB** | **1.3156** |
| **Artifact** | **15.05 MB** |

**Change**: All three validated improvements combined, each togglable via env vars.

**Observations**: Best quant bpb. Gains sub-additive (expected -0.019, got -0.0126). Quant gap reduced 0.023→0.017.

**Verdict**: ✅ **Phase 3b-Part1 SOTA. Base for all subsequent experiments.**

---

### Exp31b — RoPE Base 50k (from exp30b)

| Field | Value |
|-------|-------|
| **Folder** | `exp31b_rope-headspec_from-exp30b/` |
| **Based on** | exp30b |
| **Steps** | 1197 |
| **Raw BPB** | **1.2953** |
| **Quant BPB** | 1.3206 |
| **Artifact** | 15.01 MB |

**Change**: RoPE base 10000→50000.

**Observations**: Best raw bpb ever but quant gap widens to 0.0253 vs 0.0173 for exp30b. Net NEGATIVE after quantization.

**Verdict**: ❌ Raw gain eaten by quant degradation.

---

### Exp32b — Auxiliary Word-Boundary Loss (from exp30b)

| Field | Value |
|-------|-------|
| **Folder** | `exp32b_aux-boundary_from-exp30b/` |
| **Based on** | exp30b |
| **Steps** | 1196 |
| **Raw BPB** | 1.3153 |
| **Quant BPB** | 1.3424 |
| **Artifact** | 15.68 MB |

**Change**: Auxiliary head (512→1, sigmoid, binary CE, λ=0.15) predicting word-start.

**Observations**: Significant regression (+0.0268). Aux loss diverts gradient. Token-type already provides structural signal.

**Verdict**: ❌ Auxiliary losses counterproductive.

---

### Exp33b — Alternating RoPE Bases + NTK Scaling (from exp30b)

| Field | Value |
|-------|-------|
| **Folder** | `exp33b_swa-attn-ntkrope_from-exp30b/` |
| **Based on** | exp30b |
| **Steps** | 1200 |
| **Raw BPB** | 1.2971 |
| **Quant BPB** | 1.3145 |
| **Artifact** | 14.94 MB |

**Change**: Even blocks rope_base=50000, odd blocks rope_base=1000. NTK scaling.

**Observations**: Marginal improvement (-0.0011). Positional loss curve STILL flat after 256 tokens. Improvement is likely noise.

**Verdict**: ⚠️ Marginal. exp30b preferred for simplicity.

## 3.3 Deep Analysis Insights (from exp33b checkpoint)

- **Word-start tokens**: 27.7% of tokens but **44.2% of total loss**. Mean loss 3.72 vs 1.28 for "other".
- **Softcap not saturating**: Max |logit| = 28.2 (94% of cap=30).
- **Word-start boost learned DOWN to 0.16**: Model suppresses bigram at word boundaries — hash collisions may be confusing.
- **Token-type embedding actively used**: Scale grew 0.05→0.53. Punctuation/whitespace highest norms.
- **Top-1 accuracy: 46.6%, Top-5: 68.9%**
- **First 16 tokens**: Loss 2.96 vs 2.27 for rest (+0.70 penalty). Cold-start tokens disproportionately hard.
- **Positional loss flat after 256**: 2.53→2.33→flat ~2.2-2.4.

## 3.4 Evolution Tree

```
exp09 (pad ignore + word-start boost) ──── quant 1.3282
 ├── exp27b (resid-norm) ──────────────── quant 1.3197  ✓
 ├── exp28b (perlayer quant) ──────────── N/A (over 16MB) ✗
 ├── exp29b (loss-weight + type-emb) ──── quant 1.3176  ✓
 └── exp30b (combo: 27b+29b) ─────────── quant 1.3156  ⭐ BEST
      ├── exp31b (RoPE 50k) ──────────── quant 1.3206  ✗ (quant gap)
      ├── exp32b (aux boundary loss) ──── quant 1.3424  ✗✗ (gradient waste)
      └── exp33b (alternating RoPE+NTK) ─ quant 1.3145  ⚠️ (marginal, noisy)
```

## 3.5 Lessons Learned (Phase 3b-Part1)

8. **Residual norm control is high leverage**: RMSNorm after skip connections attacks root cause (norm growth) not symptom.
9. **Stacking orthogonal improvements works**: exp30b combined 3 independent improvements for sub-additive but substantial gain.
10. **Auxiliary losses fatal in compute-starved regimes**: ~1200 steps = every gradient must reduce CE.
11. **RoPE base changes hurt quantization**: Higher base → harder-to-compress weight distributions.
12. **Long context is a dead end**: Positional loss flat after 256 tokens regardless of RoPE.
13. **Size budget matters**: Per-layer quant delivers 16% MSE reduction but can't fit in 16MB.

---

# 4. Phase 3b-Part2: LR Fix Era (exp34b–exp48b)

> **Hardware**: 1×H100, 600s wallclock
> **Base**: exp30b with LR schedule fix
> **Critical discovery**: ITERATIONS=20000 meant warmdown NEVER fired. Fixing to 1300 = -0.0166 bpb.
> **Best result**: exp48b — quant bpb **1.2930**

All experiments include the **LR schedule fix** (ITERATIONS=1300, WARMDOWN_ITERS=400).

## 4.1 Leaderboard

| Rank | Exp | Name | Base | Quant BPB | Raw BPB | Size | Steps | Verdict |
|:----:|:---:|------|:----:|:---------:|:-------:|:----:|:-----:|:-------:|
| **1** | **48b** | 10blocks-depth | 42b | **1.2930** | 1.2824 | 15.22 MB | 1198 | ✅ Best |
| 2 | 45b | awq-alpha07 | 42b | 1.2897* | — | 14.01 MB | — | ✅ Post-train only |
| 3 | 42b | revive-block9 | 34b | 1.2969 | 1.2867 | 14.01 MB | 1196 | ✅ Layer sharing |
| 4 | 39b | swa-tuning | 34b | 1.2969 | 1.2867 | — | ~1196 | ≈ Tie with 42b |
| 5 | 34b | lr-schedule-fix | 30b | 1.2990 | 1.2891 | 15.13 MB | 1186 | ✅ LR fix breakthrough |
| 6 | 46b | full-mha | 42b | 1.2979 | 1.2896 | 15.59 MB | 1164 | ❌ 8KV not worth it |
| 7 | 43b | boundary-boost | 42b | 1.2993 | — | 14.01 MB | ~1196 | ❌ Too sparse |
| 8 | 47b | warmdown200 | 42b | 1.3081 | 1.2983 | 14.07 MB | 1200 | ❌ Too late warmdown |
| 9 | 37b | fused-cap | 34b | 1.3159 | 1.3014 | 15.17 MB | 1197 | ❌ Cap hurts |
| 10 | 35b | focal-loss | 30b | 1.3201 | — | 15.20 MB | 1196 | ❌ γ=2 too aggressive |
| 11 | 36b | cappedact-labelsmooth | 30b | 1.3248 | — | 14.50 MB | ~1180 | ❌ Both hurt |
| 12 | 44b | seqlen-curriculum | 42b | ~1.32 | — | — | ~1000 | ❌ Failed |
| 13 | 38b | speed-opt | 34b | — | — | — | — | ❌ Failed |

*exp45b is post-training AWQ alpha tuning, not a retrain.

## 4.2 Key Discoveries

14. **LR schedule was completely broken** (exp34b): ITERATIONS=20000 with 600s wallclock meant warmdown NEVER fired. Fixing to ITERATIONS=1300 gave -0.0166 quant bpb — single biggest improvement.
15. **Layer sharing revives dead blocks** (exp42b): Block 9 dead at 6.1% effective rank. Sharing block 3 at position 9 revived it to 10.3%.
16. **More depth beats more width** (exp48b vs exp46b): 10th unique block (-0.0039 bpb) > 8 KV heads (+0.001 bpb).
17. **Auxiliary losses still fatal** (exp35b, exp43b): Focal loss (γ=2), boundary boost — both hurt.
18. **Activation capping hurts training** (exp36b, exp37b): Warmdown already produces smooth-enough weights.
19. **AWQ alpha is undertested** (exp45b): Sweeping alpha 0.6→0.7 gave -0.007 bpb for free.
20. **Warmdown=400 is optimal** (exp47b): warmdown=200 too late — only 2 SWA checkpoints.
21. **Calibration improved dramatically**: With warmdown, near-perfect calibration (all bins within ±0.003 gap).

## 4.3 Deep Analysis Insights (from exp48b checkpoint)

- **Word-start tokens**: 25.5% of tokens but **42.5% of total loss**. Mean loss 3.61 vs 1.20. Top-1 accuracy only 24.7%.
- **Top confusion pairs**: `▁and`→`,` (299), `▁the`→`▁a` (263), `▁and`→`.` (200). Function word disambiguation is the core problem.
- **Positional loss flat after 256**: 2.47→2.06→flat ~2.1-2.2.
- **MLP activation outliers**: Block 1 max=1314, 10-15% of activations >4.0. Root cause of 6.86% MLP quant error.
- **Calibration near-perfect**: All bins within ±0.003.
- **Block 3 (shared) under stress**: Effective rank 57.9% — lowest of all blocks.
- **728 KB headroom**: Tighter than exp42b's 2.09 MB.
- **Sentence boundary gap**: 0.78 bpb (after-boundary 2.93 vs normal 2.15).

---

# 5. Phase 3b-Part3: Simplification + XSA (exp53b–clean_54b)

> **Hardware**: 1×H100, 600s wallclock
> **Key insight**: Removing features improved results
> **Best result**: exp54b — quant bpb **1.2708** (1×H100 SOTA)

## 5.1 Experiment Cards

### exp53b — Lean Combo (from exp48b)

| Field | Value |
|-------|-------|
| **Quant BPB** | 1.2720 |
| **Raw BPB** | 1.2640 |
| **Steps** | 1218 |
| **ms/step** | 492 |
| **Changes** | Stripped token-type + loss weighting, kept resid-norm. max-autotune-no-cudagraphs. VAL_LOSS_EVERY=800. WARMUP=20. |

---

### exp54b — XSA + zstd + c_k Fix (from exp53b) ⭐ 1×H100 BEST

| Field | Value |
|-------|-------|
| **Quant BPB** | **1.2708** |
| **Raw BPB** | 1.2642 |
| **Steps** | 1235 |
| **ms/step** | 486 |
| **Changes** | XSA on last 2 decoder layers. Fixed block 0 c_k outlier (fp16 keep). Reverted to zstd. |

---

### exp55b — Scaled XSA All Layers (from exp54b)

| Field | Value |
|-------|-------|
| **Quant BPB** | 1.2717 (marginal regression) |
| **Raw BPB** | 1.2648 |
| **Steps** | 1183 |
| **ms/step** | 507 |
| **Changes** | Learned `xsa_alpha = sigmoid(param)` per layer on ALL 10 blocks. |

**Finding**: Model learned alpha 0.75-0.99 on ALL layers — XSA universally wanted. But 507ms vs 486ms = 52 fewer steps, erasing benefit on 1×H100.

---

### exp56b — Fast XSA (Cosine Scale) (from exp55b)

| Field | Value |
|-------|-------|
| **Status** | Killed early — no speed improvement |
| **ms/step** | 508 (same as exp55b) |

**Finding**: GQA head expansion (`repeat_interleave` 4→8) is the bottleneck, not the XSA math.

---

### Community Model B — Fair 1×H100 Comparison

| Field | Value |
|-------|-------|
| **Quant BPB** | 1.2825 |
| **Raw BPB** | 1.2501 |
| **Steps** | 1479 |
| **ms/step** | 406 |
| **Key difference** | Faster per-step (partial RoPE, no resid-norm) but 5× worse quant gap (0.032 vs 0.007). |

---

### exp58b vs exp59b vs exp54b — Resid-Norm A/B Test

| Config | Steps | ms/step | Quant val_bpb |
|--------|-------|---------|:-------------:|
| exp54b (no norm) | 1235 | 486 | **1.2708** |
| exp58b (post-addition norm) | 1216 | 493 | 1.2741 |
| exp59b (pre-skip norm) | ~1216 | ~493 | ~1.274 |

**Conclusion**: With warmdown active, resid-norm is redundant. The 7ms/step overhead costs more than the quant gap benefit.

---

### clean_54b — Final Architecture (from exp54b)

| Field | Value |
|-------|-------|
| **Quant BPB** | 1.2723 |
| **Changes** | exp54b + vanilla TTT + named checkpoint save + logging fixes. No architectural changes. |

---

### Failed Experiments (Phase 3b-Part3)

| Exp | What | Why it failed |
|-----|------|--------------|
| 35b | Focal loss (gamma=2) | Too aggressive — suppressed easy token gradients |
| 36b | Capped act + label smooth | Both hurt independently and together |
| 37b | Fused cap (no label smooth) | Cap=4.0 hurt raw quality more than helped quant |
| 43b | Boundary loss boost | Too sparse (2.5% of positions) |
| 44b | Seq-len curriculum | Speed regression |
| 46b | Full MHA (8 KV heads) | Extra params but slower, no bpb improvement |
| 55b | Scaled XSA all layers | 20ms/step overhead costs 52 steps |
| 56b | Fast cosine XSA | GQA head expansion is the bottleneck |
| 58b | Resid-norm re-enabled | 7ms/step → 19 fewer steps, redundant with warmdown |
| 59b | Pre-norm skip | Same overhead, no quality difference |

## 5.2 Lessons Learned (Phase 3b-Part3)

1. **LR warmdown is critical** — Biggest single improvement (0.017 bpb).
2. **Simpler is better** — Stripping token-type and loss weighting HELPED.
3. **Steps > features** — Every ms/step matters. Features that add compute must justify their cost.
4. **Resid-norm is redundant with warmdown** — Weights already smooth.
5. **XSA last 2 is the sweet spot** — Model wants XSA everywhere but overhead makes all-layer too expensive on 1×H100.
6. **zstd > LZMA** — Better for structured quantized weights (15.19 MB vs 15.37 MB).
7. **torch.compile mode matters** — `max-autotune-no-cudagraphs` gives kernel autotuning without tensor overwrite issues.

## 5.3 TTT (Test-Time Training) Note

TTT requires matching the torch.compile context used during training for correct inference results.

## 5.4 Complete Phase 3b Lineage (exp27b → clean_54b)

```
exp09 (pad-ignore + word-start boost, quant 1.3282) ← PHASE 3a BEST
 │
 ├── exp27b [✅ POSITIVE] resid-norm (quant 1.3197, Δ-0.0085)
 ├── exp28b [❌ NEGATIVE] perlayer quant (over 16MB budget)
 ├── exp29b [✅ POSITIVE] loss-weight + token-type (quant 1.3176, Δ-0.0106)
 │
 └── exp30b [✅ POSITIVE] combo: resid-norm + loss-weight + token-type (quant 1.3156, Δ-0.0126)
      │
      ├── exp31b [❌ NEGATIVE] RoPE 50k (better raw 1.2953 but worse quant 1.3206)
      ├── exp32b [❌ NEGATIVE] aux boundary loss (quant 1.3424)
      ├── exp33b [⚪ NEUTRAL] alternating RoPE + NTK (quant 1.3145, marginal)
      │
      └── exp34b [✅✅ MAJOR] LR schedule fix (quant 1.2990, Δ-0.0166 — biggest single win)
           │
           ├── exp35b [❌ NEGATIVE] focal loss γ=2 (quant 1.3201)
           ├── exp36b [❌ NEGATIVE] capped act + label smooth (quant 1.3472)
           ├── exp37b [❌ NEGATIVE] fused cap only (quant 1.3159)
           ├── exp38b [⚪ NEUTRAL] speed opt (quant 1.3002)
           ├── exp39b [✅ POSITIVE] SWA tuning (quant 1.2985)
           │
           └── exp42b [✅ POSITIVE] layer sharing block 3→pos 9 (quant 1.2969)
                │
                ├── exp43b [⚪ NEUTRAL] boundary loss boost (quant 1.3003)
                ├── exp44b [❌ NEGATIVE] seq-len curriculum (failed)
                ├── exp45b [⚪ NEUTRAL] AWQ alpha=0.7 (quant 1.3033)
                ├── exp46b [⚪ NEUTRAL] full MHA 8 KV heads (quant 1.2979)
                ├── exp47b [❌ NEGATIVE] warmdown=200 (quant 1.3081)
                │
                └── exp48b [✅ POSITIVE] 10th unique block (quant 1.2930)
                     │
                     ├── exp49b [⚪ NOT RUN] diffusion GPT
                     ├── exp50b [⚪ NOT RUN] byte-level JEPA
                     │
                     └── exp53b [✅ POSITIVE] strip overhead (quant 1.2720, Δ-0.0210)
                          │
                          └── exp54b [✅ POSITIVE] XSA last 2 + c_k fix (quant 1.2708) ← 1xH100 BEST
                               │
                               ├── exp55b [⚪ NEUTRAL] scaled XSA all layers (quant 1.2717)
                               ├── exp56b [❌ NEGATIVE] fast cosine XSA (no speed gain)
                               ├── exp57b [❌ NEGATIVE] LoRA TTT (failed)
                               ├── exp58b [❌ NEGATIVE] resid-norm ON (quant 1.2741)
                               ├── exp59b [❌ NEGATIVE] pre-norm skip (quant 1.2743)
                               │
                               └── clean_54b [✅ POSITIVE] named save + TTT (quant 1.2723)
                                    └── clean_54b_v2 [❌ NEGATIVE] bf16 roundtrip (destroyed quality)

Community B model (fair 1xH100): quant 1.2825 ← WE BEAT BY 0.012 bpb
```

## 5.5 Complete Results Table (Phase 3b, sorted by quant bpb)

| Rank | Exp | Quant BPB | Raw BPB | Steps | ms/step | Tag | Key Change |
|:----:|:---:|:---------:|:-------:|:-----:|:-------:|:---:|------------|
| **1** | **exp54b** | **1.2708** | 1.264 | 1235 | 486 | ✅ | XSA last 2 + c_k fix |
| 2 | exp53b | 1.2720 | 1.264 | 1218 | 492 | ✅ | Strip overhead |
| 3 | clean_54b | 1.2723 | 1.264 | 1205 | 498 | ✅ | Named save |
| 4 | community | 1.2825 | 1.250 | 1479 | 406 | — | Their full arch |
| 5 | exp48b | 1.2930 | 1.282 | 1198 | 501 | ✅ | 10th unique block |
| 6 | exp42b | 1.2969 | 1.287 | 1201 | 502 | ✅ | Layer sharing |
| 7 | exp39b | 1.2985 | — | 1196 | 502 | ✅ | SWA tuning |
| 8 | exp34b | 1.2990 | 1.289 | 1186 | 506 | ✅✅ | LR schedule fix |
| 9 | exp38b | 1.3002 | 1.290 | 1196 | 502 | ⚪ | Speed opt |
| 10 | exp43b | 1.3003 | 1.290 | 1198 | 501 | ⚪ | Boundary boost |
| 11 | exp45b | 1.3033 | — | 1196 | 502 | ⚪ | AWQ α=0.7 |
| 12 | exp47b | 1.3081 | 1.298 | 1200 | 500 | ❌ | Warmdown=200 |
| 13 | exp33b | 1.3145 | — | — | — | ⚪ | Alt RoPE + NTK |
| 14 | exp30b | 1.3156 | 1.298 | 1200 | 500 | ✅ | Combo |
| 15 | exp37b | 1.3159 | 1.301 | 1197 | 501 | ❌ | Fused cap |
| 16 | exp29b | 1.3176 | 1.301 | 1202 | 499 | ✅ | Loss-wt + type-emb |
| 17 | exp27b | 1.3197 | ~1.300 | ~1200 | ~500 | ✅ | Resid-norm |
| 18 | exp35b | 1.3201 | — | 1196 | — | ❌ | Focal loss γ=2 |
| 19 | exp31b | 1.3206 | 1.295 | 1197 | 502 | ❌ | RoPE 50k |
| 20 | exp32b | 1.3424 | 1.315 | 1196 | 502 | ❌ | Aux boundary loss |
| 21 | exp36b | 1.3472 | 1.332 | 1135 | 529 | ❌ | Cap + label smooth |

## 5.6 Remaining Opportunities (identified at this stage)

### Zero-cost (no retraining, apply to exp54b checkpoint)
1. AWQ alpha sweep on exp54b (test alpha=0.3-0.8)
2. Pruning threshold sweep (0%, 1%, 2%, 5%)

### Quick retrains (10 min each)
3. Seed sweep (43, 44, 45). Variance: 0.003-0.005 bpb.
4. Weight decay tuning (0.04→0.06)
5. LR tuning (MATRIX_LR 0.025→0.030)
6. EMA with decay=0.995 (replace SWA)

### Creative submissions
7. Diffusion GPT (exp49b) — Hybrid masked diffusion + AR
8. Byte-level JEPA (exp50b) — Raw byte model

### For 8×H100 scaling
9. XSA on all layers (alpha=0.75-0.99 everywhere)
10. Partial RoPE (16 dims) — community speed trick
11. Late QAT — quant noise in last 15%
12. Predicted quant val_bpb ~1.12-1.14

---

# 6. Phase 3.5: 8×H100 Simulation (exp60–exp80)

> **Hardware**: 1×H100 simulating 8×H100 (6000s wallclock, grad_accum=8, 786K tokens/batch)
> **Base**: exp54b (clean baseline)
> **Best result**: exp74 — sliding bpb **1.1456**, artifact 15.86 MB

## 6.1 Leaderboard

| Exp | Description | Pre-quant BPB | Post-quant BPB | Sliding BPB | Artifact | Steps | Status |
|-----|-------------|:-------------:|:--------------:|:-----------:|:--------:|:-----:|:------:|
| **exp74** | pRoPE+qgain+wbigram+LLR | 1.1539 | 1.1685 | **1.1456** | 15.86 MB | 6169 | **Best** |
| exp70 | Speed-optimized from exp69 | ~1.14 | ~1.17 | ~1.15 | ~16 MB | ~7500 | Baseline |
| exp78 | WS loss curriculum | — | — | — | — | — | Better embeddings |
| exp75 | Word pool from exp74 | — | — | — | — | — | Failed (scale→0.002) |
| exp61b | XSA all + warmdown | 1.1504 | 1.1781 | — | ~16.5 MB | ~7000 | Over budget |
| exp63 | Cascade VR + adaptive WD | 1.1377 | 1.1730 | — | 16.45 MB | ~7000 | Over budget |

## 6.2 Evolution Tree

```
exp54b (clean baseline, 1.2708 bpb)
  └── exp60 (EMA, flash_attn3, 8×H100 sim)
        └── exp61b (XSA all blocks, cosine warmdown → 1.1504 pre-quant)
              └── exp63 (cascading V-residual, adaptive warmdown → 1.1377 pre-quant)
                    ├── exp64 (MLP int6 quant — never ran)
                    ├── exp65 (quant overhaul — never ran)
                    │     └── exp66 (MiLe loss + NoPE — failed)
                    │           ├── exp67 (word-start semantic attention — failed)
                    │           └── exp68 (next-word-start MTP — not run)
                    └── exp69 (better quant: mlp_proj int6, attn int5, lzma, prune 5%)
                          └── exp70 (speed: batched NS5, EMA/10, set_to_none → 1.15 bpb)
                                ├── exp71 (output bias + label smooth — not run)
                                ├── exp72 (JEPA concept loss — failed)
                                ├── exp73 (warmdown focal + TTT WS — not run)
                                ├── exp74 (pRoPE 16/64 + q_gain + word bigram + LLR → **1.1456**)
                                │     ├── exp75 (word pool injection — failed: scale→0)
                                │     └── exp76 (dual word attention — failed)
                                ├── exp77 (progressive batch + seq_len curriculum)
                                ├── exp78 (WS loss curriculum — improved embeddings)
                                ├── exp79 (position ramp + late WS boost)
                                └── exp80 (best stack: pRoPE + bigram fix + pos ramp + outlier clamp)
```

## 6.3 Key Findings

### What Worked
1. **Partial RoPE 16/64** (exp74): 41% less quant error, better word-start attention. Frees 75% of head dims for semantic matching.
2. **Diverse q_gain init** (exp74): Heads specialized faster — sharp (>2.5) for syntax, soft (<1.5) for semantics.
3. **Cascading value residual** (exp63→all): Shallow layers independent (α≈0), deep layers form value highway (α≈0.9).
4. **Better quantization** (exp69): MLP proj→int6 (3.4× less error), attn→int5 (size-neutral), magnitude pruning 5%, lzma.
5. **Speed optimizations** (exp70): Batched NS5 via bmm, EMA every 10 steps, set_to_none=True, deferred .item().

### What Failed
1. **MiLe Loss** (exp66): Downweighted easy tokens before consolidation.
2. **JEPA concept loss** (exp72): Added memory/overhead, not enough steps.
3. **Word pool injection** (exp75): Model drove scale to 0.002 — redundant.
4. **Output bias** (exp71): Needs ~500 steps to build momentum — too slow.
5. **Focal loss during training** (exp35b, exp73): Always hurts easy token accuracy.
6. **CUDAGraphs + tied embeddings** (exp66): Incompatible, caused failure.

### Key Architectural Insights from Analysis
1. **Row 78 is a universal outlier**: 9/10 mlp.proj blocks have dim 78 as worst outlier (10-22× ratio). Per-row clamping ±3σ addresses this.
2. **Embedding uses 17% of capacity**: Effective rank 87/512. Word-start tokens (325) in 44 effective dims.
3. **Word-start norms 12% lower**: Tied embeddings structurally bias toward continuations (larger norm → higher logit).
4. **Deep layers form value highway**: VR alphas 7-10 → 0.9+ (strong V inheritance). Layers 2-5 independent.
5. **Block 0 attention barely used**: attn_scale=0.10. MLP-dominant layer.
6. **Block 4 c_q condition number 49,644**: Most quantization-sensitive matrix.

### Word-Start Problem Analysis
- Word-start tokens: 40.8% of tokens, 5.05 bpb → **66% of total loss**
- Continuation tokens: 48.3%, 1.56 bpb → 24% of total loss
- Root causes: (1) full RoPE starves semantic attention, (2) tied embeddings bias continuations, (3) uniform gradient allocation, (4) bigram token-level not word-level
- Best fix: partial RoPE (architectural, not loss manipulation)

### Community Gap Analysis (vs 1.1147 bpb)

| Feature | Community | Ours (exp74) | Gap |
|---------|-----------|-------------|-----|
| Partial RoPE | 16/64 ✓ | 16/64 ✓ | Closed |
| GPTQ quantization | Full Hessian ✓ | Per-row uniform | **Open** |
| Bigram | 3072×112 | 10240×128 (larger) | Ours bigger |
| Warmdown | 4000 iters | 1500 (premature trigger) | **Open** |
| Compression | LZMA-9 ✓ | LZMA ✓ | Closed |
| TTT | Dropped (negative) | Enabled | Different |
| Selective pruning | ±1 reconstruction | 5% magnitude | Different |

---

# 7. Phase 3.6: Diagnostic-Driven Era (exp83–exp87)

> **Hardware**: 1×H100 (simulating 8×H100)
> **Philosophy shift**: Understand the model first, then act
> **Best result**: exp85 — pre-quant **1.1517**, post-quant 1.1697, artifact 15.32 MB

## 7.1 Leaderboard

| Exp | Description | Pre-quant BPB | Post-quant BPB | Artifact | Status |
|-----|-------------|:-------------:|:--------------:|:--------:|:------:|
| **exp85** | Community-derived stack | **1.1517** | 1.1697 | **15.32 MB** | **Best pre-quant** |
| **exp74** | pRoPE+qgain+wbigram+LLR | 1.1539 | **1.1685** | 15.86 MB | **Best post-quant** |
| exp87 | Fast convergence (failed) | ~1.17 | — | — | Failed |
| exp84 | Diagnostic-tuned (failed) | ~1.17 | — | — | Failed |
| exp83 | Diagnostics baseline | ~1.15 | 1.1717 | ~16 MB | Diagnostic reference |

## 7.2 Evolution Tree

```
exp70 (speed-optimized baseline)
  ├── exp83 (diagnostics: grad norms, VR health, bigram, block0 attention)
  │     → Key finding: warmdown triggers at step 2200 (premature)
  │     → Key finding: embed/matrix ratio 3.6→7.3× (misleading for Muon)
  │     → Key finding: VR highway at layers 8-10, dead at 2-5
  ├── exp84 (diagnostic-tuned: VR_init=0.3, embed_lr=0.015)
  │     → FAILED: VR alphas went negative, embed_lr change made ratio worse
  │     → Lesson: VR_INIT must be 0.5, embed_lr ratio is misleading
  ├── exp85 (community-derived: pRoPE + x0-to-V + LN scale + clip search)
  │     → **1.1517 pre-quant** (best), 15.32 MB artifact
  │     → VE scale learned: block 8=0.88 (wants identity), block 9=0.08
  │     → VR exploded to 3.26 at layer 6 (LN scale instability)
  │     → Row 78 outlier: 4.5 (3× improved from exp70's 14.6)
  ├── exp86 (deep-opt: fused QKV + int8 critical + TF32)
  │     → Not yet run
  └── exp87 (fast convergence: embed preinit + prog unfreeze + block9 AdamW)
        → FAILED: embed preinit worse than random, prog unfreeze hurt co-adaptation
        → Lesson: don't fight orthogonal init + Muon
```

## 7.3 Key Findings

### What Worked (exp85)
1. **Partial RoPE 16/64**: Consistent across exp74 and exp85. Row 78 outlier 3× reduced.
2. **x0-to-V injection**: Block 8 grew ve_scale 0.3→0.88 — model WANTS token identity in deep-layer values.
3. **Clip search quantization**: Percentile-based clip per row. 25% quant error reduction. Zero training cost.
4. **Smaller bigram 5120×64**: 0.97 MB savings, artifact at 15.32 MB.
5. **Late warmdown min_steps=3000**: Delayed trigger from 2200 to 3100.

### What Failed
1. **CASCADE_VR_INIT < 0.5**: Both 0.1 and 0.3 caused negative VR alphas.
2. **Lowering TIED_EMBED_LR**: 0.035→0.015 made ratio worse (10.4×). Muon normalizes direction differently.
3. **Embedding pre-init from SVD**: val_loss=12.21 at step 0 (vs 6.93 random). Incompatible with orthogonal weights.
4. **Progressive layer unfreezing**: Prevented deep-shallow co-adaptation. VR highway didn't form.
5. **Block 9 QKV → AdamW**: Duplicate parameter issue, inconclusive.
6. **LN Scale 1/√(layer+1)**: VR alpha explosion at layers 6-7 (3.26×).

### Diagnostic Insights (from exp83)
- Block 0 attention dies by step 2000 (structural, not fixable)
- Block 1 x0_mix amplifies to 1.95× (compensates for dead block 0)
- Bigram scale decays 0.26→0.10 (attention supersedes local patterns)
- Grad clip never fires (threshold 0.3, actual norms 0.05-0.17)
- Loss oscillates ±0.07 during warmdown with 1500 iters (need 3500)

### Remaining Gap to Community (1.1147 bpb)

| Feature | Status | Estimated Impact |
|---------|--------|:----------------:|
| Partial RoPE | ✅ Matched | — |
| x0-to-V (vs community VE) | ✅ Novel alternative | Similar |
| Warmdown 3500 | ✅ Matched | — |
| Clip search | ✅ Adopted | -25% quant error |
| **Full Hessian GPTQ** | ❌ Not implemented | ~0.010 bpb |
| **VR alpha clamping** | ❌ Needed | Fix VR explosion |
| **LN Scale fix** | ❌ Needs investigation | TBD |
| Smaller bigram | ✅ Done | -0.97 MB |

## 7.4 Complete Lineage (exp60–exp87)

```
exp54b (clean baseline, quant bpb 1.2708)
│
├── exp60 (EMA, flash_attn3, 8×H100 sim) 🟡
│   └── exp61b (XSA all blocks) 🟢 Pre-quant 1.1504
│       └── exp63 (cascading V-residual) 🟢 Pre-quant 1.1377
│           │
│           ├── exp64 (MLP int6 quant) 🟡 Never ran
│           ├── exp65 (quant overhaul) 🟡 Never ran
│           │   └── exp66 (MiLe loss + NoPE) 🔴 MiLe hurt convergence
│           │       ├── exp67 (word-start semantic attention) 🔴 failed
│           │       └── exp68 (next-word-start MTP) 🟡 Never ran
│           │
│           └── exp69 (better quant) 🟢 Closed gap 0.035→0.015
│               └── exp70 (speed-optimized) 🟢 BASELINE
│                   ├── exp71 (output bias) 🟡 Never ran
│                   ├── exp72 (JEPA concept) 🔴 overhead, no improvement
│                   ├── exp73 (warmdown focal) 🟡 Never ran
│                   ├── exp74 (pRoPE + q_gain + word bigram) 🟢 BEST sliding 1.1456
│                   │     ├── exp75 (word pool) 🔴 scale→0.002
│                   │     └── exp76 (dual attention) 🔴 failed
│                   ├── exp77old (late warmdown) 🟡
│                   ├── exp77 (progressive batch) 🟡 Never ran
│                   ├── exp78 (WS loss curriculum) 🟢 Best embedding quality
│                   │   └── exp81 (pRoPE + WS curriculum) 🟡 failed
│                   │       └── exp82 (drop layer 10) 🟡 Never ran
│                   ├── exp79 (position ramp) 🔴 premise wrong
│                   ├── exp80 (best stack) 🔴 bigram-after-norm backfired
│                   ├── exp83 (diagnostics) 🟢 7 actionable insights
│                   ├── exp84 (diagnostic-tuned) 🔴 VR negative, embed_lr worse
│                   └── exp85 (community-derived) 🟢 BEST pre-quant 1.1517
│                         └── exp86 (deep-opt) 🟡 Not yet run
│                               └── exp87 (fast convergence) 🔴 All 3 changes hurt
```

## 7.5 Summary Statistics (exp60–exp87)

| Outcome | Count | Examples |
|---------|-------|---------|
| 🟢 Positive | 8 | exp61b, exp63, exp69, exp70, exp74, exp78, exp83, exp85 |
| 🟡 Neutral | 9 | exp60, exp64, exp68, exp71, exp73, exp77old, exp77, exp82, exp86 |
| 🔴 Negative | 10 | exp66, exp67, exp72, exp75, exp76, exp79, exp80, exp84, exp87, exp65→66 |

**Success rate: 29% positive, 36% neutral, 36% negative**

---

# 8. Phase 3b-Muon: Parallel Muon Optimizer (exp70_parallel_muon–exp91)

> **Base**: exp70_speed-opt_from_exp69
> **Goal**: Faster training via Parallel Muon optimizer
> **Best result**: val_bpb **1.1440** (exp70_faster_version_parallel_muon, step 7317, 1×H100)

## 8.1 Lineage

```
exp70_speed-opt_from_exp69 (original, DDP, 750ms/step)
├── exp70_faster_version_parallel_muon [🟢 POSITIVE: 12% speed, same final bpb]
│   ├── exp70_faster_vram_optimized [🔴 NEGATIVE: data loading issue]
│   ├── exp70_cuda_graphs_fused [🔴 NEGATIVE: no improvement]
│   ├── exp90_copy_head [🟡 NEUTRAL: concept validated, 40ms overhead]
│   └── reverted_exp70 [🟢 POSITIVE: clean base with all fixes]
│       └── exp91_smooth_v0residual [🟡 NEUTRAL: pending validation]
```

## 8.2 Results Table

| Exp | Name | step_avg | Final BPB | Quant BPB | Size | Tag |
|-----|------|:--------:|:---------:|:---------:|:----:|:---:|
| exp70_parallel_muon | Parallel Muon + Banks | **658ms** | **1.1440** | 1.1715 | 16.3MB | 🟢 |
| exp70_vram_opt | Double-buffer loader | 636ms | — | — | — | 🔴 |
| exp70_cuda_fused | CUDA Graphs + Triton | 662ms | — (higher loss) | — | — | 🔴 |
| exp90_copy | TopicCopyHead (hybrid freq+attn) | 698ms | — (partial) | — | — | 🟡 |
| reverted_exp70 | Clean parallel muon base | 656ms | 1.1440 | 1.1715 | 16.3MB | 🟢 |
| exp91_smooth | V0 residual + label smooth | — | — (pending) | — | — | 🟡 |

## 8.3 Key Findings

1. **Parallel Muon gives 12% speed** via reduce-scatter/all-gather overlap and bank-native batching
2. **Per-step convergence ~0.002-0.004 bpb worse** — different torch.compile graphs, init RNG ordering
3. **CUDA Graphs incompatible with FA3** — not usable together
4. **GPTQ requires Late QAT** — without QAT-adapted weights, Cholesky error cascades
5. **Adaptive warmdown is fragile** — v1 triggers on noise, v3 never triggers on oscillating loss. Pure time-based is robust.
6. **Copy mechanism validated**: 1.19 bpb copy advantage for repeated tokens, 1.77 bpb for word-start.
7. **Model self-analysis**: word_start_boost=0.017 (dead), cascading VR layers 1-8 ≈ 0 (dead), K_1 kurtosis=33.8 (outlier-heavy), byte tokens 96% cosine similar (confused)

## 8.4 Lessons Learned

7. **Double-buffering needs N >= grad_accum_steps buffers** — insufficient buffers cause issues
8. **Custom Triton kernels for elementwise ops rarely help** — torch.compile already fuses them; precision differences compound
9. **AWQ with weight-magnitude proxy is catastrophic** — must use real activation statistics from forward hooks
10. **Selective ±1 pruning (Code 2) > blind magnitude pruning** — targets least-impactful quantized values
11. **Init order matters for reproducibility** — nn.init.orthogonal_ consumes RNG; bank vs module ordering creates different trajectories

---

# 9. Phase 3c: Architecture Rewrite + Meta-TTT (exp92–exp109)

> **Hardware**: 8×H100
> **Base**: exp70_speed-opt → exp92 (major rewrite)
> **Key finding**: Meta-TTT has an architecture-limited ceiling
> **Best result**: exp101 — legal_ttt **1.11588**

## 9.1 Lineage

```
exp70_speed-opt (1.153 bpb)
└── exp92_banks-asyncmuon-partrope-qat-ve [🟢 1.131 bpb — major rewrite]
    └── exp93_meta-ttt-inner-outer [🟢 1.120 legal_ttt]
        └── exp95_size-ttt-opt-metattt2x [🟢 1.1169 legal_ttt — SOTA at time]
            ├── exp96_warmdown-fix-trigram-sgdttt [🟡 ~1.135]
            │   ├── exp98_metattt-randomsplit-momentum [🟡 ~1.135]
            │   │   └── exp99_tripleloop-parallelres [🟡 not run]
            │   └── exp97_fp8-pipeline [not run]
            ├── exp101_poscond-bigram-trigram [🟢 1.11588 legal_ttt — new baseline]
            │   ├── exp105a_no-metattt [🟡 ablation: meta-TTT = noise]
            │   ├── exp106_metasgd-crosschunk [🟡 ceiling confirmed]
            │   │   ├── exp107_sam-inner [🔴 hurts]
            │   │   └── exp108_sp8192-brotli [🟡 no results]
            │   └── exp109_shared-blocks-softgate [🔴 decoder dead]
            └── exp100_half-metattt [not tracked here]
```

## 9.2 Results Table

| Exp | Name | val_bpb | int6_bpb | legal_ttt | Tag |
|-----|------|:-------:|:--------:|:---------:|:---:|
| exp92 | Banks + Async Muon + Partial RoPE + QAT + VE | ~1.131 | — | — | 🟢 |
| exp93 | Meta-TTT inner/outer FOMAML | 1.136 | — | ~1.116 | 🟢 |
| exp95 | Size-opt + meta-TTT 2× | 1.1363 | — | 1.1169 | 🟢 |
| exp96 | Warmdown fix + trigram | ~1.135 | — | — | 🟡 |
| exp98 | Random-split FOMAML + momentum LR match | ~1.135 | — | — | 🟡 |
| exp99 | Triple loop + parallel residuals | — | — | — | 🟡 |
| **exp101** | **Position-conditional bigram hash** | **1.1352** | **1.13930** | **1.11588** | **🟢** |
| exp105a | No meta-TTT (ablation) | 1.1353 | 1.13956 | 1.11624 | 🟡 |
| exp106 | MetaSGD + cross-chunk FOMAML | 1.1377 | 1.14160 | ~1.118 | 🟡 |
| exp107 | SAM inner loop | 1.1384 | 1.1424 | 1.11898 | 🔴 |
| exp108 | SP8192 + Brotli | — | — | — | 🟡 |
| exp109 | Block sharing K=8 + SP8192 | 1.1500 | 1.1897 | — | 🔴 |

## 9.3 Key Findings

1. **Meta-TTT ceiling is architecture-limited**: 4 experiments (exp101, 105a, 106, 107) show identical TTT delta ~0.023 bpb regardless of optimizer (SGD, MetaSGD, SAM, none). Ceiling set by bank architecture (rank × dim).
2. **Position-conditional bigram hashing** (exp101): Zero-parameter trick — split hash space by token class (word-start vs within-word). +0.001 bpb.
3. **Block sharing fails across encoder/decoder boundary** (exp109): Shared decoder positions → near-zero scales. Soft gates diagnose but can't fix.
4. **SP8192 quant degradation 10× worse than SP1024** (exp109): Large embedding table (8192×512) poorly compressed.

---

# 10. Phase 3c-Community: Community SOTA (SP8192+)

> **Source**: Community contributions on parameter-golf repository
> **Impact**: Paradigm shift from our 1.1169 to 1.0744 bpb

## 10.1 The Three Community Breakthroughs

### 10.1.1 SP8192 + 3-Layer Recurrence (2026-04-09)

| Metric | Value |
|--------|:-----:|
| **val_bpb** | 1.0873 |
| **int6_bpb** | 1.0997 |
| **legal_ttt** | **1.0808** |
| **Hardware** | 8×H100 |

**Key innovations**: SP8192 tokenizer + 3-layer depth recurrence (blocks 3-5, 2 extra passes) + parallel residuals + QK_GAIN_INIT=5.25. 17 virtual layers from 11 physical.

### 10.1.2 WiderEmb + Tap-In V6 + TTT (2026-04-10, community)

| Metric | Value |
|--------|:-----:|
| **val_bpb** | 1.0813 |
| **int6_bpb** | 1.0980 |
| **legal_ttt** | **1.0788** |
| **3-seed mean** | 1.078825 |

**Key innovations**: Wider loop (3×3) + per-pass loop embeddings (3×512, zero-init) + Tap-In V6 cross-window n-gram C++ matcher + legal score-first TTT.

### 10.1.3 ImprovedParallelResiduals (2026-04-11, community PR #1523) — CURRENT BEST

| Metric | Value |
|--------|:-----:|
| **legal_ttt val_bpb** | **1.07438** (3-seed mean) |
| **val_bpb_std** | 0.00034 |
| **Artifact** | 15,959,005 bytes (71 bytes headroom) |
| **Hardware** | 8×H100 80GB SXM |
| **step_avg_ms** | 124.68 |

**Key innovations**: Richer parallel residual routing — attn/MLP outputs written into both lanes at block end, decoder skips on lane0 only. CUTLASS EVT fusion for reproducible throughput.

**Seed results:**
| Seed | val_bpb | post_ema_val_bpb | artifact_bytes | steps | ms/step |
|:----:|:-------:|:----------------:|:--------------:|:-----:|:-------:|
| 1337 | 1.07485 | 1.08286 | 15,958,373 | 4685 | 125.53 |
| 2024 | 1.07428 | 1.08242 | 15,956,287 | 4734 | 124.25 |
| 42 | 1.07403 | 1.08212 | 15,959,005 | 4733 | 124.26 |

## 10.2 Other Community-Adjacent Experiments

| Exp | Date | Description | Tag |
|-----|------|-------------|:---:|
| 2026-04-10_RecurStepFiLM_PooledRetrieval | 2026-04-10 | FiLM conditioning + pooled retrieval | 🟡 |
| 2026-04-10_10L_RecurStepFiLM_PooledRetrieval | 2026-04-10 | 10L variant | 🟡 |
| 2026-04-11_ImprovedParallelResiduals copy | 2026-04-11 | Copy/variant | 🟡 |
| 2026-04-11_newSota | 2026-04-11 | Community SOTA integration | 🟢 |
| 2026-04-11_11L_RecurStep3_loopedonly | 2026-04-11 | 11L recurrence step 3, looped-only | 🟡 |
| 2026-04-11_11L_RecurStep3_loops3 | 2026-04-11 | 11L with 3 loops | 🟡 |
| 2026-04-11_11L_RecurStep_StochDepth_ProgLoop | 2026-04-11 | Stochastic depth + progressive loop | 🟡 |
| 2026-04-11_11L_RecurStep_StochDepth_ProgLoop_KVCache | 2026-04-11 | + KV cache for recurrence | 🟡 |
| 2026-04-11_11L_Block10MLPHalf_RecurStepFiLM_PooledRetrieval | 2026-04-11 | Block 10 MLP halved + FiLM | 🟡 |
| loop_in_SP8192_3LayerRecur | 2026-04-13 | Loop detection (timestep embed, re-injection, per-loop RMSNorm) | 🟡 not trained |

---

# 11. Phase 3c-Frontier: Pushing Past Community (exp110–exp119)

> **Base**: ImprovedParallelResiduals (1.0744 legal_ttt)
> **Theme**: Tied embedding bottleneck
> **Result**: No improvement over community baseline

## 11.1 Results Table

| Exp | Name | val_bpb | int6_bpb | legal_ttt | Size | Tag |
|-----|------|:-------:|:--------:|:---------:|:----:|:---:|
| exp110 | Per-layer quant + trigram + PARALLEL_START=7 | — | — | — | — | 🟡 |
| exp111 | LoRA TTT (rank=8) + shrunk block 10 MLP + per-layer int5 | — | — | — | — | 🟡 |
| exp112 | Gradient rescaling on weak blocks | — | — | — | — | 🔴 |
| exp113 | Drop L0 MLP + batch schedule + MTP | — | — | — | — | 🟡 |
| exp114 | embed_dim=384 decouple | 1.0950 | — | — | fits | 🔴 |
| exp115 | embed_dim=384 + drop boundary MLPs | — | — | — | — | 🟡 |
| exp116 | embed_dim=384 + no x0 pathway | — | — | — | — | 🔴 |
| exp117 | embed_dim=448 tuned | 1.0877 | 1.0982 | 1.0814 (SW) | **16.28MB** | 🔴 |
| exp118 | embed_dim=416 + parallel_start=7 + clip tuned | 1.0915 | 1.1013 | 1.0850 | **16.44MB** | 🔴 |
| exp119 | Residual low-rank proj (rank=32) | — | — | — | — | 🟡 |

## 11.2 The Tied Embedding Bottleneck

The dominant theme: the model uses the same weight matrix for input embeddings and output projection. With SP8192, this (8192×512) matrix dominates the parameter budget and forces boundary blocks (0 and 10) to specialize for embedding space rather than general computation.

**Attempted fixes:**
- **embed_dim=448** (exp117): Good BPB (1.0877), activates boundary blocks (+50% effective contribution). But **16.28MB — over budget**.
- **embed_dim=416** (exp118): Similar story at **16.44MB**.
- **embed_dim=384** (exp114): Fits budget but loses 655K params → BPB regression.
- **Residual low-rank projection** (exp119): rank-32, zero param loss — theoretically correct fix. Not run to completion.

**Verdict**: The bottleneck is real. embed_dim≠model_dim activates boundary blocks but any dimension-change approach costs either params (regression) or fp16 passthrough overhead (budget overrun).

---

# 12. Misc: Co-occurrence QK Initialization

> **Date**: 2026-03-24
> **Hardware**: 1×H100
> **Separate exploration from main competition track**

| Metric | Value |
|--------|:-----:|
| **val_bpb** | 1.3525 |
| **Pre-quant val_bpb** | 1.3245 |
| **Artifact** | 15.55 MB |
| **Seeds** | 1 (seed 42) |
| **Steps** | 1099 |
| **Wallclock** | 600.138s |
| **Base PR** | #623 |

**Approach**: Initialize W_Q and W_K in layer 0 from bigram co-occurrence statistics via SVD:
1. Build 1024×1024 co-occurrence matrix from 2M training tokens (<3s)
2. Project into model_dim via random projection
3. Factorize C_proj = USV^T → Q/K weights where Q·K^T ≈ co-occurrence at step 0

Combined with LeakyReLU(0.5)², cyclic momentum (0.85–0.95), SWA over warmdown.

**Note**: exp87 later tried SVD-based embedding pre-initialization and it regressed. The difference: co-occurrence QK init changes attention *patterns*, while embedding SVD changes *representation space* (conflicts with Muon's orthogonal constraint).

---

# 13. Known Constraints

- **TTT requires compile-matched inference**: Standalone model loading needs the same torch.compile context as training for correct numerical results.
- **SP8192 quantization sensitivity**: Large embedding table (8192×512) needs GPTQ with SDClip — naive quantization degrades 10× worse than SP1024.
- **CUDA Graphs limited**: Incompatible with FA3 and tied embeddings in `reduce-overhead` mode.

---

# 14. Key Learnings by Phase

## 14.1 Phase 3a Lessons (exp00–exp18)

1. **Step count is king**: Reducing grad_accum = biggest single win.
2. **Don't fight the quantizer**: Better convergence naturally produces better-quantizing weights.
3. **N-gram tables have diminishing returns**: Bigram valuable, trigram+ marginal.
4. **Hash collision reduction matters less than expected**: Model routes around collisions.
5. **Auxiliary losses are dangerous**: JEPA caused biggest regression.
6. **Simple targeted fixes beat complex architectures**: 0-param + 1-param > 688K params.
7. **Quant gap is the real metric**: Optimize for post-quantization, not raw.

## 14.2 Phase 3b Lessons (exp27b–clean_54b)

8. **Residual norm control is high leverage**: RMSNorm after skip connections.
9. **Stacking orthogonal improvements works**: Sub-additive but substantial.
10. **Auxiliary losses fatal in compute-starved regimes**: Every gradient must reduce CE.
11. **RoPE base changes hurt quantization**: Different landscape = harder-to-compress weights.
12. **Long context is a dead end**: Loss flat after 256 tokens.
13. **Size budget matters**: Check BEFORE celebrating.
14. **LR schedule was completely broken**: Biggest single improvement (-0.0166 bpb).
15. **Layer sharing revives dead blocks**: Block 9 dead → shared block 3 revived it.
16. **More depth beats more width**: 10th block > 8 KV heads.
17. **Activation capping hurts**: Warmdown already smooths weights.
18. **AWQ alpha undertested**: Re-sweep for each new best model.
19. **Warmdown=400 optimal**: 4 SWA checkpoints, proper decay.
20. **Simpler is better**: Stripping features HELPED convergence.
21. **Resid-norm redundant with warmdown**: 7ms/step overhead not worth it.
22. **XSA last 2 is sweet spot**: Model wants everywhere but overhead too high on 1×H100.
23. **zstd > LZMA**: Better for structured quantized weights.
24. **torch.compile mode matters**: `max-autotune-no-cudagraphs` gives best tradeoff.

## 14.3 Phase 3.5–3.6 Lessons (exp60–exp87)

25. **Partial RoPE 16/64 universally good**: 41% less quant error, head specialization.
26. **Cascading VR creates value highway**: Natural deep-layer pattern.
27. **Diagnostics are invaluable**: exp83 discovered 7 insights informing 4 experiments.
28. **The model tells you what it wants**: Listen to learned parameters.
29. **Don't fight Muon's orthogonal constraint**: VR_INIT must be 0.5, embed pre-init fails.
30. **MiLe/focal/JEPA all fail**: Loss reweighting doesn't work in limited steps.
31. **Architectural changes DO work**: Partial RoPE, cascading VR, x0-to-V — all positive.
32. **Quantization improvements are free**: int6 for MLP proj, clip search — zero training cost.

## 14.4 Phase 3b-Muon Lessons

33. **Parallel Muon gives 12% speed** but per-step convergence slightly worse.
34. **Double-buffering needs sufficient buffers** for grad accumulation steps.
35. **Custom Triton kernels rarely help** — torch.compile already fuses elementwise ops.
36. **AWQ needs real activation statistics** — weight-magnitude proxy doesn't work.
37. **Init order matters for reproducibility**.

## 14.5 Phase 3c Lessons (exp92–exp119)

38. **Meta-TTT ceiling is architecture-limited**: TTT delta invariant at ~0.023 regardless of optimizer.
39. **Block sharing fails at encoder/decoder boundary**: Decoder positions → dead.
40. **Position-conditional bigram hashing**: Zero-parameter +0.001 bpb trick.
41. **Tied embedding bottleneck is real but hard to fix**: embed_dim changes bust budget.

## 14.6 Top-Level Synthesis

1. **Loss reweighting doesn't work in 7K steps** (MiLe, focal, JEPA, position ramp — all failed)
2. **Architectural changes DO work** (partial RoPE, cascading VR, x0-to-V, XSA-all — all positive)
3. **Quantization improvements are free bpb** (int6 for MLP proj, clip search — zero training cost)
4. **Don't fight the optimizer** (Muon's orthogonal constraint is a feature; VR_INIT and embed_lr must respect it)
5. **Diagnostics are invaluable** (exp83 discovered 7 insights that informed 4 subsequent experiments)
6. **The model knows what it wants** (block 0 attention dies = structural, VE scale at block 8 grows to 0.88 = model wants identity there)

---

# 15. TLDR: Top 20 Learnings Across All Phases

1. **Steps > everything else.** Cutting grad_accum from 8→2 doubled optimizer updates in the same wallclock — biggest single win in Phase 3a. Every ms/step matters when you only get 600 seconds.

2. **Fix your LR schedule before anything else.** ITERATIONS=20000 with 600s wallclock meant warmdown never fired. Fixing to ITERATIONS=1300 gave -0.017 bpb for free (exp34b). The model was training at max LR for 100% of training.

3. **Depth recurrence is the best parameter-efficiency trick (community).** 3-layer recurrence (blocks 3-5, 2 extra passes) from the community SP8192 baseline gives 17 virtual layers from 11 physical — the single biggest architectural win. Only works within the encoder, NOT across encoder/decoder boundary.

4. **SP8192 tokenizer is transformative (community).** Community's jump from SP1024 to SP8192 unlocked ~0.04 bpb improvement. But the larger embedding table (8192×512) needs GPTQ with SDClip — naive int8+brotli gives 10× worse quant degradation.

5. **Parallel residuals improve quantization for free (community).** GPT-J-style two-lane routing (attn/MLP read same input) from the community baseline collapses the quant gap vs single-lane. Cross-lane accumulation (community ImprovedParallelResiduals, PR #1523) pushed this further to 1.0744.

6. **Meta-TTT has an architecture-limited ceiling.** 4 experiments (exp101, 105a, 106, 107) show identical TTT delta ~0.023 bpb regardless of inner-loop optimizer (SGD, MetaSGD, SAM, none). The ceiling is set by bank architecture, not training.

7. **Auxiliary losses are fatal in compute-starved regimes.** JEPA, focal loss, boundary boost, MTP — every auxiliary objective tested hurt. With 1200-4700 steps, every gradient must directly reduce CE loss.

8. **Don't fight the optimizer.** Muon's orthogonal constraint is a feature. VR_INIT must be 0.5 (lower → negative alphas). Embed LR ratio is misleading because Muon normalizes gradient direction. Progressive unfreezing prevents co-adaptation.

9. **Quantization improvements are free BPB.** Per-row clip search (-25% quant error), int6 for MLP proj (3.4× less error), GPTQ with SDClip — all zero training cost. Always sweep AWQ alpha for each new best model.

10. **Simpler is better.** Stripping token-type embedding and loss weighting from exp53b actually HELPED. Fewer competing objectives = better convergence in limited steps.

11. **QK_GAIN_INIT=5.25 is a free win (community).** Monotonic improvement from 4.0→5.25 observed in the community SP8192 baseline. Per-head query gain initialization helps attention patterns specialize faster.

12. **Partial RoPE 16/64 is universally good.** Frees 75% of head dims for semantic matching, reduces quantization outliers 3×, and improves word-start attention. Consistent across every experiment it was tested in.

13. **Word-start tokens dominate total loss.** 25-40% of tokens but 42-66% of total loss. Mean loss 3.6-5.1 vs 1.2-1.6 for continuations. The best fix is architectural (partial RoPE), not loss manipulation (focal, weighting).

14. **Layer sharing revives dead blocks.** Block 9 was dead at 6.1% effective rank. Sharing block 3 at position 9 revived it to 10.3%. Fewer unique blocks = smaller artifact = more headroom for params.

15. **Resid-norm is redundant with warmdown.** Adding RMSNorm after skip connections improves quant but costs ~7ms/step (19 fewer training steps). With proper LR warmdown, weights are already smooth enough.

16. **Block sharing fails across encoder/decoder boundary.** Shared blocks at decoder positions converge to near-zero scales — effectively dead. Soft gates correctly diagnose the problem but can't override it (exp109).

17. **The model tells you what it wants.** Block 0 attention dies (structural, MLP-dominant). Block 8 ve_scale grows to 0.88 (wants identity in deep-layer values). Bigram scale decays 0.26→0.10 (attention supersedes local patterns). Listen to the learned parameters.

18. **Co-occurrence QK initialization works.** Initializing W_Q/W_K from bigram SVD gives meaningful step-0 attention patterns instead of random noise. Validated at 1.3525 bpb on 1×H100.

19. **Warmdown timing is critical.** warmdown=400 steps (start at step 900) gives 4 SWA checkpoints and proper LR decay. Too late (warmdown=200) → only 2 checkpoints. Community uses 3500-4000 iters on longer runs.

20. **Size budget is a hard constraint — check BEFORE celebrating.** embed_dim=448 achieved great BPB (1.0877) but at 16.28MB — over the 16MB limit. embed_dim=416 similar story at 16.44MB. Multiple experiments wasted on approaches that couldn't fit.

---

# 16. Appendix: Summary Statistics

## 16.1 Phase 3a (exp00–exp18)

19 experiments. Best: exp09/exp13 at quant bpb 1.3145.

## 16.2 Phase 3b (exp27b–clean_54b)

~30 experiments. Best: exp54b at quant bpb 1.2708.

## 16.3 Phase 3.5–3.6 (exp60–exp87)

| Outcome | Count |
|---------|:-----:|
| 🟢 Positive | 8 |
| 🟡 Neutral | 10 |
| 🔴 Negative | 10 |

**Success rate: 29% positive, 36% neutral, 36% negative**

## 16.4 Phase 3b-Muon (exp70_parallel_muon–exp91)

6 experiments. 2 positive, 2 neutral, 2 negative.

## 16.5 Phase 3c (exp92–exp119 + community)

| Outcome | Count | Examples |
|---------|:-----:|---------|
| 🟢 Positive | 8 | exp92, exp93, exp95, exp101, SP8192_3LayerRecur, WiderEmb, ImprovedParallelResiduals, CooccurrenceQKInit |
| 🟡 Neutral | 14 | exp96, exp98, exp99, exp105a, exp106, exp108, exp110, exp111, exp113, exp115, exp119, FiLM variants, recurrence variants |
| 🔴 Negative | 7 | exp107, exp109, exp112, exp114, exp116, exp117, exp118 |

**Success rate: 28% positive, 48% neutral, 24% negative**

## 16.6 Overall

~119+ experiments across all phases. Overall positive rate ~28-29%.

---

# 17. Complete Experiment Index

Every experiment across all phases in one table.

| # | Experiment | Base | Motivation | Result | Learning |
|:-:|-----------|------|-----------|:------:|---------|
| | **Phase 3a (exp00–exp18)** | | | | |
| 1 | exp00 (baseline-rerun) | exp27 | Establish baseline on A100 | Baseline | Quant bpb 1.3389; bigram.proj has worst quant error |
| 2 | exp01b (ln-scale-only) | exp27 | Test layer-norm damping | Not run | — |
| 3 | exp01c (ema-only) | exp27 | Test EMA weight averaging | Not run | — |
| 4 | exp01d (xsa-only) | exp27 | Test cross-sequence attention | Negative | XSA slows steps without quality gain |
| 5 | exp02 (speed-bigramfp16-awq) | exp00 | FP16 bigram + per-category AWQ | Negative | FP16 bigram blows artifact to 17.3MB |
| 6 | exp03 (qat-ste) | exp00 | Quantization-aware training via STE | Negative | QAT-STE destabilizes training; worst result |
| 7 | exp04 (no-cyclic-momentum) | exp00 | Test fixed momentum=0.95 | Negative | Cyclic momentum is slightly helpful as regularization |
| 8 | exp05 (grad-accum4) | exp00 | Double step count via accum 8->4 | **Positive** | Major breakthrough: 2x more steps = first sub-1.32 quant bpb |
| 9 | exp06 (swa-awq-accum2) | exp05 | Push accum to 2; SWA+AWQ tuning | **Positive** | Raw bpb breaks below 1.30 for first time |
| 10 | exp07 (tighter-swa-awq) | exp06 | SWA_EVERY=150, AWQ=0.7 | Neutral | Smaller artifact, quant bpb identical; sweet spot is SWA_EVERY=100 |
| 11 | exp08 (ctx-freq-bias) | exp05 | Learned token burstiness bias (+1 param) | Neutral | Redundant with attention; smallest artifact at 15.0MB |
| 12 | exp09 (padignore-wordboost) | exp06 | Skip pad tokens + word-start boost | **Positive** | Best quant bpb (1.3145); 0+1 params beat 688K trigram params |
| 13 | exp10 (trigram-unigram) | exp09 | Trigram hash table + unigram bias | Neutral | Best raw bpb but quant bpb regresses — extra params compress poorly |
| 14 | exp11 (trigram-slim-awq07) | exp10 | Slim trigram dim=48, AWQ=0.7 | Negative | dim=48 too small, AWQ too aggressive; double regression |
| 15 | exp12 (trigram64-awq06) | exp10 | Middle-ground trigram dim=64 | Neutral | Better than exp11 but worse than exp09; hash collisions too frequent |
| 16 | exp13 (multihead-gate-bigram) | exp09 | K=2 hash heads + context gate | **Positive** | Tied best quant bpb; collision reduction real but impact negligible |
| 17 | exp14 (engram-multiorder) | exp13 | 1-5gram, 10 lookups/position | Negative | Shared n-gram embeddings cause destructive interference |
| 18 | exp15 (engram-3order) | exp14 | 1-3gram with orthogonal subspaces | Neutral | Better isolation but each subspace too small (~42 dims) |
| 19 | exp16 (jepa-aux) | exp15 | JEPA predictor MLP, MSE loss | Negative | Biggest regression; fixed hash targets provide adversarial gradient |
| 20 | exp17 (byte-engram) | exp16 | Byte boundary features | Negative | No gain; base too weak to evaluate |
| 21 | exp18 (separate-trigram64) | exp13 | Separate 64-dim trigram + projection | Neutral | 688K extra params don't survive quantization |
| | **Phase 3b-Part1 (exp27b–exp33b)** | | | | |
| 22 | exp27b (resid-norm) | exp09 | RMSNorm after skip connections | **Positive** | High-leverage: attacks root cause of quant error (norm growth 19.7->89.5) |
| 23 | exp28b (perlayer-quant) | exp09 | Variable bitwidth per layer | Negative | 16% MSE reduction but over 16MB budget |
| 24 | exp29b (lossweight-typemb) | exp09 | 1.5x word-start loss + token-type embed | **Positive** | Gradient redistribution + structural signal both help |
| 25 | exp30b (combo) | exp09 | Stack all 3 validated improvements | **Positive** | Phase 3b-Part1 SOTA (1.3156); sub-additive but substantial |
| 26 | exp31b (rope-50k) | exp30b | RoPE base 10k->50k | Negative | Best raw bpb but quant gap widens; net negative after quantization |
| 27 | exp32b (aux-boundary) | exp30b | Auxiliary word-boundary classifier | Negative | Gradient waste; token-type already provides structural signal |
| 28 | exp33b (alt-rope-ntk) | exp30b | Alternating RoPE bases + NTK | Neutral | Marginal; positional loss still flat after 256 tokens |
| | **Phase 3b-Part2 (exp34b–exp48b)** | | | | |
| 29 | exp34b (lr-schedule-fix) | exp30b | Fix ITERATIONS 20000->1300 so warmdown fires | **Positive** | Single biggest improvement (-0.0166 bpb); warmdown was never firing |
| 30 | exp35b (focal-loss) | exp30b | Focal loss gamma=2 | Negative | Too aggressive; suppresses easy token gradients |
| 31 | exp36b (cappedact-labelsmooth) | exp30b | Activation cap + label smoothing | Negative | Both changes hurt independently and together |
| 32 | exp37b (fused-cap) | exp34b | Activation cap only | Negative | Cap hurts raw quality more than it helps quant |
| 33 | exp38b (speed-opt) | exp34b | Speed optimization | Neutral | Failed (OOM) |
| 34 | exp39b (swa-tuning) | exp34b | SWA parameter sweep | **Positive** | SWA_EVERY=100 confirmed optimal |
| 35 | exp42b (revive-block9) | exp34b | Share block 3 at position 9 | **Positive** | Dead block 9 (6.1% rank) revived to 10.3% |
| 36 | exp43b (boundary-boost) | exp42b | Boundary loss boost | Neutral | Too sparse (2.5% of positions) to matter in 1200 steps |
| 37 | exp44b (seqlen-curriculum) | exp42b | Sequence length curriculum | Negative | Speed regression |
| 38 | exp45b (awq-alpha07) | exp42b | AWQ alpha sweep (post-train) | Neutral | Alpha=0.7 gave -0.007 bpb free on exp42b |
| 39 | exp46b (full-mha) | exp42b | 8 KV heads (double from 4) | Neutral | Extra params but slower; depth > width |
| 40 | exp47b (warmdown200) | exp42b | Shorter warmdown=200 | Negative | Too late; only 2 SWA checkpoints vs 4 with warmdown=400 |
| 41 | exp48b (10blocks-depth) | exp42b | Add 10th unique block | **Positive** | Depth > width confirmed; quant bpb 1.2930 |
| | **Phase 3b-Part3 (exp53b–clean_54b)** | | | | |
| 42 | exp53b (lean-combo) | exp48b | Strip token-type + loss weighting | **Positive** | Removing features HELPED; quant bpb 1.2720 (-0.021!) |
| 43 | exp54b (xsa-zstd-ckfix) | exp53b | XSA last 2 layers + c_k fix + zstd | **Positive** | 1xH100 SOTA: quant bpb 1.2708 |
| 44 | exp55b (scaled-xsa-all) | exp54b | Learned XSA alpha on all layers | Neutral | Model wants XSA everywhere (alpha=0.75-0.99) but 20ms overhead |
| 45 | exp56b (fast-cosine-xsa) | exp55b | Cosine-scale XSA approximation | Negative | GQA head expansion is bottleneck, not XSA math |
| 46 | exp57b (lora-ttt) | exp54b | LoRA-based TTT | Negative | Failed |
| 47 | exp58b (resid-norm-on) | exp54b | Re-enable resid-norm | Negative | Redundant with warmdown; 7ms/step overhead not worth it |
| 48 | exp59b (pre-norm-skip) | exp54b | Pre-skip normalization | Negative | Same overhead as full resid-norm, no quality difference |
| 49 | clean_54b (final-arch) | exp54b | Clean submission version + TTT | **Positive** | Quant bpb 1.2723; clean baseline |
| 50 | clean_54b_v2 (bf16-roundtrip) | clean_54b | BF16 roundtrip test | Negative | Destroyed quality |
| | **Phase 3.5 (exp60–exp80)** | | | | |
| 51 | exp60 (8xh100-sim) | exp54b | EMA + flash_attn3 + 8xH100 simulation | Neutral | Infrastructure for scaling; not a bpb experiment |
| 52 | exp61b (xsa-all-warmdown) | exp60 | XSA all blocks + cosine warmdown | **Positive** | Pre-quant 1.1504; XSA-all works at scale |
| 53 | exp63 (cascade-vr) | exp61b | Cascading value residual + adaptive warmdown | **Positive** | Pre-quant 1.1377; discovered deep-layer value highway |
| 54 | exp64 (mlp-int6) | exp63 | MLP int6 quantization | Not run | Superseded by exp69 |
| 55 | exp65 (quant-overhaul) | exp63 | Full quantization overhaul | Not run | Ideas flowed into exp69 |
| 56 | exp66 (mile-nope) | exp65 | MiLe loss + partial NoPE | Negative | MiLe hurts early convergence |
| 57 | exp67 (ws-semantic-attn) | exp66 | Word-start semantic attention | Negative | Failed |
| 58 | exp68 (ws-mtp) | exp66 | Next-word-start MTP head | Not run | TTT data leakage concern |
| 59 | exp69 (better-quant) | exp63 | MLP proj->int6, attn->int5, LZMA, prune 5% | **Positive** | Closed quant gap 0.035->0.015; free improvements |
| 60 | exp70 (speed-opt) | exp69 | Batched NS5, EMA/10, set_to_none, deferred .item() | **Positive** | ~1.15 bpb; speed-optimized foundation for all subsequent |
| 61 | exp71 (output-bias) | exp70 | Output bias + label smooth + Z-loss | Not run | Needs too many steps to build momentum |
| 62 | exp72 (jepa-concept) | exp70 | JEPA concept loss | Negative | Added overhead, not enough steps even at 7K |
| 63 | exp73 (warmdown-focal) | exp70 | Warmdown focal + TTT weight | Not run | Safe late-training intervention (designed) |
| 64 | exp74 (prope-qgain-wbigram) | exp70 | Partial RoPE 16/64 + diverse q_gain + word bigram | **Positive** | Sliding bpb 1.1456; heads specialized (sharp+soft) |
| 65 | exp75 (word-pool) | exp74 | Inject previous word-start embedding | Negative | Model suppressed it (scale 0.1->0.002); redundant with attention |
| 66 | exp76 (dual-word-attn) | exp74 | Dual token + word attention | Negative | Failed |
| 67 | exp77old (late-warmdown) | exp70 | Late warmdown only | Neutral | Superseded by exp77 |
| 68 | exp77 (progressive-batch) | exp70 | Progressive batch + seq_len curriculum | Not run | Theoretically sound but non-standard |
| 69 | exp78 (ws-loss-curriculum) | exp70 | Word-start loss curriculum 0.1->1.0 | **Positive** | Best embedding quality; WS rank improved |
| 70 | exp79 (position-ramp) | exp70 | Position ramp 1.0->1.2 + late WS boost | Negative | Premise wrong: late positions are EASIER (90% repeats) |
| 71 | exp80 (best-stack) | exp70 | Combine pRoPE + bigram-after-norm + pos ramp + clamp | Negative | Bigram-after-norm destabilized attention |
| | **Phase 3.6 (exp81–exp87)** | | | | |
| 72 | exp81 (prope-ws-curriculum) | exp78 | Partial RoPE + WS curriculum | Neutral | Failed |
| 73 | exp82 (drop-layer10) | exp81 | Drop layer 10 + diverse q_gain | Not run | Designed only |
| 74 | exp83 (diagnostics) | exp70 | Full diagnostic run: grad norms, VR health, block analysis | **Positive** | 7 actionable insights; premature warmdown, dead blocks identified |
| 76 | exp84 (diagnostic-tuned) | exp83 | Apply diagnostics: VR_init=0.3, embed_lr=0.015 | Negative | VR went negative; embed_lr ratio misleading with Muon |
| 77 | exp85 (community-derived) | exp83 | pRoPE + x0-to-V + LN scale + clip search + small bigram | **Positive** | Best pre-quant (1.1517); ve_scale revealed model preferences |
| 78 | exp86 (deep-opt) | exp85 | Fused QKV + int8 critical + TF32 | Not run | Designed |
| 79 | exp87 (fast-convergence) | exp85 | Embed preinit SVD + progressive unfreeze + block9 AdamW | Negative | All 3 hurt; don't fight Muon's orthogonal constraint |
| | **Phase 3b-Muon (parallel optimizer)** | | | | |
| 80 | exp70_parallel_muon | exp70 | Parallel Muon via reduce-scatter/all-gather overlap | **Positive** | 12% speed (658ms vs 750ms); same final bpb |
| 81 | exp70_vram_opt | exp70_parallel_muon | Double-buffer data loader | Negative | Insufficient buffers for grad_accum |
| 82 | exp70_cuda_fused | exp70_parallel_muon | CUDA Graphs + Triton fusion | Negative | No improvement |
| 83 | exp90 (copy-head) | exp70_parallel_muon | TopicCopyHead (hybrid freq+attn) | Neutral | Concept validated; 40ms overhead |
| 84 | reverted_exp70 | exp70_parallel_muon | Clean base with all fixes | **Positive** | Clean foundation; 656ms/step |
| 85 | exp91 (smooth-v0residual) | reverted_exp70 | V0 residual + label smoothing | Neutral | Pending validation |
| | **Phase 3c (exp92–exp109)** | | | | |
| 86 | exp92 (banks-asyncmuon) | exp70 | Major rewrite: bank tensors + async Muon + partial RoPE + QAT + VE | **Positive** | ~1.131 bpb; paradigm shift in architecture |
| 87 | exp93 (meta-ttt) | exp92 | Meta-TTT inner/outer FOMAML | **Positive** | Legal_ttt ~1.116; first meta-TTT integration |
| 88 | exp95 (size-opt-metattt2x) | exp93 | Size optimization + meta-TTT 2x | **Positive** | Legal_ttt 1.1169; SOTA at the time |
| 89 | exp96 (warmdown-trigram) | exp95 | Warmdown fix + trigram hash | Neutral | ~1.135 bpb; marginal |
| 90 | exp97 (fp8-pipeline) | exp96 | FP8 pipeline + compile | Not run | Designed |
| 91 | exp98 (metattt-randomsplit) | exp96 | Random-split FOMAML + momentum LR match | Neutral | ~1.135 bpb; no improvement |
| 92 | exp99 (tripleloop) | exp98 | Triple loop + parallel residuals | Not run | Community merged first |
| 93 | exp100 (half-metattt) | exp95 | Half meta-TTT variant | Neutral | Not tracked in detail |
| 94 | exp101 (poscond-bigram) | exp95 | Position-conditional bigram hash by token class | **Positive** | Legal_ttt 1.11588; zero-param trick splitting hash by word-start |
| 95 | exp105a (no-metattt ablation) | exp101 | Remove meta-TTT to measure its contribution | Neutral | Meta-TTT = +0.00036 bpb (noise); ceiling is architectural |
| 96 | exp106 (metasgd-crosschunk) | exp101 | MetaSGD + cross-chunk FOMAML | Neutral | TTT delta invariant at ~0.023; ceiling confirmed |
| 97 | exp107 (sam-inner) | exp106 | SAM inner loop for TTT | Negative | SAM hurts; TTT delta still ~0.023 regardless of optimizer |
| 98 | exp108 (sp8192-brotli) | exp106 | SP8192 tokenizer + Brotli compression | Neutral | No stored results |
| 99 | exp109 (shared-blocks-softgate) | exp101 | Block sharing K=8 + soft gates + SP8192 | Negative | Decoder positions dead (near-zero scales); 10x worse quant |
| | **Community SOTA (SP8192+)** | | | | |
| 100 | SP8192_3LayerRecur (community) | Community | SP8192 + 3-layer recurrence (blocks 3-5) + parallel residuals + QK_GAIN=5.25 | **Positive** | Legal_ttt 1.0808; paradigm shift — 17 virtual layers from 11 physical |
| 101 | WiderEmb_TapInV6_TTT (community) | Community | Wider loop (3x3) + per-pass embeddings + Tap-In V6 + legal TTT | **Positive** | Legal_ttt 1.0788 (3-seed mean 1.078825) |
| 102 | ImprovedParallelResiduals (community PR #1523) | Community | Cross-lane attn/MLP accumulation + CUTLASS EVT fusion | **Positive** | **Legal_ttt 1.0744** — CURRENT BEST; 71 bytes headroom |
| 103 | RecurStepFiLM_PooledRetrieval (community) | Community | FiLM conditioning + pooled retrieval | Neutral | No improvement over base |
| 104 | 10L_RecurStepFiLM_PooledRetrieval (community) | Community | 10L variant of FiLM+retrieval | Neutral | No improvement |
| 105 | newSota (community) | Community | Community SOTA integration | **Positive** | Integration checkpoint |
| 106 | 11L_RecurStep3_loopedonly | Community | 11L, recurrence step 3, looped-only | Neutral | No improvement over ImprovedParallelResiduals |
| 107 | 11L_RecurStep3_loops3 | Community | 11L with 3 loops | Neutral | No improvement |
| 108 | 11L_RecurStep_StochDepth_ProgLoop | Community | Stochastic depth + progressive loop | Neutral | No improvement |
| 109 | 11L_RecurStep_StochDepth_ProgLoop_KVCache | Community | + KV cache for recurrence | Neutral | No improvement |
| 110 | 11L_Block10MLPHalf_RecurStepFiLM | Community | Block 10 MLP halved + FiLM + retrieval | Neutral | No improvement |
| 111 | loop_in_SP8192_3LayerRecur | Community | Loop detection: timestep embed + re-injection + per-loop RMSNorm | Neutral | Not yet trained |
| | **Frontier (exp110–exp119)** | | | | |
| 112 | exp110 (perlayer-quant-trigram) | ImprovedParallelResiduals | Per-layer quant + trigram + PARALLEL_START=7 | Neutral | No improvement |
| 113 | exp111 (lora-ttt-shrunk) | ImprovedParallelResiduals | LoRA TTT rank=8 + shrunk block 10 MLP | Neutral | No improvement |
| 114 | exp112 (grad-rescaling) | ImprovedParallelResiduals | Gradient rescaling on weak blocks | Negative | Doesn't fix structural tied-embedding bottleneck |
| 115 | exp113 (drop-l0-mtp) | ImprovedParallelResiduals | Drop L0 MLP + batch schedule + MTP | Neutral | Truncated logs |
| 116 | exp114 (embed384-decouple) | ImprovedParallelResiduals | embed_dim=384 to decouple boundary blocks | Negative | 655K param loss -> BPB regression (1.0950) |
| 117 | exp115 (embed384-asymmetric) | ImprovedParallelResiduals | embed_dim=384 + drop boundary MLPs | Neutral | Truncated |
| 118 | exp116 (embed384-no-x0) | ImprovedParallelResiduals | embed_dim=384 + remove x0 pathway | Negative | No stored results |
| 119 | exp117 (embed448-tuned) | ImprovedParallelResiduals | embed_dim=448 to activate boundary blocks | Negative | Good BPB (1.0877) but 16.28MB — over budget |
| 120 | exp118 (embed416-parstart7) | ImprovedParallelResiduals | embed_dim=416 + parallel_start=7 + tighter clip | Negative | Good BPB (1.0915) but 16.44MB — over budget |
| 121 | exp119 (residual-lowrank-proj) | ImprovedParallelResiduals | Residual low-rank projection (rank=32) | Neutral | Theoretically correct fix; not run to completion |
| | **Misc** | | | | |
| 122 | CooccurrenceQKInit | PR #623 | Init W_Q/W_K from bigram co-occurrence SVD | **Positive** | Val_bpb 1.3525 on 1xH100; meaningful step-0 attention patterns |

---

*Last updated: 2026-04-13*
