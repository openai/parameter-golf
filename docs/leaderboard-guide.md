# Parameter Golf Leaderboard Guide

**Summary**: Complete map of OpenAI Parameter Golf leaderboard entries, including record submissions, non-record runs, issues, and discussions.

**Last updated**: 2026-04-14

**Challenge deadline**: April 30, 2026

---

## Quick Links

| Link | Purpose |
|------|---------|
| [Main Repository](https://github.com/openai/parameter-golf) | OpenAI Parameter Golf repo |
| [All Issues](https://github.com/openai/parameter-golf/issues) | Leaderboard claims, discussions, rule clarifications |
| [All Pull Requests](https://github.com/openai/parameter-golf/pulls?q=is%3Apr+sort%3Acreated-desc) | Submissions (open), Closed PRs (finalized) |
| [Activity Feed](https://github.com/openai/parameter-golf/activity) | Real-time submission activity |

---

## Record Submissions (Main Leaderboard — 10min 8×H100)

Track: `10min_16mb` | Scoring: Welch's t-test (p < 0.01, 3 seeds minimum)

### Current SOTA

| Rank | Submission | BPB | Date | Key Techniques |
|------|-----------|-----|------|-----------------|
| #1 | [AR Self-Gen GPTQ + XSA + BigramHash 3072×112](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md) | **1.1147** | 2026-03-25 | Full Hessian GPTQ, AR self-gen calib, XSA-all layers, BigramHash 3072×112, EMA, SWA, sliding window, LZMA |
| #2 | [Vocab4096 MLPMult4 WD085](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD090/README.md) | 1.1148 | 2026-04-01 | Larger vocab (4096), MLP mult 4x, weight decay optimization |
| #3 | [11L XSA4 EMA Int6 MLP3x WD04](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md) | 1.1271 | 2026-03-20 | 11 layers, XSA 4 layers, Int6 QAT, EMA, MLP 3×, weight decay |

### Notable Record Submissions

- [2026-04-03: Muon EqR Depth Recurrence WD090 AllInt6](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/README.md) — Depth recurrence, equivariance regularization
- [2026-03-21: 11L XSA4 EMA PartialRoPE LateQAT](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md) — Partial RoPE, late QAT
- [2026-03-19: MixedQuant Int6Int8 SlidingWindow](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md) — Mixed quantization strategy
- [2026-03-18: LongContextSeq2048](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md) — Extended context length
- [Naive Baseline](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) — Starting point: 1.2244 BPB

---

## Non-Record Submissions (Exploration & Unlimited Compute)

Track: `track_non_record_16mb` | For research, alternative architectures, or over-10min training

### Recent Non-Record Submissions

| Submission | Score | Key Innovation | Link |
|-----------|-------|-----------------|------|
| Masked Text Diffusion (SP1024) | N/A | Masked diffusion for LM | [PR #1596](https://github.com/openai/parameter-golf/pull/1596) |
| CAT, Sparsity (Structured + Hessian), MoE | N/A | Multi-technique exploration | [PR #1537](https://github.com/openai/parameter-golf/pull/1537) |
| JEPA Baseline — LLM-JEPA Pretraining | 1.2699 | Joint Embedding Predictive Architecture | [PR #1480](https://github.com/openai/parameter-golf/pull/1480) |
| Quasi 10B from 50B (SP1024, 9×512, KV4) | N/A | Knowledge distillation approach | [2026-03-18 Record](https://github.com/openai/parameter-golf/blob/main/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |
| Hybrid Recurrent U-Net (aria-redefine-qbit) | 1.40 | Alternative architecture | [PR #1577](https://github.com/openai/parameter-golf/pull/1577) |
| GDN-Hybrid + Sliding Window Attention | N/A | Gated DeltaNet hybrid | [PR #1576](https://github.com/openai/parameter-golf/pull/1576) |
| Pure Neural GDN | 1.0226 | Gated DeltaNet (test-time free) | [PR #875](https://github.com/openai/parameter-golf/pull/875) |
| Universal Transformer | N/A | Depth-shared architecture | [PR #1088](https://github.com/openai/parameter-golf/pull/1088) |

---

## Key Leaderboard Issues & Discussions

### Leaderboard Mechanics

- **[Issue #87: Live Leaderboard for Parameter Golf](https://github.com/openai/parameter-golf/issues/87)** — Unofficial live leaderboard auto-updating every 30 min
- **[Issue #158: Leaderboard Monitor CLI + Claude Code Skill](https://github.com/openai/parameter-golf/issues/158)** — Terminal tool to track competition
- **[Issue #1017: A Field Guide to Valid Submissions](https://github.com/openai/parameter-golf/issues/1017)** — Clarification on record vs. non-record
- **[Issue #1522: PR Acceptance Order and Competition Rules](https://github.com/openai/parameter-golf/issues/1522)** — Discussion on competition fairness

### Leaderboard Design Proposals

- **[Issue #1151: Complementary Fixed-Data Leaderboard](https://github.com/openai/parameter-golf/issues/1151)** — Proposal for isolation of L(N) vs dataset effects
- **[Issue #519: Add Fixed-Step and Fixed-Compute Evaluation Benchmarks](https://github.com/openai/parameter-golf/issues/519)** — Benchmark standardization
- **[Issue #817: Leaderboard Adjustment - Add Compression Ratio](https://github.com/openai/parameter-golf/issues/817)** — Suggestion to include compression ratio metric
- **[Issue #942: Credits for Runpod from OpenAI](https://github.com/openai/parameter-golf/issues/942)** — References live leaderboard, technique map, BPB timeline

### Competition Rules & Validation

- **[Issue #83: Unofficial Leaderboard - Open PRs snapshot (2026-03-19)](https://github.com/openai/parameter-golf/issues/83)** — Top 5 memorization analysis
- **[Issue #1317: Unclear H100*8 SKU](https://github.com/openai/parameter-golf/issues/1317)** — Clarification on 8×H100 SXM requirement
- **[Issue #1082: Legality Check - Pre-eval TTT in Top Neural](https://github.com/openai/parameter-golf/issues/1082)** — TTT rule validation
- **[Issue #270: FarnsworthEngine v1 (1.1303 mean, 3 seeds)](https://github.com/openai/parameter-golf/issues/270)** — Detailed technique breakdown request

---

## Leaderboard Claims (Recent)

| Issue | Score | Technique Stack |
|-------|-------|-----------------|
| [#1267: Jtss-ux](https://github.com/openai/parameter-golf/issues/1267) | 1.1301 | Track: 10min_16mb |

---

## Landmark Submissions (Historical)

| PR | Score | Approach | Link |
|----|-------|----------|------|
| #64 | 1.1250 | DominationV3 + GPTQ-lite + TTT25 (mean 3 seeds) | [Link](https://github.com/openai/parameter-golf/pull/64) |
| #77 | 1.195 | Sliding Window + LoRA TTT | [Link](https://github.com/openai/parameter-golf/pull/77) |
| #293 | 1.2827 | Custom sp4096 BPE Tokenizer | [Link](https://github.com/openai/parameter-golf/pull/293) |
| #328 | N/A | MLX Prototyping (25+ validation experiments) | [Link](https://github.com/openai/parameter-golf/pull/328) |
| #386 | N/A | Depth-Recurrent Transformer (shared block ×12) | [Link](https://github.com/openai/parameter-golf/pull/386) |
| #468 | N/A | 10L Muon (15.46 MB artifact) | [Link](https://github.com/openai/parameter-golf/pull/468) |
| #517 | 0.978 | Goldfish ML Autonomous Research (100ep Cosine) | [Link](https://github.com/openai/parameter-golf/pull/517) |
| #679 | N/A | ASQU activation, Mixture of Convolutions, BankedLinear | [Link](https://github.com/openai/parameter-golf/pull/679) |
| #780 | N/A | Systematic Eval-Time Prediction Mixing | [Link](https://github.com/openai/parameter-golf/pull/780) |
| #1461 | N/A | Directional Semantic Vectors + 1-step Random Gradient | [Link](https://github.com/openai/parameter-golf/pull/1461) |

---

## Submission Directory Structure

Every record submission follows this path:

```
records/track_10min_16mb/
├── 2026-03-17_NaiveBaseline/
├── 2026-03-18_LongContextSeq2048/
├── 2026-03-19_MixedQuant_Int6Int8_SlidingWindow/
├── 2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/
├── 2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/
├── 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/  [CURRENT SOTA]
├── 2026-04-01_Vocab4096_MLPMult4_WD090/
├── 2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/
└── 2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/
```

Each folder contains:
- `README.md` — technique description, results, comparison to baseline
- `train_gpt.py` — exact code used
- `submission.json` — leaderboard metadata
- Run logs for each seed (3 minimum)

---

## How to Review SOTA

1. **Current leader**: [AR Self-Gen GPTQ + XSA + BigramHash (1.1147 BPB)](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md)
   - Read `README.md` for technique walkthrough
   - Review `train_gpt.py` for implementation
   - Check run logs for variance across seeds

2. **Track technique evolution**: 
   - Naive baseline: 1.2244 BPB (2026-03-17)
   - Sliding window: 1.1748 BPB
   - XSA layers: 1.1271 BPB
   - Full GPTQ: 1.1147 BPB (current)

3. **Find orthogonal techniques**:
   - Check non-record submissions for alternatives (JEPA, Diffusion, U-Net)
   - Check closed/open PRs for rejected/in-progress ideas

---

## Leaderboard Statistics

- **Total submissions tracked**: 60+ (records + non-records)
- **Record submissions**: ~40
- **Non-record explorations**: ~20
- **BPB range**: 0.978 (non-record) to 1.2827 (custom tokenizer)
- **Current leader margin**: 1.1147 (leader) vs 1.1301 (2nd place) = **0.0154 BPB (~1.3% gap)**

---

## Common Submission Patterns

| Pattern | Count | Example |
|---------|-------|---------|
| Quantization (GPTQ, Int6, Int8) | ~35% | AR Self-Gen GPTQ + XSA |
| Architecture (RoPE, Depth Recurrence, Alternatives) | ~25% | GDN, U-Net, Universal Transformer |
| Optimization (Muon, Adam, SGD variants) | ~20% | Muon EqR Depth Recurrence |
| Eval-time tricks (TTT, sliding window, prediction mixing) | ~15% | LoRA TTT + Sliding Window |
| Test-time free gains (XSA, BigramHash, weight sharing) | ~10% | BigramHash 3072×112 |
| Tokenizer / Embedding (vocabulary, vocab size, custom BPE) | ~5% | sp4096, sp8192 |

---

## References

- [[wiki/parameter-golf]] — Full technique analysis and SOTA breakdown
- [[docs/techniques-analysis]] — 25 leaderboard techniques with BPB impact
- [[docs/winning-techniques]] — Tier analysis and attack vectors
- [[docs/roadmap]] — 6-phase submission plan (deadline April 30, 2026)

---

**Last updated**: 2026-04-14 | **Generated via firecrawl-map** (60 leaderboard URLs)
