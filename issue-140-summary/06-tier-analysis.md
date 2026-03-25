# Performance Tier Analysis

The competition spans 0.10 BPB from baseline (1.2244) to best pending (1.1250). Each tier represents a qualitatively different approach.

## Tier 1: Tweaking the Baseline (1.20-1.22 BPB)

- **Approach:** 1-2 changes to baseline (longer seq, LR sweep, warmdown adjustment)
- **Limitation:** At int8+zlib, not enough model capacity to go further
- **Action:** Adopt the core five (int6, MLP 3x, sliding window, FP16 embed, zstd-22) as a package — worth ~0.05-0.07 BPB combined

## Tier 2: Stacking Known Techniques (1.15-1.18 BPB)

- **Approach:** Survey PRs, identify high-impact components, build combined recipe
- **Common mistakes:** SmearGate without OrthoInit (hurts), QAT from start (hurts), SWA without sufficient WD
- **Action:** Run ablations. Remove one technique at a time. Multi-seed validation (3 seeds) essential — single-seed can be off by 0.002+

## Tier 3: Understanding Interactions (~1.130-1.15 BPB)

- **Approach:** Ablation-driven development; understanding *why* each technique works
- **Canonical example:** #198 (1.1326) — 11L + SmearGate + BigramHash + OrthoInit + WD 0.04 + SWA + FA3 as a coherent system
- **Key markers:**
  - Every addition measured, not assumed helpful
  - Precision budgeting (fp16 only where quant error hurts most)
  - Divergent exploration (#76: higher LR + no QAT works; #215: Q matrices are naturally low-rank)
  - Statistical rigor (3+ seeds, significance testing)
- **Action:** Solidify baseline with multi-seed, then attack via Reptile meta-TTT or XSA on last 3-4 layers

## Tier 4: Architecture Frontier (<~1.130 BPB)

- **Approach:** System-level understanding of technique interactions; zero-parameter innovations
- **Leader:** #315 (1.1250) — Partial RoPE + LN Scale + Late QAT on XSA+EMA base
- **Key insight:** EMA outperforms SWA on XSA stack (reverse of Tier 3). Swapping components doesn't work — system-level understanding required.
- **Action:** Reptile meta-TTT (NOT naive TTT) on #315's base; Mousse optimizer; PolyCom activations; entropy-coded weights

## Critical Principle: Interactions > Technique Count

- TTT+XSA actively hurts (#303: +0.016 worse)
- EMA fails without XSA (#201) but succeeds with it (#287)
- 12L fails at seq2048 but works at seq1024 (#219 vs #76)
- Frontier submissions address specific failure modes of their base, not stacking unrelated improvements
- Evaluate untried combinations against your specific model's weaknesses

## 12L vs 11L: Current Data

- #332 (12L + full frontier stack): 1.1320 — **worse** than #315's 11L at 1.1250
- 12L at seq2048: ~6,600 steps (94ms/step); 11L: ~8,900 steps (67ms/step)
- The extra ~2,300 steps from 11L appear to matter more than 12L's extra capacity
- Confounded by #332 not having Late QAT active

## Frontier Endpoint Prediction

- ~1.117-1.125 BPB estimated
- Reptile meta-TTT on #315's base could project to ~1.114-1.120
- Sub-1.12 looks plausible
