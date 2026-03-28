# JEPA-LM: When Synthetic Success Doesn't Transfer to Real Language

**PR #1012 | Non-Record Submission (Research Contribution / Negative Result)**
**Author:** Himanshu Dongre ([@himanshudongre](https://github.com/himanshudongre))
**Companion:** [S4D-Lin SSM Hybrid PR #1013](https://github.com/openai/parameter-golf/pull/1013) (where this research led next)
**Compute:** $0 (all experiments on Mac Mini M4, MPS backend)
**Status:** Negative result -- JEPA provides no meaningful benefit for real language modeling at this scale

---

## The Short Version

I implemented JEPA (Joint Embedding Predictive Architecture) as a training-time auxiliary loss for language modeling. On synthetic Markov chain data, JEPA showed a **dramatic -19.5% cross-entropy improvement** over a standard Transformer baseline. On real English text (Project Gutenberg), the improvement collapsed to **-0.24%** with **+40% throughput overhead** -- a massively net-negative result.

This is a cautionary tale about validation methodology: synthetic benchmarks can be wildly misleading. The repetitive statistical patterns in Markov chains are exactly what JEPA's representation prediction excels at, but natural language doesn't have those patterns at the scale where JEPA's overhead is justified.

I'm submitting this because JEPA is on OpenAI's "Requests for PRs" wishlist, and negative results with clear explanations are often more valuable than marginal positive ones. If you're considering JEPA for Parameter Golf, this document will explain why it doesn't work and save you from making the same mistake.

---

## Table of Contents

1. [Motivation](#motivation)
2. [How JEPA-LM Works](#how-jepa-lm-works)
3. [Synthetic Data Results (Promising)](#synthetic-data-results-promising)
4. [Real Text Results (Disappointing)](#real-text-results-disappointing)
5. [Why the Gap?](#why-the-gap)
6. [Could JEPA Work With Changes?](#could-jepa-work-with-changes)
7. [Connection to SSM Work](#connection-to-ssm-work)
8. [Reproducing These Results](#reproducing-these-results)

---

## Motivation

After my two-pass n-gram rescoring PR (#846) was closed in the enforcement sweep (Issue #677), I committed to pursuing pure architectural innovation -- no eval-time tricks. I wanted training-time techniques that produce better model weights without modifying the evaluation procedure.

JEPA was appealing for three reasons:

1. **Zero eval-time overhead.** The JEPA target encoder is ephemeral -- used only during training, never stored in the 16MB artifact. At eval time, the model is a standard Transformer. This sidesteps the throughput trap that kills novel architectures (see PR #831's analysis).

2. **Richer gradient signal.** Instead of just predicting next-token distributions (a sparse signal from a 1024-way vocabulary), JEPA predicts dense representations in a learned latent space. In theory, this provides a more informative training signal.

3. **OpenAI asked for it.** JEPA is explicitly listed in the "Requests for PRs" section of the README.

## How JEPA-LM Works

### Standard LM Training
```
Input tokens -> Encoder -> LM Head -> Cross-entropy loss vs. true next token
```

### JEPA-LM Training
```
Input tokens -> Online Encoder -> LM Head -> CE loss (standard)
                    |
                    +-> Predictor -> predicted representation of next token
                                            |
                                            v
Input tokens -> Target Encoder (EMA) -> target representation of next token
                                            |
                                            v
                                    JEPA loss = ||predicted - target||^2
                                            |
                                            v
                            Total loss = CE + lambda * JEPA
```

The target encoder is an exponential moving average (EMA) of the online encoder, updated every step. It provides stable prediction targets without collapse (no gradient flows through the target encoder).

### Key Design Choice: Training-Only

The target encoder, predictor, and JEPA loss are **completely discarded after training**. The exported model is a standard Transformer. The hypothesis is that JEPA's auxiliary loss produces better internal representations that persist in the trained weights.

### Parameter Overhead During Training

| Component | Params | Note |
|-----------|--------|------|
| Online encoder | Same as baseline | Standard model |
| Target encoder | Same as baseline | EMA copy (not stored) |
| Predictor | ~dim x dim | Small MLP |
| **Training overhead** | **~2x memory** | Target encoder is full copy |

At eval/export time: **zero overhead**. The target encoder doesn't exist.

## Synthetic Data Results (Promising)

### Setup
- Synthetic Markov chain text (controlled statistical patterns)
- dim=192, 6 layers, 3000 training steps
- Mac Mini M4, MPS backend

### Results

| Model | Final CE | ms/step | CE vs Baseline |
|-------|----------|---------|----------------|
| Pure CE (Baseline) | 0.8031 | 210-590ms | -- |
| **JEPA-LM Hybrid** | **0.6466** | 750-812ms | **-19.5%** |
| JEPA-LM + MoD | 0.7644 | 454-888ms | -5% |

The JEPA-LM hybrid showed:
- **-19.5% cross-entropy improvement** over pure CE training
- Slower convergence for the first ~1400 steps, then dramatic improvement
- The crossover point suggested JEPA needs time to learn useful representations before they benefit the LM head

I was genuinely excited. A 19.5% improvement, even with throughput overhead, seemed like it would easily translate to a BPB improvement at full scale.

It didn't.

## Real Text Results (Disappointing)

### Setup
- 4 Project Gutenberg books (1.9MB of real English text)
- Same architecture: dim=192, 6 layers, 2000 steps
- Same JEPA configuration that showed -19.5% on synthetic data

### Results

| Model | Eval CE | ms/step | CE vs Baseline | Net Impact |
|-------|---------|---------|----------------|------------|
| Pure CE (Baseline) | 1.2779 | 210.6ms | -- | -- |
| **JEPA-LM** | **1.2748** | **294.4ms** | **-0.24%** | **-28.2%** |

- Cross-entropy improvement: **virtually zero** (-0.24%)
- Throughput penalty: **+39.8%** (the target encoder EMA is expensive)
- Net competition impact: **strongly negative** (-28.2% after accounting for fewer training steps)

### The Reality Check

At competition scale (600s on 8xH100), the +40% throughput overhead means ~1500 fewer training steps. A -0.24% quality improvement cannot compensate for losing 30% of your training budget.

## Why the Gap?

### Markov Chains Have Exploitable Structure

Synthetic Markov chain data has **simple, repetitive statistical patterns** -- transition probabilities between states are fixed and learnable. JEPA's representation prediction excels here because:

1. The target encoder learns stable representations of these patterns quickly
2. The predictor can accurately forecast what the next representation should be
3. The resulting gradient signal genuinely helps the online encoder learn faster

### Natural Language Doesn't

Real English text has:
- **Long-range dependencies** that change with context
- **Semantic ambiguity** where the same prefix leads to many valid continuations
- **Non-stationary statistics** across documents, genres, and topics

JEPA's representation prediction becomes nearly meaningless when the next token is genuinely unpredictable from the current representation. The predictor can't learn a useful mapping from "current representation" to "next representation" because the mapping is inherently many-to-many in natural language.

### The Overhead Wasn't Worth It

Even if JEPA provided a marginal quality improvement, the training-time overhead is fundamental:
- Target encoder EMA update: O(params) per step
- Forward pass through target encoder: same cost as the main model
- Predictor forward + loss: small but nonzero

In Parameter Golf, every millisecond costs ~7 training steps. At +40% overhead, JEPA would need to improve per-step learning by 40% just to break even. That's a much higher bar than the -0.24% it achieved.

## Could JEPA Work With Changes?

### Maybe, But Unlikely at This Scale

Possible improvements:
1. **Cheaper target encoder:** Only EMA-update a subset of layers. Reduces overhead but also reduces the signal quality.
2. **Larger scale:** At dim=512+, the representation space is richer, and JEPA predictions might be more useful. But the overhead also grows.
3. **Different prediction targets:** Instead of next-token representation, predict chunk-level or multi-token representations. More stable targets, potentially more useful for language.
4. **Domain-specific fine-tuning:** JEPA might work better on highly structured text (code, math) where next-token prediction is more deterministic.

### My Honest Assessment

JEPA is a beautiful idea for vision (where spatial structure makes representation prediction natural). For language modeling at the scale of Parameter Golf, the cost-benefit ratio is wrong. The auxiliary loss is expensive and the signal is too weak for natural language at dim=192-512.

If someone wants to push this further, I'd suggest trying at dim=768+ with a much cheaper target encoder (EMA only the last 2 layers). But I wouldn't bet on it.

## Connection to SSM Work

After JEPA failed, I pivoted to S4D-Lin State Space Models (see companion PR). The key lesson from JEPA informed my SSM approach: **always validate on real text first**. I ran the SSM through local real-text validation at seq_len=512 before spending any GPU credits.

Ironically, the SSM local tests also turned out to be misleading (for different reasons -- the quality advantage at dim=192 didn't hold at dim=512). The full story is in the SSM PR.

## Reproducing These Results

### Local Experiments

All experiments run on Mac Mini M4 (MPS backend), no GPU required.

```bash
# Synthetic data test (Markov chains)
python3 -u jepa_mod_experiment.py

# Real text test (requires text_corpus.txt from Project Gutenberg)
python3 -u jepa_real_text_test.py
```

### Files

- `jepa_mod_experiment.py`: Full JEPA-LM implementation with CE, JEPA hybrid, and JEPA+MoD variants
- `jepa_real_text_test.py`: Real text validation script
- `jepa_mod_results.json`: Synthetic data benchmark results
- `jepa_real_text_results.json`: Real text benchmark results

---

*This submission checks off "JEPA" from the Requests for PRs wishlist. Sometimes the most valuable research contribution is showing definitively why a promising idea doesn't work in a specific setting.*
