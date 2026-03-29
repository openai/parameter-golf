# Parameter Golf Priority Matrix

This document turns [op_tree.md](./op_tree.md) into a practical shortlist.

The goal is not to rank ideas by novelty. The goal is to rank them by:

- expected challenge impact
- implementation risk
- attribution clarity
- ease of comparison against the baseline

## 1. Prototype Now

These are the ideas most worth testing immediately.

### 1. Mixed Export / Sensitivity-Aware Compression

Why now:
- directly aligned with the artifact constraint
- easy to compare against the baseline
- can be tested without changing the training model

What to vary:
- int4/int6/int8 allocation by tensor family
- clip percentiles
- keep-float threshold
- late export-aware warmdown

Current vehicle:
- [experiments/exp01_mixed_export](/mnt/c/Users/foada/OneDrive/Documents/hyperactive-octupus/parameter-golf/experiments/exp01_mixed_export)

### 2. Factored Embeddings

Why now:
- embedding budget is a real issue in small models
- localized architectural change
- easy to attribute

What to vary:
- `FACTORIZED_EMBED_DIM`
- tied embedding LR
- matrix LR
- warmdown

Current vehicle:
- [experiments/exp02_factored_embeddings](/mnt/c/Users/foada/OneDrive/Documents/hyperactive-octupus/parameter-golf/experiments/exp02_factored_embeddings)

### 3. Schedule / Warmup Cleanup

Why now:
- directly challenge-aligned
- low complexity
- likely useful regardless of architecture

What to vary:
- warmup steps
- warmdown iters
- sequence schedule
- validation cadence for local proxies

Why not a separate mini-project yet:
- this can be swept first on the baseline and on both experiments

### 4. Small Eval-Time Gains

Why now:
- already validated by the leaderboard
- can improve judged score without redesigning training

What to vary:
- sliding eval
- chunkwise score-first LoRA updates
- context reuse

Caution:
- keep it tiny and legal

## 2. Prototype Later

These are serious ideas, but not the first place to spend time.

### 5. Shared Depth / Mild Recurrence

Why later:
- plausible bytes win
- more complex training/runtime interaction
- harder to isolate from throughput effects

What would make it move up:
- baseline and export path become stable
- we have a clean proxy for step-time tradeoff

### 6. Larger Vocabulary With Smarter Embedding Budget

Why later:
- could improve `bpb`
- introduces tokenizer/accounting scrutiny
- interacts with embedding cost and byte accounting

What would make it move up:
- factorized embedding experiment looks promising
- we have confidence in metric accounting

### 7. Stronger Export-Aware QAT

Why later:
- likely important
- but best added after we know which export policy is worth defending

What would make it move up:
- mixed export shows size wins but roundtrip quality loss

### 8. Partial / Deep-Layer Selective Attention Variants

Why later:
- may help capability-per-byte
- but easier to misread because runtime and architecture change together

What would make it move up:
- we want a more ambitious architecture experiment after embedding/export work

## 3. Moonshots / Non-Record Bias

These may be valuable, but they are not good first record-path bets.

### 9. Native BitNet / Ternary-First Training Path

Why moonshot:
- real upside
- training and kernel complexity are high
- much harder to debug than artifact-first compression

Best use:
- non-record exploration first

### 10. Learned Fast Weights

Why moonshot:
- elegant idea
- hard to make legal, cheap, and effective at once

Best use:
- later eval research after simpler score-first LoRA variants

### 11. TurboQuant-Style Online Quantized State

Why moonshot:
- inspiring, but mainly from KV-cache / online vector quantization work
- not a direct fit for the current baseline

Best use:
- only after we have a concrete adaptive state to quantize

### 12. Heavy Tokenizer Redesign

Why moonshot:
- high upside
- high scrutiny
- easy to get accounting wrong

Best use:
- only when we are ready to justify correctness carefully

## 4. Optuna Guidance

Optuna is useful when the search space is:

- small
- local
- cheap to evaluate on a proxy budget

### Best Optuna Targets

- export clip percentiles
- keep-float thresholds
- factorized embedding dim
- LR splits
- warmdown iters
- warmup steps

### Weak Optuna Targets

- tokenizer redesign
- huge architecture spaces
- eval methods with complex legality constraints
- seed brute force

## 5. Current Order Of Work

Recommended order:

1. baseline vs mixed export
2. baseline vs factored embeddings
3. sweep the obvious local knobs with Optuna proxies
4. only then decide whether to spend effort on recurrence, larger vocab, or legal TTT improvements

## 6. Short Version

Prototype now:
- mixed export
- factored embeddings
- schedule cleanup
- small legal eval gains

Prototype later:
- mild recurrence
- larger vocab with careful accounting
- stronger QAT
- partial/selective attention variants

Moonshots:
- native BitNet
- learned fast weights
- TurboQuant-style online state quantization
- heavy tokenizer redesign
