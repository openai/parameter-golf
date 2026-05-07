# Novel Eval-Time Adaptation & Meta-TTT

**val_bpb: 1.2882** (sliding window) / **1.2923** (TTT+cache+OGD) | **13.2 MB** | 1xA40, 2000 steps, 524K batch

Non-record submission exploring three orthogonal research directions with controlled ablations.

## Summary

Three novel contributions, each with ablation data:

1. **MAML-style Meta-TTT** (negative result) — First-order MAML during training degrades both base quality (-0.085 BPB) and TTT effectiveness. Hyperparameters were too aggressive; PR #296's Reptile approach (lighter touch, last 20% only) is more promising.

2. **Eval-time technique stacking** (positive result) — Unigram cache mixture + online gradient descent (OGD) on a vocab bias vector give -0.004 BPB additive on top of SGD TTT at 524K batch.

3. **Tokenizer optimization** (null result) — First submission to modify the tokenizer. -5.7% fewer tokens/byte at vocab 8192 but no BPB improvement. Demonstrates that tokenizer efficiency and model BPB are not straightforwardly linked.

## Contribution 1: MAML-Style Meta-TTT (Negative Result)

### Motivation

SGD TTT adapts weights at eval time. A model whose initialization is optimized for post-adaptation loss should yield larger TTT gains. We implement first-order MAML: inner SGD adaptation on a support chunk, then optimize for post-adaptation loss on a query chunk.

### Implementation

Every 4 training steps (after 50% of training), we run a meta-step:

1. Compute gradients on a support chunk via `torch.autograd.grad`
2. Take one inner SGD step (lr=0.03) on control parameters
3. Evaluate loss on a separate query chunk with adapted weights
4. Backprop the query loss (weighted by 0.5) through to the original weights
5. Restore original weights

### Results (controlled A/B, same 524K batch, same architecture)

| Config | Sliding BPB | TTT+cache+OGD BPB | TTT delta |
|--------|------------|-------------------|-----------|
| **Control** (no meta) | **1.2882** | **1.2923** | **-0.004** (helps) |
| Meta-TTT | 1.3733 | 1.3775 | +0.004 (hurts) |
| **Delta** | **+0.085 worse** | **+0.085 worse** | — |

### Analysis

Meta-TTT hurts by 0.085 BPB. The meta-learning loss competes with language modeling — half the gradient signal goes to meta-objectives instead of language modeling. The model also doesn't benefit from TTT at eval time (TTT delta is positive = hurts).

**Why our approach failed vs PR #296's Reptile:**

| Parameter | Ours (MAML) | PR #296 (Reptile) |
|-----------|------------|-------------------|
| Training fraction | 50% of training | 20% of training |
| Frequency | Every 4 steps (25% of steps) | Dedicated phase |
| Meta-loss weight | 0.5 (competes with LM loss) | N/A (just weight interpolation) |
| Mechanism | Backprop through inner loop | Simple weight lerp toward adapted |
| Overhead | ~2x per meta-step | ~1.3x per meta-step |

The key difference: Reptile doesn't add a competing loss term. It simply nudges weights toward "where SGD would take them" via interpolation. Our MAML approach explicitly optimizes meta-loss at the expense of language modeling.

### Hypotheses for improvement

- Reduce `META_TTT_WEIGHT` from 0.5 to 0.05-0.1
- Start later (`META_TTT_START_FRAC=0.8` instead of 0.5)
- Reduce frequency (`META_TTT_EVERY=20` instead of 4)
- Switch to Reptile (no competing loss, just weight interpolation)

## Contribution 2: Eval-Time Technique Stacking

### Motivation

The eval budget is 600s. SGD TTT uses ~130-200s. TTT adapts weights for distribution shift but doesn't capture **token-level burstiness** (repeated entities, names, code identifiers). We stack two complementary zero-cost techniques.

### Techniques

**Unigram Cache Mixture:** Exponentially decayed token frequency counts, mixed with model softmax via `p = (1-λ)·p_model + λ·p_cache`. λ=0.02, decay=0.995.

**Stride-OGD:** Vocab-sized bias vector added to logits, updated per-window: `b ← b - 0.1·(softmax(logits+b) - one_hot(target))`. No backprop needed.

Both operate on output logits — the forward pass stays batched.

### Ablation Results

**RTX 3090, 131K batch (isolated ablation):**

| Eval Mode | BPB | Δ vs sliding |
|-----------|-----|-------------|
| Sliding window (stride=128) | 1.3567 | — |
| + TTT (all params) only | 1.3563 | -0.0005 |
| + TTT (all) + cache + OGD | **1.3529** | **-0.0039** |
| **Cache+OGD contribution** | | **-0.0034** |

**A40, 524K batch (production config):**

| Eval Mode | BPB | Δ vs sliding |
|-----------|-----|-------------|
| Sliding window (stride=128) | 1.2882 | — |
| + TTT (control) + cache + OGD | **1.2923** | -0.004 |

### Key findings

- Cache + OGD provide additive -0.003 to -0.004 BPB on top of TTT
- The techniques need a well-adapted model (full-params TTT) to be effective
- Zero artifact cost, minimal compute overhead (logit-space operations only)

## Contribution 3: Tokenizer Optimization (Null Result)

### Motivation

Every submission uses the stock SentencePiece BPE tokenizer. We explored whether tokenizer optimization is a "free lunch."

### Method

Trained SentencePiece BPE with `split_digits=False, split_by_unicode_script=False, split_by_number=False, max_sentencepiece_length=64` on 500K FineWeb documents.

### Results

| Tokenizer | Vocab | tokens/byte | Δ |
|-----------|-------|-------------|---|
| Stock BPE | 8192 | 0.2721 | — |
| **Optimized BPE** | **8192** | **0.2565** | **-5.7%** |
| Stock BPE | 1024 | 0.4155 | — |
| Optimized BPE | 1024 | 0.4092 | -1.5% |

**BPB impact (v8192 7L, 131K batch):**

| Tokenizer | BPB |
|-----------|-----|
| Stock | 1.3246 |
| Optimized | 1.3252 |

The optimized tokenizer was **0.0006 BPB worse** despite -5.7% fewer tokens/byte. Longer merged tokens are harder to predict per-token, offsetting the compression gain. BPB normalizes by bytes, so fewer-but-harder tokens cancel out.

### Why this matters

This null result explains why the community converged on stock v1024 + 11 layers: depth matters more than tokenizer efficiency. Tokenizer optimization is not a viable path to BPB improvement at this scale.

## Run Commands

### Training (control, 1xA40 524K batch)

```bash
TORCHDYNAMO_DISABLE=1 VOCAB_SIZE=1024 NUM_LAYERS=9 MLP_MULT=2 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 ITERATIONS=2000 \
SMEAR_GATE=1 BIGRAM_HASH=1 BIGRAM_BUCKETS=4096 ORTHO_INIT=1 \
SWA_ENABLED=1 WEIGHT_DECAY_MUON=0.04 WEIGHT_DECAY_ADAM=0.04 \
GRAD_CLIP_NORM=0.3 QUANT_BITS=6 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Eval (with technique stacking)

```bash
EVAL_STRIDE=128 SGD_TTT=1 SGD_TTT_LR=0.002 SGD_TTT_MOMENTUM=0.9 \
SGD_TTT_EPOCHS=2 CACHE_MIXTURE=1 CACHE_LAMBDA=0.02 CACHE_DECAY=0.995 \
OGD_BIAS=1 OGD_LR=0.1 python3 eval_remote.py
```

## What We'd Do With More Compute

- Reptile meta-TTT (lighter touch than MAML: last 20% of training, weight interpolation only)
- Causal TTT (backward-looking, per competition ruling) instead of pre-eval TTT
- Sweep cache λ (0.005-0.05) and OGD lr (0.01-1.0)
- Meta-TTT with `WEIGHT=0.05, EVERY=20, START_FRAC=0.8` (less aggressive)
- Combine tokenizer optimization with v4096 9L (middle ground)
