# Non-Record: CTW Eval-Time Augmentation on PR #549 SOTA Stack

**val_bpb = 1.1203** (seed 1337) | 15.85 MB | 8×H100 SXM

## Results

| Run | Seed | Steps | Step Avg | Pre-TTT BPB | Post-TTT BPB | TTT Time | Artifact |
|-----|------|-------|----------|-------------|-------------|----------|----------|
| Baseline (no CTW) | 1337 | 7,023 | 85.5ms | 1.1386 | **1.1203** | 352s | 15,854,788 |
| CTW (w=0.1, d=4) | 1337 | 7,023 | 85.5ms | 1.1386 | 1.1252 | 2,760s | 15,854,788 |

## Novel Contribution: CTW — A Negative Result

This submission integrates Context Tree Weighting (Willems, Shtarkov, Tjalkens 1995) into the PR #549 SOTA stack as an eval-time augmentation. CTW is a provably minimax-optimal sequential probability assignment over all variable-order Markov models up to depth D. It has zero artifact cost — the suffix tree is built entirely from already-scored tokens during evaluation.

### Integration

CTW was deeply integrated into the TTT scoring loop — not as a separate eval pass. During Phase 1 (score) of each TTT chunk, neural logits from TTT-adapted weights are mixed with CTW predictions per-token via log-linear interpolation before computing NLL:

```python
for each TTT chunk:
    Phase 1 — SCORE: sliding window eval
        for each scored token:
            mixed = (1 - w) * log_softmax(neural_logits) + w * log(ctw_probs)
            nll = cross_entropy(mixed, target)
            ctw.update(target)  # backward-looking: update AFTER scoring
    Phase 2 — TRAIN: SGD on chunk (unchanged from PR #549)
```

### Finding: CTW Hurts Strong Neural Models

**CTW degrades BPB by +0.005** at w=0.1, depth=4. The neural model at 1.12 BPB already captures n-gram patterns far better than any depth-4 Markov model. CTW's KT estimator over 1024 subword tokens is essentially a smoothed 4-gram model — the 11-layer transformer with 2048 context is already a strictly superior n-gram model. Mixing in a weaker predictor adds noise.

Additionally, the per-token Python loop makes CTW catastrophically slow (2,760s vs 352s for standard TTT), exceeding the 10-minute eval limit.

### Why This Matters

Other approaches to n-gram eval augmentation in Parameter Golf (PRs #727, etc.) succeed by using:
- Much higher order (5-7 grams) with count-min sketch
- Entropy-adaptive mixing weight (near-zero when neural model is confident)
- Vectorized GPU lookup (adds seconds, not minutes)

CTW's theoretical optimality over *all* variable-order Markov sources is irrelevant when the neural model already dominates the Markov component. The provable minimax guarantee applies to the class of tree sources — but the FineWeb validation set is not well-modeled by any depth-4 tree source that a 1024-vocab CTW can represent.

## Base Architecture (PR #549 by @abaybektursun)

- 11L, 512d, 8H/4KV, LeakyReLU(0.5)² MLP 3×
- Parameter Banking + Parallel Muon (FlashAttention 3)
- BigramHash(1536), XSA4, Partial RoPE(16), LN Scale, VE128
- EMA(0.997) + Tight SWA(50), GPTQ-lite int6 + LZMA-6
- Legal Score-First TTT (SGD, lr=0.002, 3 epochs, 32K chunks)

## Run Commands

```bash
# Baseline (reproduces PR #549)
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
CTW_WEIGHT=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# CTW enabled (negative result)
# Same as above but with: CTW_WEIGHT=0.1 CTW_DEPTH=4
```

## Credits

- CTW integration and negative result analysis: Anubhav (this submission)
- LeakyReLU²: PR #493 by @parinzee, PR #518 by @sofiabod
- Parallel Muon + Parameter Banking: PR #399 by @abaybektursun
- TTT recipe: PR #461 by @Christopher-Lee-McClendon
- Base model: PR #414 by @signalrush
