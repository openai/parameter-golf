# Verifily: Three-Tier Token Weighting + DCLS Salience (Non-Record)

## Approach

**Pure data-quality approach — zero architectural changes.**

We layer three data-quality components on top of an SP1024 11L 512d baseline (XSA-all + GPTQ + BigramHash + Parallel Muon). All modifications are in the loss computation and eval — zero additional parameters, zero extra memory beyond a 4MB bigram table.

### 1. Three-Tier Token Classification (Training)

Not all tokens deserve equal gradient. We classify each token into three tiers using a GPU-resident bigram frequency table built incrementally from training data:

| Tier | Condition | Weight | Rationale |
|------|-----------|--------|-----------|
| **Predictable** | P_bigram > ~p95 | 0.10 | Bigram handles these; free neural capacity |
| **Frontier** | Low P_bigram + high quality doc | 1.0 | Maximum gradient signal |
| **Noise** | Low P_bigram + low quality doc | 0.70 | Gentle gradient reduction |

Document quality is scored per-batch using two GPU-vectorized signals:
- Vocabulary richness (unique tokens / total via scatter)
- Repetition (fraction of tokens matching 4 positions back)

### 2. DCLS Salience Batch Reweighting (Training)

Per-batch loss multiplier in [0.85, 1.15] based on surprise (|batch_loss - EMA| / EMA) and document quality. High-surprise high-quality batches get amplified.

### 3. Quality-Conditioned Bigram Mixer (Eval)

At eval, mix neural predictions with bigram statistics where alpha is conditioned on document quality:
- High quality docs: alpha_base = 0.15 (trust neural more)
- Low quality docs: alpha_base = 0.30 (trust bigram more)
- Scaled by bigram confidence

## Results

2-seed validation on 8xH100 SXM (seed 999 lost to pod termination):

| Seed | BPB | Loss | Steps | Artifact |
|------|-----|------|-------|----------|
| 314 | 1.13414677 | 1.91495424 | 6524 | 15,841,796 bytes |
| 42 | 1.13285851 | 1.91277908 | 6732 | 15,917,868 bytes |
| **Mean** | **1.13350264** | **1.91386666** | | |

This places ~#16 on the leaderboard. The result demonstrates that data-quality signals provide measurable training improvement, but cannot close a ~0.05 BPB gap driven by architectural advances (SP8192, depth recurrence, parallel residuals, TTT).

## Ablation Environment Variables

```bash
VERIFILY_ENABLED=0         # Disable all Verifily components
VERIFILY_SALIENCE=0        # Disable salience reweighting only
VERIFILY_MIXER=0           # Disable eval-time bigram mixer only
VERIFILY_NGRAM_WARMUP=500  # Steps before activating token weighting
```

## Base Architecture (Unchanged)

SP1024, 11 layers, 512d, 8 heads, 4 KV heads, 3x MLP, XSA-all, BigramHash(2048,128), Parallel Muon+Adam, GPTQ-int6+LZMA, sliding window eval (stride 64)
