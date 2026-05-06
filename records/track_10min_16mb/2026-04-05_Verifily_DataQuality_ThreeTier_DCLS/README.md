# Verifily Data Quality for Parameter Golf

## Approach

We layer **three data-quality components** on top of the current SOTA architecture
(XSA-all + GPTQ + BigramHash + Parallel Muon). All modifications are in the
loss computation and eval pipeline — zero architecture changes, zero additional
parameters.

### 1. Three-Tier Token Classification (Training)

Not all tokens deserve equal gradient. We classify each token into three tiers
using a GPU-resident bigram frequency table (4MB) built incrementally from
training data:

| Tier | Condition | Weight | Rationale |
|------|-----------|--------|-----------|
| **Predictable** | P_bigram > ~p95 | 0.10 | Bigram handles these; soft nudge to free neural capacity |
| **Frontier** | P_bigram low + high quality doc | 1.0 | Learnable, high-value tokens |
| **Noise** | P_bigram low + low quality doc | 0.70 | Lower quality, gentle gradient reduction |

Weights validated locally on M4 Max: soft weights (0.10/0.70, eff. batch ~81%) beat
aggressive weights (0.05/0.30, eff. batch ~53%) — the model needs most of its gradient
signal intact. The threshold is data-driven (~p95 of bigram probability distribution).

Document quality is scored per-batch using two GPU-vectorized signals:
- **Vocabulary richness**: unique tokens / total (scatter-based, O(1) per token)
- **Repetition**: fraction of tokens matching 4 positions back

### 2. DCLS Salience Batch Reweighting (Training)

Adapted from Verifily's [DCLS](https://github.com/verifily/dcls) episodic memory
salience ranking. Per-batch loss multiplier in [0.85, 1.15]:

```
salience = 0.4 * surprise + 0.3 * doc_quality + 0.3 * 0.5
multiplier = 0.85 + 0.30 * salience
```

Where `surprise = |batch_loss - EMA_loss| / EMA_loss`. High-surprise, high-quality
batches get amplified; low-surprise, low-quality batches get dampened.

### 3. Quality-Conditioned Bigram Mixer (Eval)

At eval time, we mix neural predictions with bigram statistics:

```
P_mixed = (1 - alpha) * P_neural + alpha * P_bigram
```

Where alpha is **conditioned on document quality**:
- High quality (>0.6): alpha_base = 0.15 (trust neural more)
- Low quality: alpha_base = 0.30 (trust bigram more)
- Scaled by bigram confidence: alpha = alpha_base * min(P_bigram * 3, 1)

This is novel: no existing submission conditions the mixer alpha on data quality.

## Performance Characteristics

- **Training overhead**: ~0% — all operations are GPU-vectorized tensor ops
  - BigramStats: 2x `index_put_` (update) + 2x index (lookup) per step
  - DocumentQualityScorer: 1x `scatter_` + 1x comparison per step
  - SalienceTracker: 3 float operations per step
  - Three-tier weights: 2x comparison + 2x masked fill per step
- **Memory overhead**: 8MB (4MB bigram counts + 4MB bigram totals)
- **Eval overhead**: ~2% — one extra softmax + bigram lookup per eval batch

## Ablation Plan

Each component can be independently disabled via environment variables:

```bash
VERIFILY_ENABLED=0        # Disable all Verifily components
VERIFILY_SALIENCE=0       # Disable salience tracker only
VERIFILY_MIXER=0          # Disable eval-time bigram mixer only
VERIFILY_NGRAM_WARMUP=0   # Enable token weighting from step 0
VERIFILY_NGRAM_THRESH=0.185 # Bigram probability threshold for Tier 1 (~p95)
VERIFILY_QUALITY_THRESH=0.735 # Document quality threshold for Tier 3 (~median)
VERIFILY_PRED_WEIGHT=0.10  # Predictable tier weight (validated: soft nudge)
VERIFILY_NOISE_WEIGHT=0.70 # Noise tier weight (validated: gentle penalty)
```

## Base Architecture (Unchanged)

- 11 layers, 512d, 8 heads, 4 KV heads, 3x MLP, seq_len 2048
- XSA (Exclusive Self-Attention) on all layers
- BigramHash(3072, 112) + ValueEmbedding layers 9,10
- Parallel Muon optimizer with batched Newton-Schulz
- Full GPTQ with autoregressive self-generated calibration
- EMA (0.997) + SWA + LZMA compression
- Sliding window eval (stride 64)

## Running

```bash
# Standard 8xH100 run
torchrun --nproc_per_node=8 train_gpt.py

# Ablation: baseline without Verifily
VERIFILY_ENABLED=0 torchrun --nproc_per_node=8 train_gpt.py

# Ablation: Verifily without salience
VERIFILY_SALIENCE=0 torchrun --nproc_per_node=8 train_gpt.py

# Ablation: Verifily without eval mixer
VERIFILY_MIXER=0 torchrun --nproc_per_node=8 train_gpt.py
```
