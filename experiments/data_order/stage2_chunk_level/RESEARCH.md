# Data Order — Stage 2: Chunk-Level Selection

## Previous work (Stage 1)

Stage 1 ranked entire 100M-token shards by similarity to the validation set using 8 methods. After validation, the val-trained bigram LM cross-entropy (M5) was the most reliable signal (Spearman rho=0.984). However, shard-level selection barely worked: the CE spread across 80 shards was only 0.018 bits (6.0826–6.1009), and the best-first ordering gave only -0.0010 BPB improvement on a single H100. All shards look nearly identical at the bigram level when averaged over 100M tokens.

## This stage

Instead of selecting whole shards, score every **32K-token chunk** across all training data. With 8B tokens / 32K = 250,000 chunks, the within-shard variance should be far larger than the between-shard variance we saw in Stage 1. Cherry-pick the best chunks and pack them into a custom training dataset.

## Method

1. Train bigram LM on full val set (same as Stage 1 M5)
2. Score every 32K-token chunk across all 80 shards
3. Analyze the CE distribution — within-shard vs between-shard variance
4. Select top chunks to fill training budget
5. Pack into a new .bin file
6. Train and compare against baseline

## Baselines
- **1× H100 80GB (default order)**: val_bpb = 1.3055, 1836 steps @ 327ms
- **1× H100 80GB (M5 shard-ordered)**: val_bpb = 1.3045 (-0.0010)
- **8×H100 (PR #549)**: val_bpb = 1.1194 (3-seed mean)

## Results

### Chunk scoring analysis
- 244,080 chunks scored (80 shards × ~3,051 chunks each)
- CE range: 4.22 – 9.58 (5.36 bit range vs 0.018 at shard level)
- **Within-shard variance is 535× larger than between-shard variance**
- Selected top 12% (29,456 chunks): mean CE 5.965 vs full mean 6.093 (0.128 bits lower)

### Training result — NEGATIVE

| Run | val_bpb | val_loss | Steps |
|-----|---------|----------|-------|
| Baseline (default order) | 1.3055 | 2.2043 | 1836 |
| M5 shard-ordered (Stage 1) | 1.3045 | 2.2026 | 1841 |
| **Chunk-selected (top 12%)** | **1.3127** | 2.2164 | 1841 |

**Chunk selection is 0.0072 BPB WORSE than baseline.**

Training loss was lower (2.21 vs 2.27 at step 1200), confirming the model fits the selected data better. But val performance degraded — the selected subset lacks diversity needed for generalization.

### Why it failed

Selecting the lowest-CE chunks creates a **less diverse** training set. The bigram LM favors text with common val-like bigram patterns — likely generic, repetitive, "easy" text. This is the same M5-vs-M3 issue from Stage 1: low absolute CE ≠ distinctively val-like. The model memorizes common patterns but misses the long tail of val's distribution.

The 0.128-bit CE advantage in selected chunks doesn't translate to val_bpb improvement because:
1. Diversity loss outweighs distribution matching
2. Overfitting to the "easy" subset hurts generalization to the full val set

### Neural LM chunk selection

Trained a 17M-param proxy model (same architecture as competition, 9 layers/512 dim) on val for 500 steps (val loss 3.63). Scored all 244K chunks in 91 minutes on H100.

| Metric | Bigram scorer | Neural scorer |
|--------|-------------|---------------|
| CE mean | 6.093 | 3.542 |
| CE std | 0.0991 | 0.0928 |
| Coeff of variation (std/mean) | 1.6% | 2.6% |
| Within/between shard var ratio | 535× | 439× |
| Spearman(bigram, neural)* | — | 0.64 |

*Spearman computed only on the top 12% of chunks (bigram's 1GPU selection), not all 244K. Restricting to this top slice attenuates the correlation — full-population agreement is likely higher. The "different patterns" claim may be overstated.

The neural model has 60% more relative discrimination (CoV 2.6% vs 1.6%), but raw CE gains are on incomparable scales (bigram bits ≠ neural bits). The variance ratio (scale-free) is the valid comparison.

### Full results table

| Run | val_bpb | vs baseline |
|-----|---------|-------------|
| Baseline (default order) | **1.3055** | — |
| M5 shard-ordered (Stage 1) | 1.3045 | -0.0010 |
| Bigram chunk-selected (top 12%) | 1.3127 | +0.0072 |
| Neural chunk-selected (top 12%) | 1.3112 | +0.0057 |

Both chunk selection methods are **worse than baseline**. Neural is less bad than bigram, but still harmful.

### Conclusion

**Aggressive data selection hurts.** The problem isn't the scorer (bigram vs neural gives similar results) — it's the selection strategy. Taking only the top 12% of chunks by any measure removes too much diversity. The model needs exposure to the full distribution tail to generalize well.

Note: the baseline sees ~10 contiguous shards (965M tokens from the start of 8B). The chunk-selected set draws 965M tokens cherry-picked from ALL 80 shards — a wider source pool. The diversity loss comes entirely from the filtering criterion, not from sampling fewer sources. This makes the negative result even more damning for hard selection.

Possible directions that might work:
1. **Soft reweighting** instead of hard selection — repeat val-like chunks more, don't exclude anything
2. **Diversity-constrained selection** — select chunks that match val's distribution while maintaining coverage
3. **Just don't select** — the default random ordering already provides good diversity; the competition's 10-minute wall clock is the real constraint, not data quality
