# Data Order Experiment — Research Log

## Goal
Select the training shards whose distribution best matches the validation set. On 8×H100 (38 shards consumed), order so most val-similar data is seen last (general→specialized curriculum). On fewer GPUs, put best data first since only a few shards are seen.

## Baselines
- **8×H100 (PR #549)**: val_bpb = 1.1194 (3-seed mean, std 0.0006), 7185 steps @ 83.4ms, 38 shards consumed
- **1× RTX PRO 6000 Blackwell**: val_bpb = 1.3420, 1225 steps @ 490ms, ~6.4 shards consumed

## Dataset
- **Train**: 80 shards × 100M tokens = 8B tokens (fineweb10B_sp1024)
- **Val**: 1 shard, 62M tokens
- **Vocab**: 1024 (SentencePiece BPE)
- **Local copy was incomplete** (66 shards); full dataset has 80 shards

## Methods Implemented

### M1: N-gram cosine similarity
- Computes unigram + bigram joint distributions for val and each shard
- Cosine similarity in distribution space, averaged (uni+bi)/2

### M2: Jensen-Shannon Divergence
- Symmetric KL-based divergence on unigram distributions
- Bounded [0, ln2], only unigram level (no sequential structure)

### M3: Moore-Lewis cross-entropy difference
- Train bigram LM on val, train bigram LM on all-train aggregate
- Score = CE_general(shard) - CE_val(shard)
- Higher = shard is *distinctively* val-like (not just generically easy)
- **Bug found & fixed**: originally used 3 specific shards for general LM, biasing those shards. Fixed to use all-train aggregate in both analyze_shards.py and validate_methods.py.

### M4: Domain classifier (DROPPED)
- Logistic regression on unigram features, 51.6% accuracy — useless

### M5: Val-trained LM perplexity
- Train bigram LM on full 62M val tokens
- Measure cross-entropy of each shard under this model (using full shard bigram counts)
- Lower CE = shard's bigram patterns are more predictable by val model
- **Validated as most reliable method** (Spearman rho = 0.984)

### M6: Conditional bigram embedding cosine (DROPPED)
- Validated as noise (Spearman rho = -0.076)

### M7: Wasserstein distance (DROPPED)
- Methodologically flawed: W1 on CDFs requires ordered support; BPE token IDs are unordered categoricals

### M8: Importance weighting (density ratio)
- E_shard[log2(p_val(next|prev) / p_train(next|prev))]
- Weighted average over shard's bigram distribution
- Higher = shard's conditional patterns are overrepresented in val vs general train
- Uses all-train aggregate for denominator

## Validation Methodology

### First attempt (BUGGY)
- Split val by random token permutation: `val_A = val_tokens[perm[:half]]`
- **Bug**: shuffling individual tokens destroys sequential structure
- Shuffled bigrams ≈ P(a)×P(b), not real conditional patterns
- All methods showed rho > 0.99 — measuring unigram stability only

### Fixed validation
- Split val into **contiguous halves** (random split point, preserving bigram structure)
- 10 random splits, Spearman correlation between val_A ranking and val_B ranking

### Validation results (contiguous splits)
| Method | Spearman rho | Std | Verdict |
|--------|-------------|-----|---------|
| **M5: Val CE** | **0.984** | 0.009 | **Best — use this** |
| M8: Importance | 0.724 | 0.141 | Decent |
| M1: N-gram cosine | 0.697 | 0.309 | OK but high variance |
| M3: Moore-Lewis* | 0.603 | 0.218 | Moderate (*pre gen-LM fix) |
| M2: JSD | 0.458 | 0.275 | Unreliable |
| M6: Embed cosine | -0.076 | 0.075 | Useless (noise) |

Note: M3 validation was run before the gen-LM bug was fixed in validate_methods.py; its true rho may differ. Now fixed.

## Inter-method correlations

Computed on 66-shard (local) data; may shift slightly on full 80-shard set.

Two anti-correlated clusters:
- **Cluster A** (marginal/frequency): M1, M2 (rho 0.82-0.91 between them)
- **Cluster B** (conditional/LM-based): M3, M8 (rho 0.83)
- **M5** anti-correlates with Cluster A (rho ≈ -0.05 to -0.19) and with M3 (rho = -0.67)

The M5-vs-M3 anticorrelation is real, not a sign error — both metrics are stored as "higher = more val-like." The disagreement is fundamental: M5 measures *absolute* predictability (low CE under val model), while M3 measures *relative* predictability (val model predicts better than general model). Shards that are generically easy text have low CE under *both* models, so M5 ranks them high but M3 ranks them low. Validation confirms M5 (absolute) is the better signal.

## Final Selection (M5-based, 80 shards)

Updated with fixed code (full-shard counts, all-train gen LM) — see "Re-run with fixed code" section below for fresh numbers.

Top 5 most val-similar (fixed): 002, 065, 011, 028, 064
Top 5 least val-similar: 045, 008, 072, 063, 000

CE range: 6.0826–6.1009 (0.018 bits — very tight, shards are nearly identical at bigram level).

## Experiment 1: M5-selected data (best val-match first, single GPU)

**Design**: On single GPU, only ~6.4 shards are consumed. Instead of testing curriculum (best-last), we tested pure **data selection**: replace the default first 7 shards (000-006) with the M5 top 7 (012, 006, 065, 004, 041, 074, 007). Best-first ordering, because we only see these shards.

Symlinked dataset: `fineweb10B_sp1024_m5_ordered/`

### Results

| Run | val_bpb | val_loss | Steps |
|-----|---------|----------|-------|
| Baseline (default order) | 1.3420 | 2.2659 | 1225 |
| M5-selected (best first) | **1.3398** | 2.2621 | 1225 |
| **Delta** | **-0.0022** | -0.0038 | — |

### Caveats
- **Single run, no error bars.** The 8×H100 baseline has std=0.0006 across 3 seeds. Single-GPU variance is unknown. -0.0022 could be noise.
- **Tests data selection, not curriculum ordering.** The "best last" curriculum strategy was not tested here.
- 7th shard (007) is only ~40% consumed before the 600s wall clock cap.

### Interpretation
This experiment shows data *selection* works in principle on a constrained budget. The curriculum question (best-first vs best-last) requires an 8×H100 run where enough shards are consumed to make ordering within the selection meaningful.

## Files
- `analyze_shards.py` — 6 methods (M1-M3, M5, M6, M8), full shard bigram counts
- `validate_methods.py` — holdout validation with contiguous splits
- `rank_by_m5.py` — final ranking by validated best method
- `create_ordered_symlinks.py` — creates symlinked dataset in M5 order
- `selected_shards.json` — M5-based selection (best-last for 8×H100), written by rank_by_m5.py
- `composite_selection.json` — composite-method selection, written by analyze_shards.py (separate file to avoid collision)
- `shard_analysis.json` — raw scores from all methods
- `shard_ranking.json` — composite ranking (5 distributional methods)
- `BASELINE.md` — PR #549 baseline numbers (val_bpb = 1.1194)
- `baseline_blackwell.log` — single-GPU baseline training log
- `m5_ordered_blackwell.log` — single-GPU M5-ordered training log

## Re-run with fixed code (H100)

Machine: vast.ai 1× H100 80GB HBM3, PyTorch 2.11.0+cu128
All fixes applied: full-shard bigram counts, all-train gen LM, no truncation.

### Updated M5 rankings

CE spread is now **much tighter**: 6.0826–6.1009 (range 0.018) vs old 6.0473–6.1630 (range 0.116).
The 2M-token truncation was adding noise; full-shard counts reveal shards are very similar at bigram level.

New M5 top 5: 002, 065, 011, 028, 064 (different from old: 012, 006, 065, 004, 041)

### Updated M3 validation (M1/M2/M5 unchanged — they don't use gen LM)

| Method | Spearman rho | Std |
|--------|-------------|-----|
| **M5: Val CE** | **0.984** | 0.009 |
| M3: Moore-Lewis | 0.724 | 0.141 |
| M8: Importance | 0.724 | 0.141 |
| M1: N-gram cosine | 0.697 | 0.309 |
| M2: JSD | 0.458 | 0.275 |

M3 improved from 0.603→0.724 after gen-LM fix. M3 and M8 now produce identical scores (same formula when using full-shard counts with all-train denominator).

### Experiment 2: H100 single GPU (fixed code)

| Run | val_bpb | val_loss | Steps | ms/step |
|-----|---------|----------|-------|---------|
| Baseline (default order) | 1.3055 | 2.2043 | 1836 | 326.8 |
| M5-ordered (best first) | **1.3045** | 2.2026 | 1841 | 325.9 |
| **Delta** | **-0.0010** | -0.0017 | +5 | — |

Smaller improvement than Experiment 1 (-0.0010 vs -0.0022). Expected: the fixed full-shard M5 shows shards are much more similar to each other (0.018 CE range), so reordering has less impact.

**Conclusion**: Shard-level selection with a bigram model has hit its ceiling. The signal is too weak — all 100M-token shards look nearly identical at the bigram level.
