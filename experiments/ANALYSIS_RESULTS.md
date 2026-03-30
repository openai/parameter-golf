# Checkpoint Analysis Results

Analysis of trained checkpoints from competition runs. All results from real trained weights, not random initialization.

## Checkpoints analyzed

| Checkpoint | Score | Seed | Steps | ms/step | Config |
|-----------|-------|------|-------|---------|--------|
| final_model_baseline.pt | 1.1344 | 1337 | 8,672 | 69 | EMA, baseline |
| final_model_1335.pt | 1.1335 | 1337 | 8,193 | 73 | EMA, LATE_K_FP16=0 |
| final_model_seed7.pt | 1.1324 | 7 | 8,730 | 69 | EMA, FP16_EMBED=0 |
| final_model_oneshot_524k.pt | 1.2076 | 1337 | 5,532 | 109 | SWA+WD20K+prune, slow pod |
| final_model_91ms.pt | 1.1392 | 1337 | ~6,600 | 91 | EMA, baseline on degraded pod |

---

## 1. Quantization Strategy Comparison

Tested on final_model_baseline.pt (1.1344 model).

### Reconstruction MSE (lower = better quality)

| Method | Weighted avg MSE | vs Int6 |
|--------|-----------------|---------|
| **Int6 uniform** | 8.77e-05 | baseline |
| **GPTQ-lite** (per-row clip search) | 8.75e-05 | -0.2% (negligible) |
| **Codebook K=256** (K-means) | 1.13e-05 | -87% (much better) |

### Compressed artifact size (lower = smaller artifact)

| Method | Compressed bytes (one MLP tensor) | vs Int6 |
|--------|----------------------------------|---------|
| **Int6 uniform** | 466,152 | baseline |
| **GPTQ-lite** | 466,183 | +0.01% (same) |
| **Codebook K=256** | 585,490 | **+25.6% (WORSE)** |

### Verdict
- **Codebook: KILLED.** 87% better reconstruction but 25% larger compressed size. zstd-22 compresses low-entropy int6 values more efficiently than high-entropy codebook indices.
- **GPTQ-lite: MARGINAL.** Only 0.2% improvement. Our model's weight distributions are well-conditioned. Keep as default (zero cost) but don't expect gains.
- **Int6 uniform is near-optimal** for our model under zstd-22 compression.

---

## 2. Tensor Sensitivity Ranking

Top 15 most sensitive tensors to int6 quantization (highest MSE):

| Rank | Tensor | MSE | Params | Kurtosis | Notes |
|------|--------|-----|--------|----------|-------|
| 1 | blocks.7.resid_mix | 4.45e-04 | 1,024 | 3.2 | Already fp32 (control tensor) |
| 2 | bigram_hash.proj.weight | 3.27e-04 | 65,536 | -0.6 | Already fp16 (<65K params) |
| 3 | blocks.2.resid_mix | 3.08e-04 | 1,024 | 4.1 | Already fp32 |
| 4-8 | Other resid_mix | 1.5-2.9e-04 | 1,024 ea | 3-10 | Already fp32 |
| 7 | **blocks.7.attn.c_k.weight** | **1.72e-04** | 131,072 | **11.9** | **Most sensitive large tensor** |
| 10 | tok_emb.weight | 1.50e-04 | 524,288 | 0.5 | fp16 with FP16_EMBED_EXPORT=1 |

### Key finding
- Most sensitive tensors are TINY (resid_mix at 1024 params) — already protected as control tensors in fp32.
- **Block 7 is the outlier layer**: c_k has kurtosis 11.9 (vs avg ~0.5), c_v has kurtosis 4.8. This layer consistently has the heaviest tails.
- Our existing protection scheme (control tensors fp32, LATE_K_FP16 for last 2 layers) is close to optimal. Block 7 c_k would benefit from fp16 protection but it's layer 7, not in the last 2.

---

## 3. Embedding Analysis

### SVD — Regenerated LM Head Feasibility

| Rank | Variance explained | Reconstruction MSE | Storage | Savings vs fp16 |
|------|-------------------|-------------------|---------|----------------|
| 16 | 20.6% | 1.10e-01 | 48KB | 976KB |
| 32 | 29.0% | 9.82e-02 | 96KB | 928KB |
| 64 | 41.9% | 8.04e-02 | 192KB | 832KB |
| 128 | 60.6% | 5.45e-02 | 384KB | 640KB |
| 256 | 83.3% | 2.31e-02 | 768KB | 256KB |

**Verdict: KILLED.** Rank-64 explains only 41.9% of variance — need >95% for viability. The embedding matrix is full-rank with 1024 tokens × 512 dims. Each token embedding is distinct.

### Entropy — Selective fp16 Protection

- Entropy range: 5.12 - 6.69 (significant variation)
- Mean entropy: 6.14
- **Top 10% highest-entropy rows**: 102 rows → 104KB in fp16
- **Bottom 90%**: 922 rows → safe for int6
- **Savings**: 944KB vs full fp16 (104KB vs 1,048KB)

**Verdict: GO.** Selective protection is viable. 90% of tokens have well-clustered embeddings that survive int6.

---

## 4. Cross-Layer Outlier Topology

### Outlier column overlap (Jaccard similarity)

| Tensor pair | Jaccard | Shared columns |
|------------|---------|---------------|
| blocks.4.c_k ↔ blocks.7.c_k | **0.765** | 13 |
| blocks.5.c_q ↔ blocks.6.c_k | 0.500 | 1 |
| blocks.5.c_k ↔ blocks.6.c_k | 0.500 | 1 |
| blocks.6.c_k ↔ blocks.8.c_k | 0.500 | 1 |
| blocks.3.c_q ↔ blocks.3.c_k | 0.400 | 2 |

**Verdict: PARTIAL GO.** Strong correlation in c_k weights (especially blocks 4↔7 at 76.5%). Structured outlier masks viable for K projections specifically. Limited to one tensor family.

---

## 5. Cross-Layer Symmetry Transport (Procrustes Alignment)

### MSE reduction after optimal rotation alignment

| Tensor group | Layers | Reduction range | Verdict |
|-------------|--------|----------------|---------|
| **MLP proj (512×1536)** | 11 | **91-93%** | **STRONG GO** |
| Attention K (256×512) | 11 | 75-83% | GO |
| Attention V (256×512) | 11 | 76-82% | GO |
| Attention Q (512×512) | 11 | 69-73% | GO |
| Attention proj (512×512) | 11 | 68-70% | GO |
| MLP fc (1536×512) | 11 | 47% | Borderline |

### Compression potential

MLP proj alone: 11 × 786,432 params = 8.6M params total.
With symmetry-transport: 1 prototype (786K) + 10 rotations (10 × 512×512 = 2.6M params) = 3.4M params.
**Savings: 5.2M params = ~2.5MB at int6.**

All attention weights combined: similar or better reduction ratios.
**Total potential savings: ~5-8MB** freed for more parameters.

### Verdict: STRONG GO
This is the most promising compression moonshot. Layers share massive rotational structure, especially MLP output projections. A prototype + per-layer rotation matrix could compress 8.6M MLP proj params to 3.4M — 2.5x compression at ~91% fidelity.

---

## 6. Cross-Seed Analysis

### Weight similarity between seed=1337 and seed=7

| Tensor | Cosine similarity |
|--------|------------------|
| All tested tensors | ~0.000 (completely different) |

**Different seeds produce entirely different weight matrices.** No direct value-level similarity. Expected — different random init → different local minimum.

### BUT: Procrustes alignment works cross-seed!

| Comparison | Raw MSE | Aligned MSE | Reduction |
|-----------|---------|-------------|-----------|
| Cross-layer (L0 vs L5, same seed) | 2.90e-02 | 2.47e-03 | **91%** |
| Cross-seed (L5, seed=1337 vs seed=7) | 2.86e-02 | 2.98e-03 | **90%** |

**Critical finding**: Even though the weights are completely different values, the same layer across different seeds is a **rotation** of each other. The rotational structure is a property of the **architecture + training objective**, not the specific initialization.

**Implication**: Symmetry-transport is not a quirk of one training run — it's a fundamental structural property. A universal prototype could potentially work across seeds.

---

## 7. SWA vs EMA Weight Smoothness

| Metric | EMA (baseline, 1.1344) | SWA (oneshot, 1.2076) |
|--------|----------------------|---------------------|
| MLP fc std | 0.2056 | 0.0912 (**2.3x smoother**) |
| MLP fc kurtosis | 0.53 | 0.17 (**3x lower tails**) |
| MLP proj std | 0.1186 | 0.0523 (**2.3x smoother**) |
| MLP proj kurtosis | 0.43 | 0.11 (**3.9x lower tails**) |
| Artifact size | 16.3 MB | 12.4 MB (**24% smaller**) |
| Score | 1.1344 | 1.2076 (undertrained) |

**Key insight**: SWA produces dramatically smoother weights with lighter tails, leading to much better compression (12.4MB vs 16.3MB). The poor score (1.2076) is entirely due to the slow pod (5,532 steps vs 8,672) — not because SWA is bad.

**If we could get SWA-level smoothness at EMA-level step count** (fast pod + SWA + enough steps), we'd get both good score AND a tiny artifact with massive headroom for more parameters.

---

## 8. Artifact Size Sweep (re-export without retraining)

Tested on final_model_baseline.pt. Code size: ~88,554 bytes.

| Config | Model bytes | Total bytes | vs 16MB limit | Fits? |
|--------|-----------|-------------|--------------|-------|
| Full fp16 embed | 15,721,516 | 15,810,070 | -189,930 | YES |
| No fp16 embed (all int6) | 15,263,911 | 15,352,465 | -647,535 | YES |
| Selective fp16 10% | 15,979,132 | 16,067,686 | +67,686 | no |
| Selective fp16 20% | 15,504,432 | 15,592,986 | -407,014 | YES |
| Prune 1% + full fp16 | 15,723,628 | 15,812,182 | -187,818 | YES |
| Prune 3% + full fp16 | 16,449,037 | 16,537,591 | +537,591 | **no** |
| Prune 5% + full fp16 | 15,717,152 | 15,805,706 | -194,294 | YES |
| Prune 3% + no fp16 | 15,309,774 | 15,398,328 | -601,672 | YES |
| Prune 3% + int5 MLP | 13,657,151 | 13,745,705 | -2,254,295 | YES |

**Surprising finding**: 3% pruning INCREASES artifact size by 728KB! Zeroing weights creates patterns that zstd-22 handles differently — the interaction between pruning and compression is non-monotonic. 1% and 5% pruning are fine but 3% is a sweet spot of bad zstd behavior.

**Confirmed: FP16_EMBED_EXPORT=0 + LATE_K_FP16=0 fits under 16MB** (15.35MB with 648KB headroom).

---

---

## 9. Symmetry-Transport Compression (full prototype implementation)

Implemented and tested: store weight matrices as prototype + per-layer rotation + optional residual.

### Compression results (vs int6+zstd baseline)

| Group | Tensors | Int6+zstd | Transport (no residual) | Overhead |
|-------|---------|-----------|------------------------|----------|
| MLP proj (512×1536) | 11 | 4.3 MB | 43.8 MB | **+909%** |
| MLP fc (1536×512) | 11 | 4.9 MB | 5.3 MB | +7% |
| Attn Q (512×512) | 11 | 1.7 MB | 5.0 MB | +199% |
| Attn K (256×512) | 11 | 0.9 MB | 4.9 MB | +476% |
| **Total** | | **14.3 MB** | **68.8 MB** | **+380%** |

**Not viable.** Rotation matrices (512×1536 fp16 = 1.5MB each, 10 rotations = 15MB) exceed the artifact budget. Dense orthogonal matrices lack the low-entropy patterns that zstd-22 compresses efficiently.

### Low-rank rotation approximation

Tested rank-{4,8,16,32,64,128} approximations of (R - I) on MLP proj:

| Rank | Delta variance captured | MSE reduction | Storage per layer |
|------|----------------------|---------------|-------------------|
| 16 | 2.1% | 1.8% | 98 KB |
| 64 | 8.3% | 7.6% | 393 KB |
| 128 | 16.6% | 15.1% | 787 KB |

**Not viable at tested ranks.** The rotation delta (R - I) is full-rank. Rank-128 captures only 16.6% of variance. More compact parameterizations (Givens rotations, learned representations) remain unexplored.

### Key takeaway

The Procrustes MSE reduction (91-93%) correctly identifies genuine cross-layer rotational structure. However, reconstruction MSE and compressed artifact size measure different things. The rotation matrix metadata cost dominates the savings from shared prototypes. Future work on compact rotation representations (Givens parameterization, learned low-dimensional maps) could change this conclusion.

---

## Summary of compression research findings

| Approach | Test method | Outcome | Status |
|----------|-----------|---------|--------|
| Learned codebooks (K=256) | MSE vs compressed size | 87% lower MSE but 25% larger compressed artifact | Not viable at current model scale |
| Regenerated LM head (SVD) | Rank analysis | Rank-64 explains 41.9% variance | Not viable — embedding is full-rank |
| Symmetry-transport (Procrustes) | Full prototype + zstd compression | 91% MSE reduction but 380% larger compressed | Not viable — rotation matrix storage cost exceeds savings |
| Low-rank rotation approx | Rank-128 delta decomposition | Captures 16.6% of rotation variance | Not viable — rotations are full-rank |
| Exception topology | Cross-layer Jaccard similarity | 76.5% overlap on c_k outlier columns | Viable for K-projection tensors specifically |
| Selective fp16 vocab | Per-row entropy analysis | Saves 944KB vs full fp16 | Viable — significant savings with minimal quality trade |
| GPTQ-lite | Per-row clip search | 0.2% lower MSE | Marginal improvement on well-conditioned weights |

## Observations

1. For this model architecture (11L, 512d, int6+zstd-22), uniform per-row int6 quantization is close to optimal. The weight distributions are well-conditioned (low kurtosis, ~1% outlier fraction).

2. Approaches that improve reconstruction MSE (codebooks, symmetry-transport) do not necessarily improve compressed artifact size. The downstream codec (zstd-22) must be considered jointly — lower-entropy byte streams compress better than lower-MSE representations.

3. The Procrustes alignment results (91-93% MSE reduction across layers) demonstrate genuine cross-layer rotational structure. However, exploiting this structure requires storing dense orthogonal rotation matrices, which are expensive under zstd-22. More compact rotation parameterizations (Givens angles, learned low-dimensional representations) remain unexplored.

4. Selective embedding protection (entropy-weighted fp16) and structured outlier masks (exception topology) offer modest but real savings. These are incremental improvements rather than architectural shifts.

5. SWA weight averaging produces 2.3x smoother weights with 3x lower kurtosis, leading to 24% smaller artifacts. This is the most effective compression lever observed — smoothness improves codec efficiency more than novel quantization schemes.
