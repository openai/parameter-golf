# Quantization

Reference: [`research/AXES_ANALYSIS.md#axis-3-quantization`](../../research/AXES_ANALYSIS.md)

*Bit-width allocation, clipping schemes (SDClip, etc.), GPTQ/Hessian-aware methods, QAT, mixed precision, Hadamard rotation, post-GPTQ refinement.*

## Hypothesis

The current merged SOTA (**PR-1493, 1.0810 BPB**) uses **sensitivity-unaware** quantization: the entire model is split into only two bit tiers based on a single string match (`"tok_emb"` vs everything else). All MLP, attention, BigramHash, VE, and auxiliary matrices get identical int6 treatment regardless of layer index, head type, loop-recurrence exposure, or empirically-measured sensitivity. Measured data from PR-1420 shows loop layers are 2.2× more sensitive to quantization and individual matrices (V-projection) can be ~80× more sensitive per byte than the least sensitive matrix. Exploiting this with a sensitivity-driven bit budget is the largest unexploited lever.

Secondary hypothesis: the symmetric uniform grid itself is suboptimal for Gaussian-distributed weights. Non-uniform grids (NF4 / MXFP4 / Lloyd-Max) or per-group scaling compose cleanly with the primary hypothesis and could compound.

---

## What PR-1493 actually does

Source (decompressed): `research/sota_analysis/records_normalized/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT.py`.

Note: the shipped `records/.../train_gpt.py` is lzma-compressed-code packaged; the logic is inside the `exec(L.decompress(B.b85decode(...)))` blob.

### Lineage

PR-1493 stacks on PR-1477 on PR-1413 on PR-1394. The quantization machinery was introduced in **PR-1394** (clarkkev's SP8192 + SDClip) and has been inherited unchanged through the chain. So the analysis below applies equally to the current top-5 merged records.

### Storage reality

- "int6" and "int8" refer to *value range*, not physical bit width. Both are stored in `torch.int8` tensors (1 byte per weight).
- int6 → values in `[-31, 31]` (top 2 bits redundant, wasted in memory)
- int8 → values in `[-127, 127]` (full byte used)
- **No bit-packing step in the quantization code.** The effective shrink to ~6 bits/weight on disk comes from **byte-shuffle + Brotli-11** applied to the serialized state dict. Brotli exploits the int8-containing-int6 redundancy (top bits always zero/sign-extended, high autocorrelation).

### Tier assignment

`gptq_mixed_quantize` (line 682), dispatching per tensor in order:

| Tensor class | Rule | Format | Storage |
|---|---|---|---|
| Non-float OR ≤ 65,536 numel | small/discrete | fp16 passthrough | small tensors |
| `"tok_emb"` in name | token embedding | **int8 via SDClip (k=20.0)** + full-Hessian GPTQ | ~1 MB |
| Everything else big-and-float | MLP, attn Q/K/V/O, BigramHash, VE, aux matrices | **int6 via SDClip (k=12.85)** + full-Hessian GPTQ | ~13 MB |

That's it. Two branches, one name match.

### GPTQ + SDClip details

`gptq_quantize_weight` (line 645):
```python
row_std  = W.std(dim=1)
scale    = clip_sigmas * row_std / clip_range   # clip_range = 2^(bits-1) - 1
q_col    = round(w_col / scale).clamp(-clip_range, clip_range)
```

- **SDClip replaces PR-1019's 5-point percentile grid.** Clip is set to `k × row_std` directly. Principled rate-distortion, zero search.
- Per-row `fp16` scale (no per-group, no per-column)
- Symmetric range: `[-31, 31]` for int6, `[-127, 127]` for int8. Wastes the `-32` / `-128` slots.
- Full-Hessian Cholesky block-wise error compensation, `block_size=128`
- Column reordering by descending `diag(H)`, damping of `0.01 * mean(diag(H))`
- Calibration: 64 batches from the train loader (`collect_hessians`, line 601)

### Compression pipeline

- `_byte_shuffle` stride-2 transposes bytes to improve autocorrelation (line 727)
- `brotli.compress(data, quality=11)` (line 762)
- Replaces PR-1019's LZMA-9. Brotli-11 is 3-5% denser on this payload shape in the archive's measurements.

### What changed vs PR-1019 (the previous widely-analyzed base)

| Component | PR-1019 | PR-1493 |
|---|---|---|
| Clip search | 5-point percentile grid, argmin MSE | SDClip `k × std` single formula |
| Tier assignment | 3-branch: `{mlp, attn} → int6`, `{embed, other} → int8` | 2-branch: `tok_emb → int8`, `everything else → int6` |
| BigramHash / VE | int8 (fallback) | **int6** (caught by "else" branch) |
| Compression | LZMA preset=9 | byte-shuffle + Brotli-11 |
| Selective ±1 pruning | yes (fit to 15.9 MB) | none needed (fits natively) |

---

## Why it's lacking (ordered by impact)

### 1. Sensitivity is computed but never used for bit allocation

- `classify_param` (line 479) still exists but is effectively dead code — the dispatcher at line 691 only checks `'tok_emb' in name`. No other categories influence bit allocation.
- The Hessian IS computed per-tensor (line 601) but used only for intra-tensor error compensation within GPTQ. **No cross-tensor bit-budget optimization.**
- No layer-index awareness: loop layers (3,4,5 in PR-1493) get identical treatment to non-loop layers despite 2.2× error amplification measured in PR-1420.
- No head-type awareness: attention V is ~80× more sensitive per byte than K (PR-1420), but all four projections get int6 at k=12.85.
- No layer-depth awareness: late-layer MLP `down` matrices are empirically more sensitive in most LM literature; not exploited.
- No per-matrix `k` scheduling: SDClip takes a single global `k=12.85` for every non-embed matrix.

### 2. The bit split is string-matched on `tok_emb` only

- `lm_head`, BigramHash, VE, skip-scaffold aux matrices all land in the int6 branch. No one has measured whether any of them is *more* sensitive than `tok_emb` and deserves int8.
- Conversely, nobody has tested whether `tok_emb` can tolerate int6 with a higher `k`. If it can, that frees ~150 KB artifact budget post-Brotli.

### 3. Uniform symmetric grid is suboptimal for Gaussian weights

- Weights are approximately Gaussian. Uniform int6 gives the tails (|w| > 2σ) the same resolution as the bulk (|w| < σ), despite the bulk holding 68% of mass.
- Non-uniform grids (NF4, Lloyd-Max, learned codebook) match density. Published transformer-weight results suggest ~0.3-0.5 effective bit savings.
- **Untried on parameter-golf.**

### 4. No per-group scaling

- One fp16 scale per output row (512-4096 elements depending on tensor). A single outlier column forces the whole row's scale up, degrading resolution across the rest.
- Per-group-128 (multiple groups per row) typically saves ~0.005-0.01 BPB at similar bit budget, at the cost of ~1.25× scale storage.
- **Untried.**

### 5. Single global `k` for all matrices

- `k=12.85` is a single tuned constant across all non-embed tensors. Tighter `k` for sensitive, looser for tolerant, would allocate the same total int6 budget more efficiently. PR-1412 ("Hessian-Aware SDClip") gestures at this but does not do per-tensor k-scheduling.

### 6. Wasted `-32` / `-128` slots

- Symmetric `[-31, 31]` = 63 values, fits in 6 bits (64-value range). `[-32, 31]` asymmetric uses the full range. Marginal but free. Same for int8 at `-128`.

### 7. No outlier-aware splitting

- SmoothQuant / AWQ-style equivalent weight-activation rescaling is standard in production quantization, especially for activation outliers. Not attempted.

### 8. No Hadamard rotation

- PR-1400 attempted it (claimed 68× MSE reduction, but under compliance review). PR-1418 attempted it poorly and saw 1.16× only. Would reduce weight outliers by rotating into a denser basis. Composable with everything else.

### 9. Train-time calibration (train loader) is legally risky

- `collect_hessians` reads 64 batches from the train loader. That's within the 600s training budget so it's legal, but several predecessor PRs (#535, #569, #593, #609) were ruled illegal for calibrating on training data outside the budget. Different legal definition, same concern. PR-1019 solved it by AR-self-generating calibration; PR-1493 reverted to train-loader calibration inside the budget. Worth verifying our reproduction matches the legal framing used by the reviewers.

### 10. No post-GPTQ refinement

- GPTQ minimizes MSE against the calibration Hessian. BPB is not MSE-equivalent. EGGROLL-style coordinate descent on integer bins using actual val loss (PR-1156) catches this discrepancy. Not in any of the top-5 merged records.

### 11. Brotli already exploits int6-in-int8, but the scale table isn't encoded efficiently

- The per-row fp16 scales are stored as raw bytes. These have a specific distribution (clip_sigmas × row_std) and could be compressed separately with a better entropy coder. Tiny potential gain (~a few KB), but trivial if implemented.

---

## Proposed experiments

All post-training ops on saved weights. Workflow: reproduce PR-1493 once, save EMA weights + calibration Hessians + template state dict + un-quantized reference eval, then iterate quantization offline on 2×H200.

| # | Experiment | Expected Δ BPB | Effort | Dependencies |
|---|---|---|---|---|
| Q1 | Measure Hessian trace + activation variance per-tensor; produce sensitivity ranking (informational, no BPB experiment yet) | n/a | Low | saved weights + Hessians |
| Q2 | Test whether `tok_emb` can tolerate int6 at higher `k` (e.g. k=20 at int6 vs k=20 at int8 — free budget if viable) | 0 to +0.001 BPB (neutral); frees ~150 KB budget | Trivial | saved weights |
| Q3 | Per-tensor `k` scheduling (sensitive matrices → lower k; tolerant → higher k, subject to total bit budget) | −0.001 to −0.003 | Low | Q1 |
| Q4 | Loop-layer bit bump (int7 for layers 3-5; int6 elsewhere, preserve total bytes) | −0.002 to −0.005 | Low | saved weights |
| Q5 | V-projection int8 protection (blocks that are most sensitive per Q1) | −0.001 to −0.003 | Trivial | Q1 |
| Q6 | Sensitivity-based bit-budget reallocation (top-5 → int7, bottom-10 → int5, rest → int6, subject to total bytes) | −0.003 to −0.008 | Medium | Q1 |
| Q7 | Per-group-128 scaling (replace per-row with per-group-128) | −0.003 to −0.006 | Low | saved weights |
| Q8 | NF4 / non-uniform grid for big linears | −0.003 to −0.008 | Medium | saved weights |
| Q9 | Hadamard rotation pre-GPTQ | −0.002 to −0.010 | Medium | saved weights |
| Q10 | Post-GPTQ EGGROLL (coordinate descent on bins using val loss) | −0.001 to −0.003 | Low-med | best single-axis result |
| Q11 | Asymmetric int6 `[-32, 31]` (use the wasted slot) | −0.0001 to −0.001 | Trivial | saved weights |

### First-batch sequence

**Phase 1 — reconnaissance** (cheap, informational):
- Reproduce PR-1493 on 2×H200 (~40 min), save EMA weights + Hessians + un-quantized reference eval
- **Q1**: compute per-tensor Hessian trace, activation variance, and end-to-end perturbation sensitivity. Rank-order. Plot.
- **Q11**: asymmetric int6 re-run (free win check)
- **Q2**: tok_emb downgrade probe (free-budget check)

**Phase 2 — directional bets** (pick 1-2 based on Q1):
- If sensitivity gradient is strong across blocks: **Q4** (loop bump) or **Q5** (v_proj protect) or **Q6** (full reallocation)
- If within-tensor outliers dominate: **Q7** (per-group-128)
- If the weight distribution shape itself is the issue: **Q8** (NF4)

**Phase 3 — stacking** (after best Phase-2 result):
- **Q3** (per-tensor k scheduling) on top of Phase 2 winner
- **Q10** (EGGROLL) as final polish

---

## Experiments

### Infrastructure & Baseline

| ID | Date | Config | val_bpb | Artifact | Notes |
|----|------|--------|---------|----------|-------|
| `pr1493_bundle_seed42` | 04-16 | PR-1493 defaults, seed=42, 8×H100 Modal | **1.0872** pre-quant | — | Ceiling reference. Bundle on HF. |
| `pr1493_quantize_reference_v2` | 04-16 | Baseline GPTQ: int6 k=12.85, int8 tok_emb k=20.0 | pre **1.0872** / post **1.0984** | **15.97 MB** | Reference. Quant gap = +0.011. Sliding 1.0818 matches PR-1493's 1.0827. |

### k (clip range) tuning — controls bin width

All on baseline bundle. Wider k = more near-zero values round to 0 = better Brotli compression but coarser bins.

| ID | k | post-quant BPP | BPP cost | Artifact | Notes |
|----|---|----------------|----------|----------|-------|
| E1_k4 | 4.0 | 1.0914 | +0.004 | 22.58 MB | Best quality, terrible compression. Tighter k = higher entropy ints = Brotli can't compress. |
| E1_k6 | 6.0 | 1.0911 | +0.004 | 20.23 MB | Same pattern. |
| E1_k8 | 8.0 | 1.0923 | +0.005 | 18.64 MB | Same pattern. |
| baseline | **12.85** | **1.0984** | **+0.011** | **15.97 MB** | Current PR-1493. Tuned for Brotli, not MSE. |
| E19_k19 | **19.3** | **1.1124** | **+0.025** | **13.72 MB** | Inflated scale idea. **Best compression lever: 2.25 MB freed.** 63 levels preserved for outliers. |
| E20_k25_sparse | 25.0+sparse | 1.1383 | +0.051 | 12.29 MB | Too aggressive. 3.7 MB freed but +0.051 cost. |

**Key insight**: k=12.85 is NOT tuned for MSE — it's tuned for Brotli. 70.6% of quantized values are {-2,-1,0,1,2}. Tighter k spreads values across full [-31,31] range → higher entropy → worse compression. The rate-distortion tradeoff is dominated by Brotli's compression efficiency, not by quantization error.

### Bit width changes

| ID | Config | post-quant BPP | BPP cost | Artifact | Notes |
|----|--------|----------------|----------|----------|-------|
| E3_int5_k6 | int5 k=6 | 1.0983 | +0.011 | 16.18 MB | Same quality as baseline — fewer bits + tighter k = same resolution per bin. 175 KB over cap. |
| E6_int5_k1285 | int5 k=12.85 | 1.1398 | +0.053 | 12.02 MB | Best raw compression (33.8% zeros). Quality collapse — coarse 0.83σ bins. |
| E4_nf5 | NF5 | 1.1112 | +0.024 | 22.43 MB | Worse on both axes. Max-entropy levels kill Brotli. Heavy-tailed tensors (blocks 7-10) don't match Gaussian assumption. |
| E2_embed_int6 | tok_emb int6 | 1.1222 | +0.035 | 14.94 MB | tok_emb genuinely needs int8. Saves 1 MB but costs +0.035 BPP. |

**Key insight**: int5 k=6 ≈ int6 k=12.85 because bin width is nearly identical (0.39σ vs 0.41σ). The model is effectively a ~3.56 bit/value model after Brotli regardless of container format.

### SparseGPT (joint zero-or-quantize in GPTQ sweep)

Per-element decision during Hessian column sweep: zero if cost < threshold × quantize cost. Error compensated via same Cholesky framework.

| ID | Threshold | Sparsity | BPP cost | Artifact | Notes |
|----|-----------|----------|----------|----------|-------|
| E16_sparse_t1 | 1.0 | 0% | 0 | 15.97 MB | Zeroing is never cheaper than quantizing at t=1. |
| E17_sparse_t2 | 2.0 | ~20% | +0.0004 | 15.86 MB | Near-free sparsity but negligible artifact savings. |
| E17_sparse_t5 | 5.0 | ~23% | +0.002 | 15.73 MB | 0.24 MB saved. |
| E17_sparse_t10 | 10.0 | ~27% | +0.008 | 15.59 MB | 0.38 MB saved. Diminishing returns. |
| E20_k19_sparse | k=19.3 + t=2 | ~29% | +0.027 | 13.65 MB | Barely better than k=19.3 alone (13.72). Wider bins already push near-zero to 0. |

**Key insight**: SparseGPT finds ~20% safely-zeroable weights but Brotli already compresses near-zero ints well — so actual zeros don't save much more than small integers. The inflated-k approach (k=19.3) is far more effective for compression.

### Per-tensor allocation

| ID | Config | BPP cost | Artifact | Notes |
|----|--------|----------|----------|-------|
| E5_tiered_v1 | NF3 for T5, int5 k=6 for T4, int6 k=12.85 for T1-T3 | +0.003 | 16.00 MB | Marginal. GPTQ compensates internally — rearranging bits barely matters. |
| E7_tiered_k | All int5, k=6 sensitive / k=12.85 tolerant | +0.016 | 14.84 MB | 1.13 MB saved. Better than uniform but k=19.3 is simpler and saves more. |

**Key insight**: GPTQ's Hessian error compensation absorbs most allocation differences. Cross-tensor bit budgeting is a weak lever because GPTQ already optimizes within each tensor.

### Post-hoc pruning (prune then GPTQ)

All at 30% pruning, int6 k=12.85.

| ID | Method | post-quant BPP | BPP cost | Artifact | Notes |
|----|--------|----------------|----------|----------|-------|
| E8_prune30 | Magnitude (|w|) | 1.1326 | +0.034 | 15.48 MB | 0.49 MB saved. No model adaptation. |
| E10_hessian30_v2 | Hessian-aware (|w|×√H_diag) | 1.1197 | +0.021 | 15.48 MB | **38% less quality loss than naive magnitude.** Same artifact. |
| E12_random30 | Random | 2.9826 | +1.884 | 14.95 MB | Destroyed model. Which weights you prune matters enormously. |
| E11_large30 | Prune largest | 3.1902 | +2.092 | 14.94 MB | Worse than random. Large weights are critical — they fought WD to stay large. |

At 50%: Hessian-aware = 1.2816 (+0.183), magnitude = 1.3986 (+0.300). Both catastrophic at this model size.

**Key insight**: at 36M params, there's minimal redundancy for post-hoc pruning. The Hessian-aware criterion is strictly better than magnitude, but even optimal pruning can't find much safe sparsity without training-time adaptation.

### Training-time experiments

Require retraining ($5 per run on 8×H100). Bundles evaluated on Modal 1×H100.

| ID | Config | pre-quant BPP | post-quant BPP | Artifact | Notes |
|----|--------|---------------|----------------|----------|-------|
| E14_wd015 | MUON_WD=0.15 (fixed) | 1.0890 | 1.0989 | 15.97 MB | Quant gap -11% but **artifact unchanged.** WD is scale-invariant under per-row SDClip: Q(γW)=Q(W). |
| L1 fixed 0.0001 | Proximal L1 during warmdown | 1.3651 | — | — | **Crashed at step 4500.** Cumulative L1 over 2500 steps overwhelmed the model as LR decayed. |
| L1 LR-scaled 0.0001 | L1 threshold × lr_scale | 1.3780 | — | — | **Same crash.** LR scaling didn't help — cumulative damage still too high. |
| L1=0.00001 + floor 2% + WD ramp→0.3 | Combined | 1.0943 | ~15.9 MB (est) | — | Stable but +0.007 BPP. WD ramp too aggressive. L1 too weak to create sparsity. |
| Pruning during training (30%) | Hessian-aware gradual pruning | 1.1029 | 1.1111 | 15.46 MB | +0.013 BPP, 0.51 MB saved. EMA enforcement was essentially post-hoc. |
| WD taper (PR-1729) | WD 0.095→0.048 in last 30% | **1.0868** | **1.0982** | **15.97 MB** | **-0.0002 BPP vs baseline.** Quality improves by reducing late-training friction. Artifact unchanged (scale-invariant). |
| Cautious WD | CAUTIOUS_WD=1 (Muon sign-agreement masking) | **pending** | pending | pending | Non-proportional shrinkage — should break scale-invariance. Running now. |

**Key insight (scale invariance)**: L2 weight decay CANNOT help artifact size because SDClip normalizes per-row: `scale = k × row_std / clip_range`. Shrinking all weights proportionally shrinks the scale too — quantized integers are unchanged. Only methods that change the distribution SHAPE (not just scale) can help compression. L1 changes shape but is unstable. Cautious WD may change shape safely via selective gradient masking.

**Key insight (WD taper)**: WD should be tuned for MODEL QUALITY, not compression. Tapering WD late reduces optimizer friction → better pre-quant BPP. Compression is handled post-hoc.

## Findings

### PR-1493 quantization analysis
- Bit split: `tok_emb → int8`, everything else → int6. No per-tensor differentiation.
- SDClip (`k × row_std`) with global k=12.85. Hessians used only intra-tensor, never cross-tensor.
- `classify_param` exists but is dead code for bit allocation.

### Compression is already near-optimal
- Shannon entropy = 3.327 bits/value. Brotli achieves **101.3% of Shannon** (13.37 MB for int6 values vs 13.19 MB theoretical).
- Huffman coding gives identical results to Brotli on these values. No entropy coder switch helps.
- The 15.97 MB artifact includes tok_emb (~1 MB), scales (~0.1 MB), and torch.save overhead (~1 MB) beyond the 13.4 MB of compressed int6 values.

### Weight distribution
- 70.6% of quantized int6 values are in {-2,-1,0,1,2}. 16.8% exactly 0.
- Weights are approximately Gaussian (kurtosis 0.05-0.3 for most tensors). Late decoder attention (blocks 7-10) is heavier-tailed (kurtosis 1.4-2.3).
- At k=12.85, only 4 out of 31.7M values get clipped (0.00001%). The wide k "wastes" bins on tails but Brotli loves the peaked integer distribution.

### Sensitivity analysis
- 951× Hessian sensitivity range across tensors (blocks.0.mlp.proj most sensitive, blocks.10.attn.proj least).
- MLP down-projections dominate sensitivity, not attention v_proj as PR-1420 suggested.
- Loop layers (3-5) are ~1.5× more sensitive with loop correction, but layer-0 effect dwarfs the loop effect.
- Negative magnitude-sensitivity correlation in ~half the tensors: small weights can be in important columns.
- Large weights are definitively critical: pruning them is worse than random pruning.

### Scale invariance (the core constraint)
- **L2 WD is scale-invariant under per-row SDClip**: `Q(γW) = Q(W)`. Proportional weight shrinkage doesn't change quantized integers or artifact size.
- WD=0.15 weights are 29.4% smaller in RMS but artifact is identical (15.97 MB). The per-row scale absorbs the shrinkage.
- Only methods that change distribution SHAPE (not scale) can help compression: L1, pruning, Cautious WD.
- L1 changes shape but is cumulatively unstable at this model size. Cautious WD may change shape safely.

### GPTQ compensates for almost everything
- Cross-tensor bit allocation, per-tensor k scheduling, sensitivity-based reallocation — GPTQ's Cholesky error propagation absorbs the impact. Rearranging the same total bits barely moves BPP.
- The rare outlier values (3.4% outside {-2,-1,0,1,2}) are load-bearing: GPTQ places them deliberately in sensitive columns.

### Best post-training levers
1. **k inflation (k=19.3)**: 2.25 MB freed at +0.014 BPP. Widens bins, pushes more near-zero values to 0, preserves 63 levels for outlier capacity. Best single compression lever.
2. **SparseGPT**: finds ~20% safely-zeroable weights at near-zero cost, but artifact savings are negligible (Brotli already compresses near-zero ints).
3. **WD taper (PR-1729)**: -0.0002 BPP quality improvement by reducing WD friction late in training. Doesn't affect artifact (scale-invariant). Should be tuned for model quality, not compression.

## Next

- **Cautious WD** (running): gradient-momentum sign-agreement masking in Muon. Non-proportional shrinkage that may break scale-invariance. Training on Modal 8×H100.
- **Cautious WD + k=19.3**: if Cautious WD produces a peakier distribution, combining with inflated k could give both quality and compression gains.
- **WD taper + k=19.3**: combine quality lever (taper) with compression lever (inflated k).

## References

- **PR-1493** (merged SOTA, 1.0810): `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/` (on `upstream/main`); decompressed source at `research/sota_analysis/records_normalized/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT.py`
- **PR-1394** (SDClip origin, 1.0856): `research/prs_archive/PR-1394/`
- PR-1412 Hessian-aware SDClip (1.0835): `research/prs_archive/` (verify path)
- PR-1413 QK-Gain 5.0 + legal TTT (1.0828): `research/prs_archive/PR-1413/`
- PR-1477 parallel residuals + TTT (1.0822): check upstream `records/`
- PR-1019 AR self-gen calibration (1.1147, previous SOTA): `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
- PR-1420 mechanistic sensitivity analysis (2.2× loop, 80× v_proj): `research/prs_archive/PR-1420/`
- PR-1400 Hadamard rotation (under review): `research/prs_archive/PR-1400/`
- PR-1418 Hadamard + int4 (poor implementation): `research/prs_archive/PR-1418/`
- PR-1426 int4 bit-packing attempt: `research/prs_archive/PR-1426/`
- PR-1156 EGGROLL post-GPTQ refinement: `research/prs_archive/PR-1156/`
- PR-692 CROWN-Q quant-variance penalty: `research/prs_archive/PR-692/`
- PR-1272 comprehensive negative results: `research/prs_archive/PR-1272/`
