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

| ID | Date | Branch | Config | val_bpb | Base | Notes |
|----|------|--------|--------|---------|------|-------|

## Findings

- PR-1493's quantization was inherited from PR-1394 with no bit-allocation changes through the lineage (PR-1394 → PR-1412 → PR-1413 → PR-1477 → PR-1493).
- The bit split is even coarser than PR-1019: just `tok_emb → int8`, everything else → int6. BigramHash and VE, which were int8 in PR-1019, now go int6.
- `classify_param` still exists in the code but is dead code for bit-allocation purposes (only `tok_emb` name-match is used).
- SDClip (`k × row_std`) is cleaner than PR-1019's percentile grid, but a single global `k=12.85` applies to every non-embed matrix.
- Hessians are computed per-tensor but used only for intra-tensor MSE compensation inside GPTQ — never for cross-tensor budget decisions.
- Brotli-11 + byte-shuffle replaces LZMA-9 for the final compression stage, doing the real "bit-packing" against the int8-containing-int6 redundancy.
- PR-1420's mechanistic sensitivity analysis (2.2× loop-layer amplification, ~80× v_proj sensitivity) is a strong prior but has never been exploited in any frontier-lineage submission.

## Next

- Reproduce PR-1493 on 2×H200 and save the iteration-ready artifact bundle (EMA weights, Hessians, template SD, un-quantized reference eval).
- Phase 1 experiments above (Q1, Q2, Q11) before committing to a directional bet.
- Verify legal framing of train-loader calibration vs AR-self-gen matches current reviewer guidance.

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
