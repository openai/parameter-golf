# Quantization

Reference: [`research/AXES_ANALYSIS.md#axis-3-quantization`](../../research/AXES_ANALYSIS.md)

*Bit-width allocation, clipping schemes (SDClip, etc.), GPTQ/Hessian-aware methods, QAT, mixed precision, Hadamard rotation, post-GPTQ refinement.*

## Hypothesis

The current merged SOTA (PR-1019, 1.11473 BPB) uses **sensitivity-unaware** quantization: every MLP and attention weight gets identical int6 treatment regardless of layer index, head type, or empirically-measured sensitivity. Measured data from PR-1420 shows loop layers are 2.2× more sensitive to quantization and single matrices (v_proj) can be ~80× more sensitive per byte than the least sensitive matrix in the same model. Exploiting this gap with a sensitivity-driven bit budget is probably the largest unexploited lever.

Secondary hypothesis: the uniform symmetric int grid itself is wasteful for Gaussian-distributed weights. Non-uniform grids (NF4 / MXFP4 / Lloyd-Max) or per-group scaling compose cleanly with the primary hypothesis and could compound.

---

## What PR-1019 actually does

Source: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`.

### Storage reality

- The terms **int6** and **int8** in PR-1019 refer to *value range*, not physical bit width. Both formats are stored in `torch.int8` tensors (1 byte per weight).
- int6 = values in `[-31, 31]` (6 bits of info); top 2 bits are redundant and wasted in memory.
- There is **no bit-packing step** in the quantization code. The final compression from "int8 container with int6 values" → "actually ~6 bits per weight on disk" comes entirely from `lzma.compress(..., preset=9)` (line 2045). LZMA exploits the structural redundancy.

### Tier assignment

`mixed_quantize_int6()` (line 1493), called at line 1994 with `int6_cats={"mlp", "attn"}`:

| Tensor class | Classifier rule | Format | Notes |
|---|---|---|---|
| Non-float OR ≤ 65,536 numel | all small | fp16 passthrough | scales, LN gains, small aux |
| Matches `CONTROL_TENSOR_NAME_PATTERNS` | control params | fp32 passthrough | sensitive scalars |
| 2D AND `_classify_param == "mlp"` | MLP fc, proj | int6 + full-Hessian GPTQ | |
| 2D AND `_classify_param == "attn"` | Q, K, V, out-proj | int6 + full-Hessian GPTQ | |
| Everything else 2D-and-big | `tok_emb`, BigramHash, VE128 | int8 (per-row percentile) | **name-based fallback, not sensitivity-based** |

### int6 GPTQ path (line 1171)

- Symmetric clip `[-31, 31]` (wastes the `-32` slot that 6 bits could hold)
- **Per-row fp16 scale** (no per-group, no per-column)
- Clip search over 5 percentiles `{0.999, 0.9995, 0.9999, 0.99999, 1.0}` — argmin MSE
- Full-Hessian Cholesky block-wise error compensation, `block_size=128`
- AR-self-generated calibration: 64 sequences × 2048 tokens, temp=0.8, fixed seed

### int8 fallback path (line 389)

- Per-row percentile clip (99.99984%)
- fp16 scale
- **No Hessian compensation** — just plain percentile clip + round
- Applied to all non-MLP/non-attention big tensors (embeddings, BigramHash, etc.)

---

## Why it's lacking (ordered by impact)

### 1. Sensitivity is computed but never used for bit allocation

- `_classify_param` (line 1142) only returns coarse categories (`"embed" | "mlp" | "attn" | "other"`), used for a single boolean int6-vs-int8 branch. Name-based, not measurement-based.
- The Hessian IS computed per-tensor (line 1104), but used only for intra-tensor error compensation. **There is no cross-tensor bit-budget optimization.**
- No layer-index awareness: loop layers (typically blocks 4, 5) get identical treatment to non-loop layers despite 2.2× error amplification per PR-1420.
- No head-type awareness: attention V is ~80× more sensitive per byte than K in the same layer (PR-1420), but all four projections get int6.
- No layer-depth awareness: late-layer MLP `down` matrices are empirically more sensitive in most LM literature; not exploited.

### 2. The int6/int8 split is string-matched, not measured

- Embeddings and BigramHash fall to int8 purely because their names don't match `"mlp"` or `"attn"`. **Nobody has tested whether these survive int6.**
- If they tolerate int6, that frees ~0.5 MB pre-LZMA → ~150 KB post-LZMA of artifact budget, funding wider BigramHash / more layers / extra width.

### 3. Uniform symmetric grid is suboptimal for Gaussian weights

- Weights are approximately Gaussian. Uniform int6 gives the tails (|w| > 2σ) the same resolution as the bulk (|w| < σ), despite the bulk holding 68% of mass.
- Non-uniform grids (NF4, Lloyd-Max, learned codebook) match density. Published transformer-weight results suggest ~0.3-0.5 bit equivalent savings.
- **Untried in parameter-golf.**

### 4. No per-group scaling

- One fp16 scale per output row (512 or 1536 elements per row). A single outlier column forces the whole row's scale up, degrading resolution across the rest of the row.
- Per-group-128 (8 groups per 1024-col row) typically saves ~0.005-0.01 BPB at similar bit budget, at the cost of ~1.25× scale storage.
- **Untried.**

### 5. Coarse clip search

- Grid of 5 discrete percentiles `{0.999, 0.9995, 0.9999, 0.99999, 1.0}`.
- PR-1394 introduced SDClip (σ-based clipping with k parameter, `k=12.85` default). Per-layer k-scheduling (tighter k for sensitive, looser for tolerant) was never tried.

### 6. `-32` slot wasted

- Symmetric `[-31, 31]` = 63 values, fits in 6 bits (64-value range). Asymmetric `[-32, 31]` would use the full range. Trivial, ~1 extra level. Marginal but free.

### 7. No outlier-aware splitting

- SmoothQuant / AWQ-style equivalent weight-activation scaling is standard in production quantization. Not attempted.

### 8. AR self-gen calibration may mis-match sensitivity

- The Hessian quality depends on calibration-distribution match. Self-generated data at temp=0.8 after training concentrates on high-probability tokens; rare events where quantization errors compound are underrepresented. No ablation against alternative legal calibration sources.

### 9. No post-GPTQ refinement

- GPTQ minimizes MSE against the calibration Hessian. BPB is not MSE-equivalent. EGGROLL-style coordinate descent on integer bins using actual val loss (PR-1156) catches this discrepancy. Not in PR-1019 stack.

### 10. Hadamard rotation not used

- Claimed 68× MSE reduction (PR-1400, under review). Reduces outliers by rotating the weight basis. Composable.

---

## Proposed experiments

All post-training ops on saved weights. Workflow: reproduce PR-1019 once, save EMA weights + AR-calibration Hessians + template state dict, then iterate quantization offline on 2×H200.

| # | Experiment | Expected Δ BPB | Effort | Dependencies |
|---|---|---|---|---|
| Q1 | Measure Hessian trace per-tensor; produce sensitivity ranking (no BPB experiment yet) | n/a (info) | Low | saved weights + Hessians |
| Q2 | Embeddings + BigramHash int8 → int6 downgrade | likely 0; frees budget for wider BigramHash | Trivial | saved weights |
| Q3 | v_proj int8 protection for sensitive blocks (e.g. blocks 4, 5 only) | −0.001 to −0.003 | Trivial | saved weights |
| Q4 | Loop-layer bit bump (int7 for looped layers; int6 elsewhere) | −0.002 to −0.005 | Low | saved weights |
| Q5 | Sensitivity-based bit-budget reallocation (top-5 sensitive → int7, bottom-10 → int5, rest → int6, subject to total bytes) | −0.003 to −0.008 | Medium | Q1 |
| Q6 | Per-group-128 scaling (replace per-row with 128-group per-tensor) | −0.003 to −0.006 | Low | saved weights |
| Q7 | NF4 / non-uniform grid for big linears | −0.003 to −0.008 | Medium | saved weights |
| Q8 | SDClip k-scheduling per-layer | −0.001 to −0.003 | Low | saved weights + per-layer sensitivity |
| Q9 | Post-GPTQ EGGROLL (60s coordinate descent on bins using val loss) | −0.001 to −0.003 | Low-med | best single-axis result from Q3-Q7 |
| Q10 | Hadamard rotation pre-GPTQ | −0.002 to −0.010 | Medium | saved weights |
| Q11 | Asymmetric int6 `[-32, 31]` (free slot) | −0.0001 to −0.001 | Trivial | saved weights |

### First-batch sequence

**Phase 1 — reconnaissance** (cheap, informational):
- Reproduce PR-1019 on 2×H200 (~40 min), save EMA weights + Hessians + un-quantized reference eval
- **Q1**: compute per-tensor Hessian trace; rank-order sensitivity; plot
- **Q2**: re-quantize with embeddings + BigramHash set to int6; measure BPB and artifact size
- **Q11**: asymmetric int6 re-run (free win check)

**Phase 2 — directional bets** (pick 1-2 based on Q1 results):
- If Phase 1 shows strong sensitivity gradient across blocks: **Q4** (loop bump) or **Q3** (v_proj protect)
- If Phase 1 shows strong within-tensor outliers: **Q6** (per-group-128) — more of a free-win than a sensitivity bet
- If Phase 1 shows strong within-tensor non-Gaussian shape: **Q7** (NF4)

**Phase 3 — stacking** (after best single-axis result):
- **Q8** (SDClip k-scheduling) on top of Phase 2 winner
- **Q9** (EGGROLL) as the final polish pass

---

## Experiments

| ID | Date | Branch | Config | val_bpb | Base | Notes |
|----|------|--------|--------|---------|------|-------|

## Findings

- PR-1019's "int6" is stored in int8 containers; LZMA does the effective bit-packing at serialization time.
- Embeddings / BigramHash / VE128 go int8 only because their names don't match the string filter; sensitivity is not measured.
- Hessians ARE computed but only used for intra-tensor MSE minimization, never for cross-tensor bit allocation.
- PR-1420's sensitivity analysis (2.2× loop, 80× v_proj per byte) is a strong prior but was never exploited in the frontier submission lineage.

## Next

- Reproduce PR-1019 on 2×H200 and save the iteration-ready artifact bundle (EMA weights, Hessians, template SD, un-quantized eval reference).
- Phase 1 experiments above (Q1, Q2, Q11) before committing to a directional bet.

## References

- **PR-1019** (merged SOTA, 1.11473): `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
- PR-1420 mechanistic sensitivity analysis (2.2× loop, 80× v_proj): `research/prs_archive/PR-1420/`
- PR-1394 SDClip introduction: `research/prs_archive/PR-1394/`
- PR-1400 Hadamard rotation: `research/prs_archive/PR-1400/`
- PR-1418 Hadamard + int4 (poor implementation, 1.16× gain): `research/prs_archive/PR-1418/`
- PR-1426 int4 bit-packing attempt: `research/prs_archive/PR-1426/`
- PR-1156 EGGROLL post-GPTQ refinement: `research/prs_archive/PR-1156/`
- PR-692 CROWN-Q quant-variance penalty: `research/prs_archive/PR-692/`
- PR-1272 comprehensive negative results: `research/prs_archive/PR-1272/`
