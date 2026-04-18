# Initial Plan

## 1. Leaderboard Lineage (chronological)

| Date | Score | Key delta(s) | Cumulative Δ from baseline |
|------|------:|--------------|---------------------------|
| 03-17 | 1.2244 | **Naive Baseline**: 9L/512d/1024vocab, int8+zlib, non-overlapping eval | — |
| 03-18 | 1.2197 | FP16 tied embed + LR tuning | −0.005 |
| 03-18 | 1.2147 | 10L, mixed int8/int6 | −0.010 |
| 03-19 | 1.2060 | 2048 seq len (train+eval) | −0.018 |
| 03-19 | 1.2014 | 4096 seq len + hyper tuning | −0.023 |
| 03-19 | 1.1928 | LoRA TTT (doc-aware, per-chunk) | −0.032 |
| 03-19 | 1.1925 | **Sliding window eval** stride=64 | −0.032 |
| 03-19 | 1.1748 | Muon WD + 10L + spectral embed init | −0.050 |
| 03-19 | 1.1630 | Mixed quant int6/int8 + 3x MLP + sliding eval | −0.061 |
| 03-19 | 1.1586 | 10L int6 QAT + zstd-22 + MLP 2.6x | −0.066 |
| 03-19 | 1.1556 | **SmearGate** + OrthoInit + Muon WD + int6 STE QAT | −0.069 |
| 03-20 | 1.1458 | **BigramHash** + 3x MLP + SWA | −0.079 |
| 03-20 | 1.1428 | Int5-MLP + BigramHash(10240) + SWA(0.4) | −0.082 |
| 03-20 | 1.1307 | **XSA** (last 3L, efficient GQA-aware) | −0.094 |
| 03-20 | 1.1271 | XSA (last 4L) + **EMA** replacing SWA | −0.097 |
| 03-21 | 1.1248 | **Partial RoPE** (16/64) + **LN Scale** | −0.100 |
| 03-22 | 1.1228 | **GPTQ-lite** (clip search) + EMA + warmdown3500 + QAT@0.15 | −0.102 |
| 03-23 | 1.1194 | **LeakyReLU(0.5)²** + legal score-first **TTT** + Parallel Muon | −0.105 |
| 03-25 | 1.1147 | AR self-gen **Full Hessian GPTQ** + XSA-all-11 + BigramHash 3072×112 | −0.110 |
| 03-31 | 1.1063 | **Parallel Residuals** (L7+) + **mini depth recurrence** (L4-5) + AR GPTQ | −0.118 |
| 04-01 | 1.0979 | **SP4096** vocab + MLP 4x + WD=0.085 + simplifications (removed TTT, hash embed, SmearGate, VR) | −0.127 |
| 04-03 | 1.0912 | **MuonEq-R** + depth recurrence + WD=0.090 + **all-int6** GPTQ | −0.133 |
| 04-04 | 1.0897 | SP4096 + depth recurrence + parallel residuals + MuonEq-R + QK-Gain 5.0 | −0.135 |
| 04-05 | 1.0856 | **SP8192** + GPTQ embeddings + depth recurrence + **SDClip** | −0.139 |
| 04-06 | 1.0835 | Parallel residuals + **Hessian-aware SDClip** + progressive recurrence | −0.141 |
| 04-06 | 1.0828 | QK-Gain 5.0 + legal score-first **TTT** on SP8192 | −0.142 |
| 04-08 | 1.0822 | Parallel residuals on SP8192 + score-first TTT | −0.142 |
| 04-09 | **1.0810** | **3-layer recurrence** (L3-5) + parallel residuals + **QK-Gain 5.25** + legal TTT | **−0.143** |

## 2. Anatomy of the Current SOTA (1.0810)

### Architecture
- 11 physical layers × 512d × 8H / 4KV (GQA)
- MLP 4x expansion (2048 hidden), LeakyReLU(0.5)² activation
- Partial RoPE (16 of 64 dims)
- Layerwise LN Scale: 1/√(layer+1)
- Tied token embeddings (SP8192 BPE vocab)
- Logit softcap = 30.0
- U-Net skip connections with sigmoid gates
- XSA on all 11 layers
- Parallel residuals from layer 7+ (GPT-J style)
- 3-layer depth recurrence: layers 3-5 loop, activated at frac=0.35 → 17 virtual layers

### Optimizer
- MuonEq-R (row-normalized Muon, Newton-Schulz 5 steps) for matrix params
- AdamW for embeddings/scalars
- WD=0.095 (Muon), WD=0.085 (embed), WD=0.02 (Adam)
- EMA decay=0.9965
- Warmdown frac=0.72, MLR=0.022
- ~4550 steps in 588s

### Quantization & Compression
- Full Hessian GPTQ with SDClip: `clip = k * std(row)`
- int6 for attention/MLP matrices (k=12.85), int8 for embeddings (k=20.0)
- Byte-shuffle + Brotli-11
- Self-extracting LZMA code wrapper (~16.6KB)
- ~15.99 MB total artifact

### Evaluation
- Sliding window stride=64
- Legal score-first TTT: SGD(lr=0.005, momentum=0.9), 3 epochs per 32K chunk, cosine LR decay
- ~500s eval (370s TTT + 83s sliding + 8s roundtrip)

## 3. Design Choice Taxonomy

### A. Tokenizer / Vocabulary

| Choice | When introduced | Impact | Saturation |
|--------|----------------|--------|-----------|
| SP1024 (baseline) | 03-17 | baseline | replaced |
| SP4096 | 04-01 | ~−0.012 vs SP1024 stack | replaced |
| SP8192 | 04-05 | ~−0.008 vs SP4096 stack | **current, possibly saturated** |
| Larger vocab (16k+)? | never tried | unknown — more context/step but bigger embed table | **open** |

Notes: Each vocab increase lets the model see more context per fixed seq_len. The embed table grows but GPTQ-quantized int8 embeds compress well. Diminishing returns expected; 8192→16384 is plausible but untested. The embed table at 8192×512 = 4M params is already a meaningful fraction of the budget.

### B. Model Depth / Width / Capacity

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| 9L → 10L | 03-18 | −0.003 | replaced |
| 10L → 11L | 03-20 | −0.005 | **current** |
| MLP 2x → 3x | 03-19 | −0.010 (single biggest early gain) | replaced |
| MLP 3x → 4x | 04-01 | ~−0.005 (enabled by higher WD compression) | **current** |
| 512d unchanged | — | — | appears optimal for 16MB budget |

Notes: 11L×512d with MLP 4x is now standard. Deeper models are too slow per step. Wider models tried (640d, 704d in recurrence writeup) hurt due to fewer steps. The depth/width/MLP tradeoff looks **close to saturated** at current compression tech.

### C. Depth Recurrence

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| No looping | baseline | — | — |
| Loop L4-5 (1 extra pass) | 03-31 | −0.003 | replaced |
| Loop L4-5 (2 extra) | 04-05 | ~−0.003 more | replaced |
| Loop L3-5 (3-layer, 2 extra) | 04-09 | ~−0.002 more | **current** |
| Activated mid-training (frac=0.35) | 04-09 | better than always-on | **current** |

Notes: Recurrence gives a moderate but real gain. The key insight is **mini recurrence** (a few middle layers) with **delayed activation** to avoid losing too many steps. Full universal-transformer-style recurrence is a bust (see non-record writeup: +0.025 worse in controlled comparison). The 3-layer config at current is likely near-optimal for the step-time budget. Further recurrence gains would require faster per-step time (megakernels?) or better loop configurations.

### D. Attention Innovations

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| XSA last 3L | 03-20 | −0.002 | replaced |
| XSA last 4L | 03-20 | −0.001 more | replaced |
| XSA all 11L | 04-01 | −0.001 more | **current** |
| Partial RoPE 16/64 | 03-21 | −0.002 | **current** |
| QK-Gain 1.5→4.0→5.0→5.25 | 03-19→04-09 | ~−0.003 total | **current, maybe still improvable** |

Notes: XSA on all layers is zero-parameter-cost and appears saturated. QK-Gain has been monotonically increasing; 5.25 is current. Worth trying 5.5, 6.0 but likely small returns. Partial RoPE at 16/64 has been stable since 03-21.

### E. Residual / Skip Structure

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| U-Net skip connections | baseline | baked in | **current** |
| Sigmoid-gated skips | 04-01 | small improvement over raw skips | **current** |
| Parallel residuals L7+ | 03-31 | −0.002 to −0.004 | **current** |

Notes: Parallel residuals from L7+ is a consistent, free-parameter win. The learned routing is asymmetric (MLP barely writes to attn lane). The optimal start layer and routing structure are underexplored.

### F. Optimizer

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| Muon (base) | baseline | — | replaced |
| Muon WD=0.01→0.04 | 03-19 | −0.003 | replaced |
| Muon WD=0.04→0.085 | 04-01 | −0.005 (via compression headroom) | replaced |
| Muon WD=0.085→0.095 | 04-09 | small (via more int6 headroom) | **current** |
| MuonEq-R (row normalize) | 04-03 | −0.001 | **current** |
| EMA (replacing SWA) | 03-20 | −0.001 | **current** (decay=0.9965) |
| Parallel Muon + banking | 03-23 | ~0 bpb (speed gain) | **dropped in later stacks** |

Notes: WD is deeply coupled to compression quality. The insight that higher WD → smaller weights → better brotli → more int6 layers → lower bpb is one of the single most impactful findings in the entire competition. Further WD increases face diminishing returns (underfitting). EMA decay has been tuned from 0.997 to 0.9965. MuonEq-R is a free win.

### G. Quantization & Compression

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| Int8 + zlib | baseline | — | replaced |
| Int6 QAT (STE) | 03-19 | −0.010+ (smaller artifact → more params) | replaced |
| Mixed int5/int6 | 03-20 | small (freed space for 10th layer) | replaced |
| GPTQ-lite (clip search) | 03-22 | −0.001 | replaced |
| Full Hessian GPTQ | 03-25 | −0.005 (strictly better quantizer) | **current** |
| zstd-22 → Brotli-11 | 04-01 | better compression ratio | **current** |
| SDClip (k * std(row)) | 04-05 | principled, faster, more tunable | **current** |
| GPTQ embeddings (int8) | 04-05 | small (vs fp16 or naive round) | **current** |
| All-int6 (via WD headroom) | 04-03 | −0.001 | **current** |
| Hessian-aware SDClip | 04-06 | ~−0.002 (non-record) | **promising, underexplored** |
| Byte-shuffle before Brotli | 04-01 | ~5% compression improvement | **current** |

Notes: Quantization + compression is the single richest vein in this competition. The interaction between WD, clip range, bit width, and post-quant compression is deeply non-obvious (Kevin Clark's SDClip writeup is the key reference). **Per-group or per-layer k allocation is the most obvious next step** — the Hessian-SDClip writeup shows stable group-level importance across seeds. Output-Hessian weighting and layer-adaptive bit allocation are also open.

### H. Evaluation-Time Techniques

| Choice | When | Impact | Saturation |
|--------|------|--------|-----------|
| Non-overlapping eval | baseline | — | replaced |
| Sliding window stride=64 | 03-19 | −0.032 (!!!) | **current** |
| LoRA TTT (per-doc) | 03-19 | −0.004 (on top of sliding) | replaced |
| Legal score-first TTT (SGD, all params) | 03-23 | −0.002 to −0.003 | **current** |
| TTT dropped (04-01 stack) | 04-01 | reclaimed eval budget for systems wins | TTT came back 04-06 |
| TTT on SP8192 stack | 04-06 | −0.002 | **current** |

Notes: Sliding window was the single largest individual improvement in the whole competition. TTT adds a consistent −0.002 on top. The current TTT is simple full-weight SGD. **Selective TTT** (low-rank, high-perplexity focus, learning-rate scheduling per layer) is underexplored.

### I. Removed / Abandoned Techniques

| Technique | Why removed |
|-----------|------------|
| SmearGate | redundant with larger vocab (SP4096+) |
| BigramHash | removed in SP4096+ stack (replaced by larger vocab semantics) |
| Value Residuals | incompatible with depth recurrence (+0.14 worse) |
| Gated Attention | no clear benefit |
| QAT (STE) | torch.compile dead-code-eliminated it; little or no benefit anyway |
| Parameter Banking | complexity for no bpb gain |
| SWA | replaced by EMA |

## 4. Search Space Map

Dimensions ordered by estimated remaining headroom:

### Tier 1: High headroom, strong evidence
1. **Quantization: per-group/per-layer clip allocation** — Hessian traces are 30x different between early and late blocks but k is global. Non-uniform k is the most obvious untried thing.
2. **TTT improvements** — Current is naive full-weight SGD. Low-rank adapters, perplexity-weighted chunk selection, per-layer LR, or momentum tuning all open.
3. **Vocab size 8192 → 16384** — More context per step, larger embed table (but GPTQ-quantized embeds compress). Diminishing returns expected but delta could be 0.002-0.005.
4. **Warmdown / LR schedule tuning** — Warmdown frac went from 0.3 to 0.72; precise shape matters. Cosine vs linear unexplored.

### Tier 2: Moderate headroom, needs validation
5. **QK-Gain > 5.25** — Monotonic improvement so far. Try 5.5, 6.0, per-layer gains.
6. **Loop config tuning** — 3-layer vs 4-layer, different layer ranges, asymmetric repeats.
7. **Parallel residual topology** — Start layer, learned routing weights, lane merge strategy.
8. **Training sequence length > 2048** — 4096 helped early (−0.02) but was replaced by other wins. On the current stack, longer seq + sliding eval could still help.
9. **EMA decay tuning** — 0.9965 is current; the space between 0.996 and 0.998 is underswept.
10. **Better activation function** — LeakyReLU(0.5)² is standard; GEGLU, SwiGLU, or a learned negative slope are untried on the top stack.

### Tier 3: Speculative, larger changes
11. **Hybrid SSM/attention** — Replace some late blocks with Mamba-style SSM. No one has tried.
12. **H-Net dynamic chunking** — Replace the fixed tokenizer with learned segmentation. High-upside moonshot.
13. **Multi-token prediction** — MTP with 4 heads / weight 0.3 showed promise in the recurrence writeup but was never tested on the top stack.
14. **JEPA for text** — Non-AR training signal. Research lane, not immediate record bet.
15. **Megakernel fusion** — Fuse attention+MLP into fewer kernel launches. Pure systems win for more steps.

## 5. Stack-Ranked Research Ideas

Priority order for incremental record submissions on the current SP8192 stack.

### Experiment 1: Per-group SDClip allocation
- Use the stable group-level Hessian trace hierarchy (early >> loop >> mid >> late) to set different k values per block group.
- Expected gain: 0.001-0.003 bpb (tighter allocation recovers some quant error in sensitive early layers while spending less entropy budget on less sensitive late ones).
- Risk: low. Zero training cost, just a quant-time change.
- Cost: 1 run to sweep k per group.

### Experiment 2: SP16384 vocabulary
- Train a SP16384 tokenizer on the FineWeb corpus. Each token covers more bytes → more context per 2048-token window.
- Expected gain: 0.002-0.005 bpb (extrapolating from SP1024→SP4096→SP8192 trend, with diminishing returns).
- Risk: medium. Embed table grows to 16384×512 = 8.4M params; need to verify it still fits in 16MB with int8 GPTQ embeddings. Tokenizer changes are scrutinized.
- Cost: 2-3 runs to validate.

### Experiment 3: Selective TTT with low-rank adapters
- Instead of updating all params with SGD, update only a rank-4 or rank-8 adapter on q/v projections.
- Spend TTT epochs only on high-perplexity chunks (score all chunks first, then train on the worst ones).
- Expected gain: 0.001-0.002 bpb (more targeted adaptation + less catastrophic forgetting on good chunks).
- Risk: low-medium. Need to verify adapters are compatible with quantized weights.
- Cost: 3-5 runs.

### Experiment 4: QK-Gain sweep (5.5, 6.0, per-layer)
- The trend from 1.5 → 4.0 → 5.0 → 5.25 has been monotonically improving.
- Try 5.5, 6.0, and per-layer learned gains (different init per layer depth).
- Expected gain: 0.0005-0.001 bpb.
- Risk: very low. Single hyperparameter change.
- Cost: 1 run per value.

### Experiment 5: Warmdown shape tuning
- Warmdown frac at 0.72 is aggressive. Try cosine warmdown instead of linear.
- Sweep warmdown_frac in [0.65, 0.70, 0.75, 0.80].
- Try a two-phase warmdown (fast decay then slow tail).
- Expected gain: 0.0005-0.001 bpb.
- Risk: very low.
- Cost: 4 runs.

### Experiment 6: Layer-adaptive bit allocation
- Instead of all-int6, try int7 on the most sensitive layers (early blocks) and int5 on the least sensitive (late blocks). SDClip math shows int7 with 2x k has similar compressed size.
- Expected gain: 0.001-0.002 bpb.
- Risk: low-medium. Need to verify artifact stays under 16MB.
- Cost: 2-3 runs.

### Experiment 7: Training seq length 4096
- The current stack trains at 2048. Doubling to 4096 halves throughput but gives the model more context.
- Combine with gradient accumulation adjustments to maintain batch tokens.
- Expected gain: 0.002-0.005 bpb (based on the early 03-19 result, but the effect may compound differently on the current stack).
- Risk: medium. Step time doubles; may not get enough training steps.
- Cost: 2-3 runs.

### Experiment 8: Multi-token prediction on current stack
- MTP with 4 heads / weight 0.3 was optimal on the recurrence stack.
- Never tested on the SP8192 + parallel residual + TTT stack.
- Expected gain: 0.001-0.003 bpb (auxiliary loss may improve representation quality without extra eval cost).
- Risk: medium. MTP auxiliary heads add small param overhead and need to verify compile compatibility.
- Cost: 2-3 runs.

### Experiment 9: Loop config refinement
- Current: layers 3-5, 2 extra passes, activated at frac=0.35.
- Try: layers 2-5 (4-layer loop, 1 extra pass), or layers 3-6.
- Try progressive activation: first loop at 0.35, second at 0.50.
- Expected gain: 0.0005-0.001 bpb.
- Risk: low.
- Cost: 3-4 runs.

### Experiment 10: EMA + late SWA combination
- EMA provides continuous smoothing; adding a few late SWA checkpoints during the final 5% of training may further smooth the weight landscape.
- SWA on the recurrence stack inverted the quant gap in some cases.
- Expected gain: 0.0005-0.001 bpb.
- Risk: very low.
- Cost: 1-2 runs.

### Summary: Recommended execution order

| Order | Experiment | Expected Δ | Runs | Risk |
|-------|-----------|-----------|------|------|
| 1 | Per-group SDClip | 0.001-0.003 | 1-2 | low |
| 2 | QK-Gain sweep | 0.0005-0.001 | 2-3 | very low |
| 3 | Warmdown shape | 0.0005-0.001 | 4 | very low |
| 4 | SP16384 vocab | 0.002-0.005 | 3 | medium |
| 5 | Selective TTT | 0.001-0.002 | 3-5 | low-medium |
| 6 | Layer-adaptive bits | 0.001-0.002 | 2-3 | low-medium |
| 7 | MTP on current stack | 0.001-0.003 | 2-3 | medium |
| 8 | Training seq 4096 | 0.002-0.005 | 2-3 | medium |
| 9 | Loop config | 0.0005-0.001 | 3-4 | low |
| 10 | EMA + SWA combo | 0.0005-0.001 | 1-2 | very low |

The first 3 experiments are cheap, low-risk, and compound with each other. Experiments 4-6 require more investment but have the largest potential individual deltas. Experiments 7-10 are refinements best done after the high-conviction changes are locked in.

A new record needs ≥0.005 nats improvement over 1.0810. That's roughly ≥0.003 bpb. Getting there likely requires stacking 2-3 of the above ideas rather than any single change.

# Research Notes

## Regime change: 20 min wallclock + sliding window eval

Starting from iter_7, we're running 20 min wallclock (was 10 min) with sliding window eval re-enabled. TTT stays off for now (adds ~370s for ~0.002 bpb — save for final validation). On the Brev H100 PCIe (~680 tok/s), this gives ~780 steps (was ~390). On RunPod SXM (~840 tok/s), it would give ~1200 steps. The SOTA used 8xH100 SXM for ~4550 steps in 10 min.

The doubling of steps should make EMA/optimizer experiments viable again and improve compression quality (closer to 16MB). Sliding window historically gives −0.032 bpb — we'll now measure this directly.

## Experimental findings (iter_0 through iter_6)

### What worked
| Experiment | Δ quantized_val_bpb | Notes |
|-----------|-------------------|-------|
| Per-group SDClip (iter_1) | −0.018 | Biggest single win. Early layers k×1.15, mid unchanged, late k×0.88 |
| QK-Gain 5.25→5.5 (iter_2) | −0.002 | Monotonic trend continues |
| QK-Gain 5.5→6.0 (iter_3) | −0.001 | Still improving, not saturated |
| Cosine warmdown (iter_4) | −0.001 | Smoother decay > linear cliff |

### What failed
| Experiment | Δ quantized_val_bpb | Notes |
|-----------|-------------------|-------|
| EMA decay 0.9965→0.995 (iter_5) | +0.104 | Catastrophic post-EMA degradation |
| warmdown_frac 0.72→0.80 (iter_6) | +0.333 | Even worse; model barely trains at full LR |

### Key insight: post-EMA quality is decoupled from in-training loss
All runs showed nearly identical in-training val_bpb (~1.21–1.28) but post-EMA ranged from 1.356 (good) to 1.688 (terrible). The EMA model averages over a trajectory whose shape depends heavily on schedule params. Small perturbations to EMA decay or warmdown can route the trajectory into completely different basins, and this only manifests after EMA weight substitution — not visible during training. With 2x steps this sensitivity should diminish, but we should still prefer small incremental changes to schedule params.

### Hardware note: PCIe vs SXM throughput
Brev H100 PCIe gives ~680 tok/s (390 steps / 10 min) vs RunPod SXM ~840 tok/s (606 steps / 10 min). This is a 36% throughput gap. At 20 min on PCIe we get ~780 steps, roughly matching old SXM 10-min runs. Results from different hardware should be compared cautiously — fewer steps means undertrained model, worse EMA, worse compression.

## Revised experiment priorities (post-regime-change)

### High confidence — run next
1. **Re-baseline at 20 min + sliding window** (iter_7). Critical to establish the new reference point before testing changes. Expect a significant improvement from 2x steps and sliding window alone.

2. **QK-Gain 7.0** (iter_8). The 5.25→5.5→6.0 trend was monotonically improving. Push further. If still improving, try 8.0.

3. **Finer SDClip: per-layer k** (iter_9). Our 3-group scheme (early/mid/late) gave −0.018. Per-layer allocation using Hessian trace ratios could squeeze another 0.001–0.003. The Hessian data in the SOTA write-up shows traces vary 30x across layers — 3 groups is coarse.

### Medium confidence — revisit with more steps
4. **EMA decay sweep (conservative)** — try 0.997, 0.996 (small moves from 0.9965). With 2x steps, the EMA trajectory is smoother and should tolerate small changes. But after the iter_5 disaster, move in 0.0005 increments only.

5. **warmdown_frac sweep (conservative)** — try 0.68, 0.70 (both *less* warmdown than 0.72, opposite direction from the failed 0.80). The intuition: with 2x total steps, the absolute number of warmdown steps is already doubled; we may not need such a large *fraction*.

6. **Muon WD 0.095→0.11** (originally planned). Higher WD → smaller weights → better brotli compression. This was always the plan; delayed due to EMA sensitivity concerns. With more steps it should be testable.

7. **GPTQ calibration batches 64→128** — more calibration data for the Hessian computation should improve quantization quality. Zero training cost, only ~10s extra quant time.

### New ideas (from the regime change)
8. **Exploit the step-count increase for model capacity** — with ~780 steps instead of ~390, we can train a slightly larger model that still converges. Options:
   - MLP 4x → 4.5x (small capacity bump, ~3M more params)
   - 12 layers instead of 11 (need to verify step-time impact)
   - These need to still fit in 16MB after GPTQ, so careful sizing is needed.

9. **Sliding window stride tuning** — the default is stride=64. Now that sliding window is on, try stride=32 for a richer eval at the cost of ~2x eval time. Or stride=128 to save time if the bpb delta is small.

10. **Two-phase warmdown** — instead of a single cosine curve, try: fast decay (cosine over first 60% of warmdown) to 0.1, then a slow linear tail to min_lr. This keeps the model exploring longer at moderate LR before the final convergence.

11. **Grad accumulation reduction** — current `grad_accum_steps=8` with `train_batch_tokens=786432`. If we halve to `grad_accum_steps=4`, we double the number of optimizer steps (noisier gradients, but 2x the step count). On 1xH100 this might let us reach ~1500 steps in 20 min. The trade-off: noisier gradients vs more updates. Worth testing.

### Deprioritized / deferred
- **SP16384 vocab** — risky, large embed table, probably needs TTT re-enabled to see the full benefit. Defer until we've locked in quant/schedule gains.
- **MTP (multi-token prediction)** — interesting but architectural complexity; save for later.
- **Hybrid SSM/attention** — speculative; not worth the iteration cost now.
- **TTT improvements** — can't validate without TTT enabled. Defer to final testing phase.

## Proposed iteration plan

| Iter | Experiment | Result |
|------|-----------|--------|
| iter_7 | 20 min baseline + sliding window | 1.2464 (new ref) |
| iter_8 | QK-Gain 6.0→7.0 | +0.002 REVERT |
| iter_8b | QK-Gain 6.0→6.5 | +0.0002 REVERT (QK saturated at 6.0) |
| iter_9 | Per-layer SDClip (11-vector) | flat bpb, −30KB size, **KEEP** |
| iter_10 | GPTQ calibration 64→128 | flat REVERT |
| iter_11 | warmdown_frac 0.72→0.68 | −0.0002 KEEP (fragile) |
| iter_12 | Muon WD 0.095→0.11 | +0.003 REVERT |
| iter_13 | embed_clip_sigmas 20→15 | +230KB size, entropy up REVERT |
| iter_14 | Steeper SDClip (1.22→0.65) | +247KB size REVERT |
| iter_15 | matrix_clip_sigmas 12.85→14.0 | 341KB under but +0.001 bpb, superseded |
| iter_15b | matrix_clip_sigmas 12.85→13.5 | **FIRST PASS 16MB**, best_bpb 1.2463, **KEEP** |
| iter_16-19 | train_batch_tokens sweep 786K→196K | see note below |
| iter_20 | warmdown_frac 0.68→0.60 | +0.003 REVERT |

## Course correction: step-count-sensitive wins don't transfer

Iters 16-19 revealed that the dominant lever on 1xH100 PCIe is batch size. Reducing `train_batch_tokens` from 786K (SOTA default) down to 262K nearly doubled optimizer steps (790→2172) and gave a massive −0.080 bpb cumulative gain (best 1.1659). Iter_19 at 196K hit the crossover point (+0.001 regression from gradient noise).

**These wins are artifacts of 1xH100 PCIe throughput and will NOT transfer.** On the 8xH100 SXM submission target, the SOTA already gets ~4550 steps in 10 min at 786K batch. Reducing batch to our 262K "optimum" would push that to ~10-13k steps — far past any sweet spot. The right batch size on 8xH100 is probably close to the SOTA default, not ours.

Same concern applies to EMA decay, warmdown_frac, and any schedule hyperparameter tuned at a particular step count. The iter_11 warmdown 0.68 win was tiny (−0.0002 at 790 steps); whether it helps at 4550 steps is unknown.

**Decision: revert iters 16-21 (non-transferable) and re-focus on hardware-independent experiments.**

Kept (transferable) changes so far:
- iter_9: per-layer SDClip (quantization — math-only, hw-independent)
- iter_15b: matrix_clip_sigmas 13.5 (quantization — math-only, hw-independent)
- iter_11: warmdown_frac 0.68 (schedule — fragile at 790 steps, marginal; keep but flag)

## Revised plan: hardware-independent improvements only

Categories that transfer across training regimes:

**Quantization / compression (dev-friendly, cheap, math-only):**
- GPTQ block size / ordering variants
- Per-head SDClip (not just per-layer) — Q/K/V heads have different sensitivities
- int5/int7 on specific sensitive layers (first attention, last MLP)
- Alternative compressor tuning (brotli-11 vs brotli-11 -W 24, zstd-22 long mode)
- Byte-shuffle stride variations (current: 2 bytes? tune for the quantized tensor layout)
- Mixed-bit per-layer allocation (int6 early, int5 late)

**Architecture (transfers directly, may need size-budget balancing):**
- 11 layers → 12 layers (check if still fits in 16MB with int6)
- MLP expansion 4x → 3.5x or 4.5x (bigger ≠ better under quant)
- Shared MLP weights between loop iterations (capacity doubling for free)
- Depth-wise weight tying experiments
- Different attention QK-head ratio

**Optimizer / init (step-count-agnostic at first order):**
- Newton-Schulz 5→6 iterations (more accurate Muon update)
- Different init std on tied_embed / tok_emb
- QK-Gain initialization schedule (linear ramp during warmup)

**Eval-only (don't change training, only scoring pipeline):**
- Sliding window stride tuning (64→32, 128)
- GPTQ hessian with importance-weighted calibration

## Re-ranking original ideas by transferability

Re-reading §4/§5 through the hardware-independence lens (what survives going from 1xH100 PCIe @ ~1.4k steps to 8xH100 SXM @ ~4.5k steps?):

- **Step-count-sensitive** (hold for final 8xH100 tuning — do NOT burn dev cycles here):
  warmdown shape, EMA decay, SWA timing, `train_batch_tokens`, any LR schedule detail.
  These are first-order functions of total steps. Dev-time wins are anti-signal.
- **Step-count-agnostic / transferable** (burn dev cycles here):
  quantization math, compression, attention/MLP structure, optimizer update rule (NS iters, WD),
  init std, vocab size, MTP auxiliary heads, GPTQ calibration design.

Re-ranked candidates from §4/§5 that were not yet executed:

| Old rank | Idea | Transferable? | New rank | Why moved |
|----|----|----|----|----|
| T1.3, §5-E1 | Hessian-aware SDClip (per-row k from H trace) | Yes — pure math | **1** | Writeup calls it "promising, underexplored" (~−0.002). Underexplored, hw-indep, cheap. |
| §5-E6 | Layer-adaptive bit allocation (int5/int6/int7 mix) | Yes — quant math | **2** | Strong Hessian evidence for heterogeneity; direct size/quality trade. |
| T1.2, §3-G | Per-head SDClip (Q/K/V rows treated separately) | Yes — quant math | **3** | Natural extension of iter_9's per-layer win; heads within a layer differ. |
| §5-E8 | MTP (4 heads, weight 0.3) | Yes — arch/loss | **4** | §4 Tier 3; "showed promise but never tested on top stack." Promoted — no step-count coupling. |
| §5-E4 | QK-Gain sweep (5.5 vs 6.0 vs per-layer) | Yes — init | **5** | Cheap, 1-commit; iter_7 already validated sensitivity direction. |
| T1, §3-G | Newton-Schulz 5→6 steps | Yes — optimizer | **6** | Better orthogonalization; cost is compile time only. |
| §5-E2 | SP16384 vocabulary (6144 → 16384) | Yes — tokenizer | **7** | Highest single-lever potential (−0.003 to −0.008) but expensive: retokenize, rebuild BigramHash, re-fit sizes. Fire once other knobs are tuned. |
| §5-E9 | Loop config (depth/width/iters within recurrence) | Yes — arch | **8** | Structural; cheap to try once we have a stable baseline. |
| §5-E7 | Training seq length 4096 | Partial — changes grad noise | **9** | Arguably step-count-sensitive via effective batch; defer. |
| §5-E3 | Selective TTT with low-rank adapters | Yes — eval-time | **10** | Eval-only so fully safe, but TTT is disabled in dev for iteration speed. Revisit pre-submission. |
| §5-E5, §5-E10 | Warmdown shape / EMA+SWA | **No** — step-sensitive | — | Explicitly deferred to final 8xH100 runs. |

## Next experiments (transferable)

| Iter | Experiment | Category | Expected | Result |
|------|-----------|----------|----------|--------|
| iter_22/a-d | Hessian-aware SDClip (H-weighted per-row std) | Quantization | −0.001–0.003 | **REVERT** — neutral-to-negative at matched size (see note) |
| iter_23/b/c | Mixed-bit per-layer | Quantization | −0.001–0.003 | **REVERT** — int5 cost > int7 gain |
| iter_24 | Per-role SDClip (c_v +15%, c_q/c_k -5%, attn.proj +5%) | Quantization | −0.001–0.002 | **KEEP** (−0.00017, smaller too) |
| iter_25/b | MTP 4 heads, weight 0.3 | Architecture | −0.001–0.003 | **REVERT** — +0.11 bpb, MTP overhead kills step count (see note) |
| iter_26/b | QK-Gain per-layer init schedule | Init | −0.001–0.002 | **REVERT** — +0.0006 (ramp up) / +0.0016 (ramp down) |
| iter_27/b | Newton-Schulz 5→6 / 5→4 | Optimizer | −0.000–0.002 | **REVERT** — 6 +0.0043, 4 +0.0019 (NS=5 is a local min) |
| iter_28 | SP16384 vocabulary rebuild | Tokenizer | −0.003–0.008 (biggest but priciest) | pending |
| iter_29+ | Loop config refinement / based on wins above | Architecture | TBD | pending |

### iter_22 notes (H-aware SDClip, four sub-iters)

Replaced row-wise `std(dim=1)` with a Hessian-weighted row std:
`row_std_h = sqrt( Σ_j h_weight[j] (W[i,j] − μ_h)^2 / Σ h_weight )`
where `h_weight[j] = H[j,j] / mean(H.diag())`.

Size vs `matrix_clip_sigmas` (all H-weighted, iter_15b baseline is `std`-based at 13.5 → 15.84MB):
- c_s=12.5 → 16.62MB (smaller c_s ⇒ saturated ints ⇒ WORSE compression — counterintuitive but real)
- c_s=13.5 → 16.21MB
- c_s=14.0 → 16.03MB (29KB over)
- c_s=14.5 → 15.84MB ✓ under cap, bpb **1.2470** vs iter_15b **1.2463** (+0.0007)

Learnings:
1. Smaller `clip_sigmas` → saturated int distribution → worse brotli compression. Intuition (more clipping = smaller ints = better compression) is wrong; what matters is the int histogram's entropy at large values.
2. H-weighted per-row std is **redundant with GPTQ**: the GPTQ loop already propagates Hessian-weighted error across columns via the Hinv update. Adding H weighting to the scale formula double-counts.
3. Best version (c_s=13.5, 16.21MB) tied iter_15b on quality — so the H-weighting is quality-neutral, not additive to GPTQ.
4. **If revisited:** try H weighting **instead of** GPTQ's ordering/Hinv (replace the mechanism, don't stack it), or move to group-level rather than row-level H weighting.

Reverted to iter_15b baseline.

### iter_24 notes (per-role SDClip — KEPT)

Added `_ROLE_K_MUL` that adjusts `clip_sigmas` by tensor role on top of the per-layer multiplier. Hypothesis: c_v directly transforms values (outliers matter more than in softmax-filtered c_q/c_k), and attn.proj is on the residual path (sensitive).

Initial values:
- c_v: 1.15x, attn.proj: 1.05x (wider, preserve outliers)
- c_q: 0.95x, c_k: 0.95x (tighter, softmax robust to small errors)
- mlp.fc / mlp.proj: 1.00x (unchanged)

Result: −0.00017 bpb at slightly smaller size (15.83 vs 15.84MB). Small win but transferable.

**Follow-ups (for post-iter_28 deep dive):**
- Sweep c_v multiplier in [1.05, 1.10, 1.15, 1.20, 1.25] — 1.15 was a guess
- Try per-head-within-c_v (8 heads, separate k each) — requires grouped row_std per head
- MLP roles were set to 1.0 without test — sweep them too
- Combine with per-layer K_MUL re-tuning (current K_MUL was tuned with uniform role multipliers)

### iter_23 notes (mixed-bit per-layer, three sub-iters)

Added `_tensor_bits(name, h)` that returns different bit widths per layer. Size delta per layer ≈ ±360KB for int7/int5 vs int6 (measured empirically).

| Sub-iter | Allocation | Size | best_bpb | Δ vs iter_15b |
|---|---|---|---|---|
| iter_23 | int7: 0-1 (2 lyr), int5: 8-10 (3 lyr) | 15.48MB | 1.247650 | +0.0013 |
| iter_23b | int7: 0-2 (3 lyr), int5: 10 (1 lyr) | 16.56MB (over) | 1.246103 (pre-size-cap) | −0.0002 |
| iter_23c | int7: 0-1 (2 lyr), int5: 9-10 (2 lyr) | 15.84MB (size-matched) | 1.246765 | +0.0004 |

Learnings:
1. **int5 has a sharper quality cliff than int7's gain**. At matched size (iter_23c), dropping 2 late layers to int5 costs more than bumping 2 early layers to int7 buys back.
2. **Size-cost of one layer's bit change is ~360KB** after compression — makes the precision budget very tight.
3. iter_23b (heavy int7) showed the quality win exists (−0.0002) but only at 16.56MB — the cap kills it.
4. **If revisited:** try (a) **asymmetric early-only int7** combined with `clip_sigmas` tightening (not just bit change) to recover size, (b) int6→int7 on attention only (fewer tensors, cheaper), (c) attempt layer 0 int8 like tok_emb (highest sensitivity per K_MUL), and/or (d) try **per-head bit allocation** within attention.

Reverted to iter_15b baseline.

### iter_25 notes (MTP 4 heads, weight 0.3 — REVERTED)

Added 4 extra `CastedLinear(model_dim, model_dim)` heads that project the final hidden state to shifted-by-k targets (k=1..4). Auxiliary loss = `(mtp_weight / K) * Σ_k CE(project(head_k(x))[:-k], targets[k:])`. MTP heads gated on `self.training` so validation is main-loss only; serialization filters `mtp_heads.*` so on-disk weights are unchanged (heads are training-only scaffolding).

| Sub-iter | Fix | steps | pre-EMA val_bpb | quant_val_bpb | Decision |
|---|---|---|---|---|---|
| iter_25 (first run) | eval still counted MTP aux → inflated val | 598 | 2.096 | 1.3579 | broken eval |
| iter_25b | `self.training` gate fixes eval | 598 | 1.367 | **1.3572** | **REVERT** |

Baseline (iter_24): ~790 steps, quant_val_bpb ≈ 1.2461. MTP run: 598 steps, 1.3572 → **+0.111 bpb regression**.

Learnings:
1. **MTP pays in training time, not bpb** at this budget. Four extra projections per forward cut tok/s from ~580K to ~490K; at 10 min wallclock that's ~24% fewer optimizer steps, which alone costs ~0.02 bpb (extrapolating from iter_16-19 step-count scaling).
2. Pre-EMA val (1.367) is only slightly worse than post-quant (1.357) → EMA actually *recovers* some of the loss, so the under-trained checkpoint is the main culprit, not the MTP signal per se.
3. **If revisited:** (a) reduce K to 2 heads, (b) only activate MTP for the last fraction of training (`frac > 0.6`) so it doesn't starve early steps, (c) share a single projection across all k-offsets via a learned offset embedding, (d) try smaller-dim heads (d/2) to halve FLOPs, (e) only test MTP on a hardware regime where step count is less binding (8xH100 SXM final runs).
4. Worth noting the on-disk weights logic worked cleanly — the filter-in-serialize pattern means we can freely experiment with auxiliary-only training signals without paying size.

Reverted to iter_24 baseline (per-role SDClip kept).

### iter_26 notes (per-layer QK-Gain init schedule — REVERTED)

`q_gain` is a learnable per-head parameter, but all layers share init `qk_gain_init=6.0`. Hypothesis: depth changes optimal attention sharpness, so layer-varying init could land in a better basin.

Threaded `layer_idx` + `num_layers` into `CausalSelfAttention` and initialized `q_gain[L] = qk_gain_init * (a + b * L/(N−1))`.

| Sub-iter | Schedule (L0→L10) | quant_sliding_window_val_bpb | Δ vs iter_24 |
|---|---|---|---|
| iter_26 | 0.75x → 1.25x (up ramp, shallow → sharp) | 1.246768 | +0.00061 |
| iter_26b | 1.15x → 0.85x (down ramp, sharp → shallow) | 1.247734 | +0.00158 |

Learnings:
1. **Symmetric regression in both directions** — this is the cleanest signal that uniform init at 6.0 already sits in a reasonable basin that diverging from (in either direction, with 11 layers of gradient flow to find per-layer adjustments) makes worse.
2. The per-head `q_gain` is learnable, so the schedule is only an init perturbation. At 790 steps the model doesn't fully undo a bad init.
3. The up-ramp was slightly less harmful than down-ramp — *weakly* consistent with the intuition that deeper layers want sharper attention — but the magnitude is too small relative to noise.
4. **If revisited:** (a) try a non-linear (concave/convex) schedule centered on 6.0, (b) try fixing early layers at a **different value** rather than a ramp (e.g. L0 only at 7.5), (c) combine with per-head gain differentiation *within* a layer (outlier heads given different init), (d) reconsider after Newton-Schulz iters (iter_27) changes the optimizer's ability to move q_gain quickly.

Reverted to iter_24 baseline.

### iter_27 notes (Newton-Schulz steps 5 → 6 / 4 — REVERTED)

`zeropower_via_newtonschulz5` uses fixed cubic coefficients `(a,b,c) = (3.4445, -4.775, 2.0315)` from Keller Jordan's Muon reference — tuned specifically for 5 iterations (the polynomial sequence was optimized so 5 applications approximate SVD-whitening on expected singular-value distributions).

| Sub-iter | NS steps | quant_sliding_window_val_bpb | Δ vs iter_24 |
|---|---|---|---|
| iter_27 | 6 | 1.250480 | +0.00432 |
| iter_27b | 4 | 1.248019 | +0.00186 |

Learnings:
1. **NS=5 is a local minimum** — both +1 and −1 step regress, consistent with the coefficients being over-fit to exactly 5 iterations (adding a step overshoots past orthogonal; dropping a step under-whitens).
2. The 6-step hit (+0.00432) is worse than the 4-step hit (+0.00186), suggesting the polynomial blows up faster when iterated past its design point than it fails to converge when truncated early.
3. **If revisited:** this is a "big idea" whose win requires **re-deriving the (a,b,c) coefficients for the new step count** (via the Lagrangian polynomial fit from Jordan's writeup). Plain step-count tuning without coef retuning is fundamentally the wrong experiment.
4. Could also try **double-precision in the quadratic update** (some users report fp64 A@A helps more than extra steps), or a heterogeneous mix — NS=6 only for the biggest tensors where whitening matters most.

Reverted to iter_24 baseline.

Deferred to final 8xH100 tuning (do NOT run in dev): `train_batch_tokens`, warmdown_frac shape, EMA decay, SWA, LR schedule tails.